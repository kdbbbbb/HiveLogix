#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HiveLogix — Phase 7 Shared PPO-Lite model.

说明：
  - 当前实现以 torch 为运行前提，但模块导入本身不强依赖 torch；
  - actor 只消费 ObservationBatch + action_mask；
  - critic 额外消费 CriticBatch。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


try:  # pragma: no cover - 本地环境可能未安装 torch
    import torch
    import torch.nn.functional as F
    from torch import Tensor, nn
except ImportError:  # pragma: no cover
    torch = None
    F = None
    Tensor = Any
    nn = None


if torch is not None:  # pragma: no cover

    @dataclass(frozen=True)
    class PolicyForwardOutput:
        root_branch_logits: Tensor
        order_logits: Tensor
        mode_logits: Tensor
        value: Tensor


    class SharedPPOActorCritic(nn.Module):
        """Phase 7 最小可训练的 Shared PPO-Lite。"""

        def __init__(
            self,
            *,
            uav_feat_dim: int,
            order_feat_dim: int,
            infra_feat_dim: int,
            history_feat_dim: int,
            critic_order_feat_dim: int,
            critic_uav_feat_dim: int,
            critic_station_feat_dim: int,
            critic_plan_feat_dim: int,
            critic_sys_feat_dim: int,
            d_model: int = 128,
            ff_dim: int = 256,
            lstm_hidden: int = 128,
            lstm_layers: int = 1,
        ) -> None:
            super().__init__()
            self.d_model = int(d_model)
            self.lstm_hidden = int(lstm_hidden)
            self.lstm_layers = int(lstm_layers)
            self._lstm_device_override: torch.device | None = None

            self.uav_proj = nn.Linear(uav_feat_dim, d_model)
            self.order_proj = nn.Linear(order_feat_dim, d_model)
            self.infra_proj = nn.Linear(infra_feat_dim, d_model)
            self.history_proj = nn.Linear(history_feat_dim, d_model)

            self.history_encoder = nn.LSTM(
                input_size=d_model,
                hidden_size=lstm_hidden,
                num_layers=lstm_layers,
                batch_first=True,
            )
            self.recurrent_core = nn.LSTM(
                input_size=d_model,
                hidden_size=lstm_hidden,
                num_layers=lstm_layers,
                batch_first=True,
            )
            self.context_proj = nn.Sequential(
                nn.Linear(d_model * 5 + lstm_hidden, ff_dim),
                nn.ReLU(),
                nn.Linear(ff_dim, d_model),
                nn.ReLU(),
            )
            self.recurrent_proj = nn.Linear(lstm_hidden, d_model)

            self.root_head = nn.Linear(d_model, 2)
            self.order_head = nn.Sequential(
                nn.Linear(d_model * 2, ff_dim),
                nn.ReLU(),
                nn.Linear(ff_dim, 1),
            )
            self.mode_head = nn.Sequential(
                nn.Linear(d_model * 2, ff_dim),
                nn.ReLU(),
                nn.Linear(ff_dim, 2),
            )
            critic_token_dim = d_model * 4 + critic_plan_feat_dim + critic_sys_feat_dim
            self.critic_order_proj = nn.Linear(critic_order_feat_dim, d_model)
            self.critic_uav_proj = nn.Linear(critic_uav_feat_dim, d_model)
            self.critic_station_proj = nn.Linear(critic_station_feat_dim, d_model)
            self.critic_head = nn.Sequential(
                nn.Linear(critic_token_dim, ff_dim),
                nn.ReLU(),
                nn.Linear(ff_dim, ff_dim),
                nn.ReLU(),
                nn.Linear(ff_dim, 1),
            )

        def enable_lstm_cpu_fallback(self) -> None:
            """Run LSTM modules on CPU while the rest of the model may use MPS."""

            self.history_encoder.to("cpu")
            self.recurrent_core.to("cpu")
            self._lstm_device_override = torch.device("cpu")

        def forward(
            self,
            *,
            observation_batch: Any,
            action_mask: Any,
            critic_batch: Any,
            lstm_state: tuple[Tensor, Tensor] | None = None,
        ) -> tuple[PolicyForwardOutput, tuple[Tensor, Tensor]]:
            policy_out, next_lstm_state = self.forward_sequence(
                observation_batch=observation_batch,
                action_mask=action_mask,
                critic_batch=critic_batch,
                lstm_state=lstm_state,
            )
            return (
                PolicyForwardOutput(
                    root_branch_logits=policy_out.root_branch_logits.squeeze(1),
                    order_logits=policy_out.order_logits.squeeze(1),
                    mode_logits=policy_out.mode_logits.squeeze(1),
                    value=policy_out.value.squeeze(1),
                ),
                next_lstm_state,
            )

        def forward_sequence(
            self,
            *,
            observation_batch: Any,
            action_mask: Any,
            critic_batch: Any,
            lstm_state: tuple[Tensor, Tensor] | None = None,
        ) -> tuple[PolicyForwardOutput, tuple[Tensor, Tensor]]:
            device = self.uav_proj.weight.device
            lstm_device = self._lstm_device_override or device
            uav_self = _ensure_sequence_dim(
                _to_float_tensor(observation_batch.uav_self_token, device=device),
                single_rank=1,
                name="observation_batch.uav_self_token",
            )
            order_tokens = _ensure_sequence_dim(
                _to_float_tensor(observation_batch.order_tokens, device=device),
                single_rank=2,
                name="observation_batch.order_tokens",
            )
            infra_tokens = _ensure_sequence_dim(
                _to_float_tensor(observation_batch.infra_tokens, device=device),
                single_rank=2,
                name="observation_batch.infra_tokens",
            )
            history_tokens = _ensure_sequence_dim(
                _to_float_tensor(observation_batch.history_tokens, device=device),
                single_rank=2,
                name="observation_batch.history_tokens",
            )
            order_padding_mask = _ensure_sequence_dim(
                _to_bool_tensor(observation_batch.padding_mask, device=device),
                single_rank=1,
                name="observation_batch.padding_mask",
            )
            history_padding_mask = _ensure_sequence_dim(
                _to_bool_tensor(observation_batch.history_padding_mask, device=device),
                single_rank=1,
                name="observation_batch.history_padding_mask",
            )
            lstm_state = _move_lstm_state(lstm_state, device=lstm_device)

            batch_size, seq_len = uav_self.shape[:2]
            uav_embed = self.uav_proj(uav_self)
            order_embed = self.order_proj(order_tokens)
            infra_embed = self.infra_proj(infra_tokens)
            history_embed = self.history_proj(history_tokens)

            history_embed = history_embed.reshape(
                -1,
                history_embed.size(2),
                history_embed.size(3),
            )
            history_padding_mask_flat = history_padding_mask.reshape(-1, history_padding_mask.size(2))
            history_embed = history_embed.to(lstm_device)
            history_padding_mask_flat = history_padding_mask_flat.to(lstm_device)
            history_embed = history_embed.masked_fill(history_padding_mask_flat.unsqueeze(-1), 0.0)
            history_out, _ = self.history_encoder(history_embed)
            history_summary = _masked_mean(
                history_out,
                ~history_padding_mask_flat,
                dim=1,
            ).reshape(batch_size, seq_len, -1).to(device)
            order_valid_mask = ~order_padding_mask
            order_mean = _masked_mean(order_embed, order_valid_mask, dim=2)
            order_max = _masked_max(order_embed, order_valid_mask, dim=2)
            order_lse = _masked_logmeanexp(order_embed, order_valid_mask, dim=2)
            infra_summary = infra_embed.mean(dim=2)
            base_context = self.context_proj(
                torch.cat(
                    [
                        uav_embed,
                        order_mean,
                        order_max,
                        order_lse,
                        infra_summary,
                        history_summary,
                    ],
                    dim=-1,
                )
            )
            recurrent_out, next_lstm_state = self.recurrent_core(
                base_context.to(lstm_device),
                lstm_state,
            )
            recurrent_context = self.recurrent_proj(recurrent_out.to(device))
            context = base_context + recurrent_context

            root_branch_logits = self.root_head(context)

            context_per_order = context.unsqueeze(2).expand(-1, -1, order_embed.size(2), -1)
            order_logits = self.order_head(
                torch.cat([context_per_order, order_embed], dim=-1)
            ).squeeze(-1)

            mode_logits = self.mode_head(torch.cat([context_per_order, order_embed], dim=-1))

            critic_order = self.critic_order_proj(
                _ensure_sequence_dim(
                    _to_float_tensor(critic_batch.global_order_pool_tokens, device=device),
                    single_rank=2,
                    name="critic_batch.global_order_pool_tokens",
                )
            )
            critic_uav = self.critic_uav_proj(
                _ensure_sequence_dim(
                    _to_float_tensor(critic_batch.global_uav_tokens, device=device),
                    single_rank=2,
                    name="critic_batch.global_uav_tokens",
                )
            )
            critic_station = self.critic_station_proj(
                _ensure_sequence_dim(
                    _to_float_tensor(critic_batch.global_station_tokens, device=device),
                    single_rank=2,
                    name="critic_batch.global_station_tokens",
                )
            )
            critic_order_mask = ~_ensure_sequence_dim(
                _to_bool_tensor(critic_batch.global_order_padding_mask, device=device),
                single_rank=1,
                name="critic_batch.global_order_padding_mask",
            )
            critic_uav_mask = ~_ensure_sequence_dim(
                _to_bool_tensor(critic_batch.global_uav_padding_mask, device=device),
                single_rank=1,
                name="critic_batch.global_uav_padding_mask",
            )
            critic_station_mask = ~_ensure_sequence_dim(
                _to_bool_tensor(critic_batch.global_station_padding_mask, device=device),
                single_rank=1,
                name="critic_batch.global_station_padding_mask",
            )
            critic_order_summary = _masked_mean(critic_order, critic_order_mask, dim=2)
            critic_uav_summary = _masked_mean(critic_uav, critic_uav_mask, dim=2)
            critic_station_summary = _masked_mean(critic_station, critic_station_mask, dim=2)
            critic_plan = _ensure_sequence_dim(
                _to_float_tensor(critic_batch.coarse_plan_summary_vec, device=device),
                single_rank=1,
                name="critic_batch.coarse_plan_summary_vec",
            )
            critic_sys = _ensure_sequence_dim(
                _to_float_tensor(critic_batch.global_system_summary_vec, device=device),
                single_rank=1,
                name="critic_batch.global_system_summary_vec",
            )
            value = self.critic_head(
                torch.cat(
                    [
                        critic_order_summary,
                        critic_uav_summary,
                        critic_station_summary,
                        critic_plan,
                        critic_sys,
                        recurrent_context,
                    ],
                    dim=-1,
                )
            ).squeeze(-1)

            return (
                PolicyForwardOutput(
                    root_branch_logits=root_branch_logits,
                    order_logits=order_logits,
                    mode_logits=mode_logits,
                    value=value,
                ),
                next_lstm_state,
            )

        def sample_action(
            self,
            *,
            policy_out: PolicyForwardOutput,
            action_mask: Any,
            deterministic: bool = False,
        ) -> tuple[dict[str, int | None], Tensor]:
            device = policy_out.root_branch_logits.device
            root_mask = _ensure_batch_dim(
                _to_bool_tensor(action_mask.root_branch_mask, device=device),
                single_rank=1,
                name="action_mask.root_branch_mask",
            )
            order_mask = _ensure_batch_dim(
                _to_bool_tensor(action_mask.order_mask, device=device),
                single_rank=1,
                name="action_mask.order_mask",
            )
            mode_mask = _ensure_batch_dim(
                _to_bool_tensor(action_mask.mode_mask, device=device),
                single_rank=2,
                name="action_mask.mode_mask",
            )

            root_logits = _masked_logits(policy_out.root_branch_logits, root_mask)
            root_dist = torch.distributions.Categorical(logits=root_logits)
            root_idx = torch.argmax(root_logits, dim=-1) if deterministic else root_dist.sample()
            log_prob = root_dist.log_prob(root_idx)

            batch = root_idx.size(0)
            result = {
                "root_branch_idx": int(root_idx[0].item()),
                "order_idx": None,
                "mode_idx": None,
            }
            if batch != 1:
                raise ValueError("当前 sample_action 仅支持 batch_size=1")
            if int(root_idx[0].item()) == 0:
                return result, log_prob.squeeze(0)

            order_logits = _masked_logits(policy_out.order_logits, order_mask)
            order_dist = torch.distributions.Categorical(logits=order_logits)
            order_idx = torch.argmax(order_logits, dim=-1) if deterministic else order_dist.sample()
            log_prob = log_prob + order_dist.log_prob(order_idx)
            result["order_idx"] = int(order_idx[0].item())

            mode_logits = _select_order_slot(
                policy_out.mode_logits,
                order_idx,
                name="policy_out.mode_logits",
            )
            chosen_mode_mask = _select_order_slot(
                mode_mask,
                order_idx,
                name="action_mask.mode_mask",
            )
            mode_logits = _masked_logits(mode_logits, chosen_mode_mask)
            mode_dist = torch.distributions.Categorical(logits=mode_logits)
            mode_idx = torch.argmax(mode_logits, dim=-1) if deterministic else mode_dist.sample()
            log_prob = log_prob + mode_dist.log_prob(mode_idx)
            result["mode_idx"] = int(mode_idx[0].item())

            return result, log_prob.squeeze(0)

        def evaluate_actions(
            self,
            *,
            policy_out: PolicyForwardOutput,
            action_mask: Any,
            action_indices: dict[str, Tensor],
        ) -> tuple[Tensor, Tensor]:
            device = policy_out.root_branch_logits.device
            root_mask = _ensure_batch_dim(
                _to_bool_tensor(action_mask.root_branch_mask, device=device),
                single_rank=1,
                name="action_mask.root_branch_mask",
            )
            order_mask = _ensure_batch_dim(
                _to_bool_tensor(action_mask.order_mask, device=device),
                single_rank=1,
                name="action_mask.order_mask",
            )
            mode_mask = _ensure_batch_dim(
                _to_bool_tensor(action_mask.mode_mask, device=device),
                single_rank=2,
                name="action_mask.mode_mask",
            )

            root_logits = _masked_logits(policy_out.root_branch_logits, root_mask)
            root_dist = torch.distributions.Categorical(logits=root_logits)
            root_idx = action_indices["root_branch_idx"]
            log_prob = root_dist.log_prob(root_idx)

            # True joint entropy:
            #   H(root) + P(dispatch) * [H(order) + E_o H(mode|o)]
            # 这里必须对 order 分布取期望，而不是沿用 action_indices["order_idx"]。
            # 否则 WAIT 样本里被占位成 0 的 order_idx 会把 mode 熵错误地压到
            # 第 0 个订单分支，导致其他订单分支几乎没有熵正则。
            p_dispatch = root_dist.probs[:, 1]
            entropy = root_dist.entropy()

            dispatch_mask = root_idx == 1
            order_idx = action_indices["order_idx"]
            order_logits, order_has_valid = _safe_masked_logits(
                policy_out.order_logits,
                order_mask,
            )
            order_dist = torch.distributions.Categorical(logits=order_logits)
            log_prob = log_prob + order_dist.log_prob(order_idx) * (
                dispatch_mask & order_has_valid
            )
            order_probs, order_entropy = _masked_categorical_probs_entropy(
                policy_out.order_logits,
                order_mask,
            )
            entropy = entropy + p_dispatch * order_entropy

            batch = root_idx.size(0)
            mode_logits = _select_order_slot(
                policy_out.mode_logits,
                order_idx,
                name="policy_out.mode_logits",
            )
            chosen_mode_mask = _select_order_slot(
                mode_mask,
                order_idx,
                name="action_mask.mode_mask",
            )
            mode_logits, mode_has_valid = _safe_masked_logits(
                mode_logits,
                chosen_mode_mask,
            )
            mode_dist = torch.distributions.Categorical(logits=mode_logits)
            mode_idx = action_indices["mode_idx"]
            log_prob = log_prob + mode_dist.log_prob(mode_idx) * (
                dispatch_mask & mode_has_valid
            )
            mode_probs, mode_entropy = _masked_categorical_probs_entropy(
                policy_out.mode_logits,
                mode_mask,
            )
            expected_mode_entropy = (order_probs * mode_entropy).sum(dim=-1)
            entropy = entropy + p_dispatch * expected_mode_entropy

            return log_prob, entropy


    def _to_float_tensor(value: Any, *, device: torch.device) -> Tensor:
        return torch.as_tensor(value, dtype=torch.float32, device=device)


    def _to_bool_tensor(value: Any, *, device: torch.device) -> Tensor:
        return torch.as_tensor(value, dtype=torch.bool, device=device)


    def _select_order_slot(value: Tensor, order_idx: Tensor, *, name: str) -> Tensor:
        if value.dim() != 3:
            raise ValueError(f"{name} 期望为 3D tensor，实际 shape={tuple(value.shape)}")
        if order_idx.dim() != 1:
            raise ValueError(
                f"order_idx 期望为 1D tensor，实际 shape={tuple(order_idx.shape)}"
            )
        if value.size(0) != order_idx.size(0):
            raise ValueError(
                f"{name} batch 维度与 order_idx 不一致: "
                f"{value.size(0)} != {order_idx.size(0)}"
            )
        index = order_idx.to(dtype=torch.long).reshape(-1, 1, 1).expand(
            -1,
            1,
            value.size(-1),
        )
        return torch.gather(value, dim=1, index=index).squeeze(1)


    def _ensure_batch_dim(
        tensor: Tensor,
        *,
        single_rank: int,
        name: str,
    ) -> Tensor:
        if tensor.dim() == single_rank:
            return tensor.unsqueeze(0)
        if tensor.dim() == single_rank + 1:
            return tensor
        raise ValueError(
            f"{name} 维度不符合预期: 期望 rank={single_rank} 或 {single_rank + 1}，"
            f"实际={tensor.dim()} shape={tuple(tensor.shape)}"
        )


    def _ensure_sequence_dim(
        tensor: Tensor,
        *,
        single_rank: int,
        name: str,
    ) -> Tensor:
        if tensor.dim() == single_rank:
            return tensor.unsqueeze(0).unsqueeze(0)
        if tensor.dim() == single_rank + 1:
            return tensor.unsqueeze(1)
        if tensor.dim() == single_rank + 2:
            return tensor
        raise ValueError(
            f"{name} 维度不符合 sequence 预期: 期望 rank={single_rank} / {single_rank + 1} / {single_rank + 2}，"
            f"实际={tensor.dim()} shape={tuple(tensor.shape)}"
        )


    def _move_lstm_state(
        lstm_state: tuple[Tensor, Tensor] | None,
        *,
        device: torch.device,
    ) -> tuple[Tensor, Tensor] | None:
        if lstm_state is None:
            return None
        hidden, cell = lstm_state
        return hidden.to(device), cell.to(device)


    def _masked_logits(logits: Tensor, valid_mask: Tensor) -> Tensor:
        if logits.shape != valid_mask.shape:
            raise ValueError(f"logits/mask shape 不一致: {logits.shape} vs {valid_mask.shape}")
        if (~valid_mask).all():
            raise ValueError("masked_logits 收到全 False mask")
        return logits.masked_fill(~valid_mask, -1e9)


    def _safe_masked_logits(logits: Tensor, valid_mask: Tensor) -> tuple[Tensor, Tensor]:
        if logits.shape != valid_mask.shape:
            raise ValueError(f"logits/mask shape 不一致: {logits.shape} vs {valid_mask.shape}")
        has_valid = valid_mask.any(dim=-1, keepdim=True)
        safe_logits = logits.masked_fill(~valid_mask, -1e9)
        safe_logits = torch.where(has_valid, safe_logits, torch.zeros_like(safe_logits))
        return safe_logits, has_valid.squeeze(-1)


    def _masked_categorical_probs_entropy(
        logits: Tensor,
        valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if logits.shape != valid_mask.shape:
            raise ValueError(f"logits/mask shape 不一致: {logits.shape} vs {valid_mask.shape}")

        has_valid = valid_mask.any(dim=-1, keepdim=True)
        safe_logits = logits.masked_fill(~valid_mask, -1e9)
        safe_logits = torch.where(has_valid, safe_logits, torch.zeros_like(safe_logits))
        probs = torch.softmax(safe_logits, dim=-1)
        probs = torch.where(valid_mask, probs, torch.zeros_like(probs))
        probs = torch.where(has_valid, probs, torch.zeros_like(probs))
        log_probs = torch.log(probs.clamp_min(1e-12))
        entropy = -(probs * log_probs).sum(dim=-1)
        entropy = torch.where(
            has_valid.squeeze(-1),
            entropy,
            torch.zeros_like(entropy),
        )
        return probs, entropy


    def _masked_mean(values: Tensor, valid_mask: Tensor, *, dim: int) -> Tensor:
        weights = valid_mask.to(dtype=values.dtype).unsqueeze(-1)
        weighted = values * weights
        denom = weights.sum(dim=dim).clamp(min=1.0)
        return weighted.sum(dim=dim) / denom


    def _masked_max(values: Tensor, valid_mask: Tensor, *, dim: int) -> Tensor:
        mask = valid_mask.unsqueeze(-1)
        has_valid = valid_mask.any(dim=dim, keepdim=False).unsqueeze(-1)
        masked_values = values.masked_fill(~mask, -1e9)
        pooled = masked_values.max(dim=dim).values
        return torch.where(has_valid, pooled, torch.zeros_like(pooled))


    def _masked_logmeanexp(values: Tensor, valid_mask: Tensor, *, dim: int) -> Tensor:
        mask = valid_mask.unsqueeze(-1)
        has_valid = valid_mask.any(dim=dim, keepdim=False).unsqueeze(-1)
        safe_values = values.masked_fill(~mask, -1e9)
        pooled = torch.logsumexp(safe_values, dim=dim)
        count = valid_mask.sum(dim=dim).clamp(min=1).to(dtype=values.dtype).unsqueeze(-1)
        pooled = pooled - torch.log(count)
        return torch.where(has_valid, pooled, torch.zeros_like(pooled))


else:

    @dataclass(frozen=True)
    class PolicyForwardOutput:
        root_branch_logits: Any
        order_logits: Any
        mode_logits: Any
        value: Any


    class SharedPPOActorCritic:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("缺少 torch，无法实例化 SharedPPOActorCritic")


__all__ = ["PolicyForwardOutput", "SharedPPOActorCritic"]
