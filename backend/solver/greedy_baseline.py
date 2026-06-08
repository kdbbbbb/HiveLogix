"""
HiveLogix — 极简贪心基线（Simple Baseline）

设计目标：
    1. 去掉多候选评估（不再比较 B/C/A 多模式得分）
    2. 默认优先使用模式 B_WAIT（卡车携带无人机到站点后起飞）
    3. 仅当订单载重超过机队最大载荷时，直接切换卡车模式 A
    4. B_WAIT 候选按“最短总路径”选择（卡车到起飞站 + 无人机往返）
    5. 继续复用父类的卡车末端插入与路径构建逻辑
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from solver.greedy_mmce import GreedyMMCE, AllocationResult

if TYPE_CHECKING:
    from core.entities.drone import Drone
    from core.entities.order import Order
    from core.entities.primitives import Position3D


class GreedyBaseline(GreedyMMCE):
    """最基础贪心：默认 B_WAIT，超载或不可行时回退 A。"""

    def _allocate_order(
        self,
        order: "Order",
        current_time: float,
        allocated_drones: set[str],
        truck_last_pos: dict[str, "Position3D"],
    ) -> AllocationResult:
        # 仅当订单重量超过机队最大载荷时，直接走卡车。
        if order.payload_weight > self._max_drone_payload_capacity():
            return self._try_mode_a(order, current_time, allocated_drones, truck_last_pos)

        # 其余订单优先 B_WAIT；若资源/能量不可行，再回退 A。
        result_b_wait = self._try_mode_b_wait_shortest(order, current_time, allocated_drones, truck_last_pos)
        if result_b_wait.feasible:
            return result_b_wait

        return self._try_mode_a(order, current_time, allocated_drones, truck_last_pos)

    def _max_drone_payload_capacity(self) -> float:
        """返回机队最大载荷能力；若无无人机则为 0。"""
        drones = list(self.entity_mgr.drones.values())
        if not drones:
            return 0.0
        return max(float(d.payload_capacity) for d in drones)

    def _try_mode_b_wait_shortest(
        self,
        order: "Order",
        current_time: float,
        allocated_drones: set[str],
        truck_last_pos: dict[str, "Position3D"],
    ) -> AllocationResult:
        """模式 B_WAIT：在可行候选中按最短总路径选解。"""
        trucks = list(self.entity_mgr.trucks.values())
        if not trucks:
            return AllocationResult(
                order_id=order.order_id,
                vehicle_id="",
                mode="B_WAIT",
                distance=float("inf"),
                feasible=False,
                reason="无可用卡车",
            )

        available_drones = self._get_available_drones()
        available_drones = [d for d in available_drones if d.drone_id not in allocated_drones]
        available_drones = [d for d in available_drones if d.payload_capacity >= order.payload_weight]
        drone = self._find_capable_drone(order.payload_weight, available_drones)
        if drone is None:
            return AllocationResult(
                order_id=order.order_id,
                vehicle_id="",
                mode="B_WAIT",
                distance=float("inf"),
                feasible=False,
                reason="无未分配的载重匹配无人机",
            )

        recovery_pool = self._get_recovery_pool()
        best_scenario: dict | None = None
        best_truck_id = ""
        best_total_path = float("inf")

        for truck in trucks:
            predicted_stations = self._predict_truck_charging_stations(truck, current_time)
            truck_loc = truck_last_pos.get(truck.truck_id) or truck.get_location(current_time)

            for station_id, station_loc in predicted_stations:
                truck_distance_to_launch = self._road_dist(truck_loc, station_loc)
                launch_delay = (
                    truck_distance_to_launch / truck.speed
                    if truck.speed > 0
                    else float("inf")
                )
                scenario = self._evaluate_charging_station_departure(
                    drone=drone,
                    truck_tail_loc=truck_loc,
                    launch_loc=station_loc,
                    launch_station_id=station_id,
                    truck_distance_to_launch=truck_distance_to_launch,
                    launch_delay=launch_delay,
                    delivery_loc=order.delivery_loc,
                    payload=order.payload_weight,
                    recovery_pool=recovery_pool,
                    current_time=current_time,
                    order=order,
                )
                if not scenario or not scenario.get("feasible", False):
                    continue

                # 最短总路径：卡车到起飞站 + 无人机起飞站到配送点再到回收点。
                total_path = truck_distance_to_launch + float(scenario["distance"])
                if total_path < best_total_path:
                    best_total_path = total_path
                    best_scenario = scenario
                    best_truck_id = truck.truck_id

        if best_scenario is None:
            return AllocationResult(
                order_id=order.order_id,
                vehicle_id="",
                mode="B_WAIT",
                distance=float("inf"),
                feasible=False,
                reason="无可行充电站方案（能量或距离不可行）",
            )

        return AllocationResult(
            order_id=order.order_id,
            vehicle_id=best_truck_id,
            mode="B_WAIT",
            distance=best_total_path,
            feasible=True,
            recovery_station_id=best_scenario["recovery_station_id"],
            drone_id=drone.drone_id,
            launch_station_id=best_scenario["launch_station_id"],
            launch_time=best_scenario["launch_time"],
            wait_duration=best_scenario["wait_duration"],
            score_total=best_total_path,
            cost_dist=best_total_path,
            cost_energy=0.0,
            cost_penalty=0.0,
        )
