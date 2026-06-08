from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GACandidate:
    order_id: str
    mode: str
    feasible: bool
    reason: str = ""

    truck_id: str = ""
    drone_id: str = ""
    launch_node_id: str = ""
    recover_node_id: str = ""

    completion_time: float = 0.0
    truck_distance: float = 0.0
    uav_distance: float = 0.0
    truck_energy: float = 0.0
    uav_energy: float = 0.0
    waiting_time: float = 0.0
    lateness: float = 0.0

    cost_dist: float = 0.0
    cost_energy: float = 0.0
    cost_penalty: float = 0.0
    mode_reward: float = 0.0
    score_total: float = math.inf

    truck_stops: list[dict[str, Any]] = field(default_factory=list)
    drone_route_fragment: Any | None = None
    allocation_fragment: Any | None = None

    truck_final_node_id: str = ""
    truck_final_time: float = 0.0
    drone_final_node_id: str = ""
    drone_final_time: float = 0.0
    drone_energy_used: float = 0.0
    drone_energy_after: float = 0.0
    delivered_order_ids: list[str] = field(default_factory=list)


class PhysicalEvaluator:
    def __init__(self, entity_mgr, greedy_helper, config):
        self.entity_mgr = entity_mgr
        self.greedy = greedy_helper
        self.config = config

    def _read_field(self, record: Any, field_name: str, default: Any = None) -> Any:
        if isinstance(record, dict):
            return record.get(field_name, default)
        return getattr(record, field_name, default)

    def _write_field(self, record: Any, field_name: str, value: Any) -> None:
        if isinstance(record, dict):
            record[field_name] = value
        else:
            setattr(record, field_name, value)

    def _config_float(self, field_name: str, default: float) -> float:
        value = self._read_field(self.config, field_name, default)
        return float(default if value is None else value)

    def _as_list(self, value: Any) -> list:
        if value is None:
            return []
        if isinstance(value, list):
            return list(value)
        if isinstance(value, tuple | set):
            return list(value)
        return [value]

    def _ga_list(self, record: Any, ga_field: str, fallback_field: str | None = None) -> list:
        value = self._read_field(record, ga_field)
        if value is not None:
            return self._as_list(value)
        if fallback_field is None:
            return []
        return self._as_list(self._read_field(record, fallback_field))

    def _write_ga_list(self, record: Any, ga_field: str, values: list) -> None:
        deduped = list(dict.fromkeys(values))
        self._write_field(record, ga_field, deduped)

    def _mapping(self, state: Any, field_name: str) -> dict:
        value = self._read_field(state, field_name)
        if isinstance(value, dict):
            return value
        state_mgr = (
            self._read_field(state, "entity_mgr")
            or self._read_field(state, "entity_manager")
        )
        if state_mgr is not None:
            value = self._read_field(state_mgr, field_name)
            if isinstance(value, dict):
                return value
        fallback = self._read_field(self.entity_mgr, field_name, {})
        return fallback if isinstance(fallback, dict) else {}

    def _current_time(self, state: Any) -> float:
        return float(self._read_field(state, "current_time", 0.0) or 0.0)

    def _depot_items(self, state: Any) -> list[tuple[str, Any]]:
        return list(self._mapping(state, "depots").items())

    def _station_items(self, state: Any) -> list[tuple[str, Any]]:
        return list(self._mapping(state, "stations").items())

    def _first_depot(self, state: Any) -> tuple[str, Any]:
        depots = self._depot_items(state)
        if not depots:
            raise KeyError("no depot available")
        return depots[0]

    def _is_depot_node_id(self, node_id: str) -> bool:
        normalized = str(node_id).strip().upper()
        return normalized == "DEPOT" or normalized.startswith("DEPOT") or normalized.startswith("DEP-")

    def _node_host_type(self, state: Any, node_id: str) -> str:
        if self._is_depot_node_id(node_id) or node_id in self._mapping(state, "depots"):
            return "DEPOT"
        if node_id in self._mapping(state, "stations"):
            return "STATION"
        return ""

    def _find_depot_by_node_id(self, state: Any, node_id: str) -> tuple[str, Any]:
        depots = self._mapping(state, "depots")
        if node_id in depots:
            return node_id, depots[node_id]
        return self._first_depot(state)

    def _find_station_by_node_id(self, state: Any, node_id: str) -> tuple[str, Any] | tuple[None, None]:
        stations = self._mapping(state, "stations")
        if node_id in stations:
            return node_id, stations[node_id]
        return None, None

    def _truck_contains_drone(self, truck: Any, drone_id: str) -> bool:
        return drone_id in self._ga_list(truck, "_ga_docked_drones", "docked_drones")

    def _is_drone_on_ga_truck(self, state: Any, drone: Any, drone_id: str) -> bool:
        host_type = self._read_field(drone, "_ga_host_type")
        if host_type is not None:
            return str(host_type).upper() == "TRUCK"
        if self._read_field(drone, "_ga_transport_truck_id") is not None:
            return bool(self._read_field(drone, "_ga_transport_truck_id"))
        return any(self._truck_contains_drone(truck, drone_id) for truck in self._mapping(state, "trucks").values())

    def _is_drone_at_ga_depot(self, state: Any, drone: Any, drone_id: str) -> bool:
        host_type = self._read_field(drone, "_ga_host_type")
        if host_type is not None:
            return str(host_type).upper() == "DEPOT"
        for _, depot in self._depot_items(state):
            idle_drones = self._ga_list(depot, "_ga_idle_drones", "idle_drones")
            if drone_id in idle_drones and self._distance_to_depot(drone, depot) <= 30.0:
                return True
        return False

    def _is_drone_at_ga_station(self, state: Any, drone: Any, drone_id: str) -> bool:
        host_type = self._read_field(drone, "_ga_host_type")
        if host_type is not None:
            return str(host_type).upper() == "STATION"
        for _, station in self._station_items(state):
            idle_drones = self._ga_list(station, "_ga_idle_drones", "idle_drones")
            if drone_id in idle_drones:
                return True
        return False

    def _independent_launch_node(self, state: Any, drone: Any, drone_id: str) -> tuple[str, Any] | tuple[None, None]:
        host_type = str(self._read_field(drone, "_ga_host_type", "") or "").upper()
        host_node_id = str(self._read_field(drone, "_ga_host_node_id", "") or "")
        if host_type == "DEPOT" and host_node_id:
            launch_pos = self.get_node_position(host_node_id, state)
            if launch_pos is not None:
                return host_node_id, launch_pos
        if host_type == "STATION":
            return None, None

        for depot_id, depot in self._depot_items(state):
            idle_drones = self._ga_list(depot, "_ga_idle_drones", "idle_drones")
            if drone_id in idle_drones:
                return depot_id, self._read_field(depot, "location")

        depot_id, depot = self._first_depot(state)
        return depot_id, self._read_field(depot, "location")

    def _remove_drone_from_ga_hosts(self, state: Any, drone_id: str) -> None:
        for truck in self._mapping(state, "trucks").values():
            drones = self._ga_list(truck, "_ga_docked_drones", "docked_drones")
            self._write_ga_list(truck, "_ga_docked_drones", [uid for uid in drones if uid != drone_id])

        for _, depot in self._depot_items(state):
            drones = self._ga_list(depot, "_ga_idle_drones", "idle_drones")
            self._write_ga_list(depot, "_ga_idle_drones", [uid for uid in drones if uid != drone_id])

        for _, station in self._station_items(state):
            idle_drones = self._ga_list(station, "_ga_idle_drones", "idle_drones")
            waiting_drones = self._ga_list(station, "_ga_waiting_drones", "waiting_drones")
            self._write_ga_list(station, "_ga_idle_drones", [uid for uid in idle_drones if uid != drone_id])
            self._write_ga_list(station, "_ga_waiting_drones", [uid for uid in waiting_drones if uid != drone_id])
            existing_queue_state = self._read_field(station, "_ga_queue_state")
            queue_state = [
                item
                for item in self._as_list(existing_queue_state)
                if self._read_field(item, "drone_id") != drone_id
            ]
            if existing_queue_state is not None:
                self._write_field(station, "_ga_queue_state", queue_state)

    def _append_station_queue_state(self, station: Any, drone_id: str, candidate: GACandidate) -> None:
        queue_state = self._as_list(self._read_field(station, "_ga_queue_state"))
        queue_state.append(
            {
                "drone_id": drone_id,
                "order_id": candidate.order_id,
                "recover_node_id": candidate.recover_node_id,
                "available_time": candidate.drone_final_time,
            }
        )
        self._write_field(station, "_ga_queue_state", queue_state)

    def _set_drone_host(
        self,
        state: Any,
        drone: Any,
        drone_id: str,
        host_type: str,
        host_node_id: str,
        truck_id: str | None = None,
        candidate: GACandidate | None = None,
    ) -> None:
        host_type = host_type.upper()
        self._remove_drone_from_ga_hosts(state, drone_id)
        self._write_field(drone, "_ga_host_type", host_type)
        self._write_field(drone, "_ga_host_node_id", host_node_id)
        self._write_field(drone, "_ga_transport_truck_id", truck_id if host_type == "TRUCK" else None)
        self._write_field(drone, "_ga_waiting_station_id", host_node_id if host_type == "STATION" else None)

        if host_type == "TRUCK" and truck_id:
            truck = self._mapping(state, "trucks").get(truck_id)
            if truck is not None:
                docked = self._ga_list(truck, "_ga_docked_drones", "docked_drones")
                self._write_ga_list(truck, "_ga_docked_drones", docked + [drone_id])
        elif host_type == "DEPOT":
            _, depot = self._find_depot_by_node_id(state, host_node_id)
            idle_drones = self._ga_list(depot, "_ga_idle_drones", "idle_drones")
            self._write_ga_list(depot, "_ga_idle_drones", idle_drones + [drone_id])
        elif host_type == "STATION":
            _, station = self._find_station_by_node_id(state, host_node_id)
            if station is not None:
                waiting_drones = self._ga_list(station, "_ga_waiting_drones", "waiting_drones")
                self._write_ga_list(station, "_ga_waiting_drones", waiting_drones + [drone_id])
                if candidate is not None:
                    self._append_station_queue_state(station, drone_id, candidate)

    def _status_name(self, value: Any) -> str:
        if value is None:
            return ""
        if hasattr(value, "value"):
            value = value.value
        return str(value).strip().upper()

    def _is_idle(self, entity: Any) -> bool:
        if bool(self._read_field(entity, "_ga_force_available", False)):
            return True
        status = self._read_field(entity, "status")
        if hasattr(status, "is_dispatchable"):
            return bool(status.is_dispatchable)
        return self._status_name(status) == "IDLE"

    def _has_pending_route(self, drone: Any) -> bool:
        if bool(self._read_field(drone, "_ga_force_available", False)):
            return False
        value = self._read_field(drone, "has_pending_route")
        if value is not None:
            return bool(value)
        route_plan = self._read_field(drone, "route_plan", []) or []
        waypoint_idx = int(self._read_field(drone, "current_waypoint_index", 0) or 0)
        return waypoint_idx < len(route_plan)

    def _distance_to_depot(self, drone: Any, depot: Any) -> float:
        drone_pos = self._read_field(drone, "_ga_position") or self._read_field(drone, "current_loc")
        depot_pos = self._read_field(depot, "location")
        if drone_pos is None or depot_pos is None:
            return 0.0
        if hasattr(drone_pos, "distance_2d"):
            return float(drone_pos.distance_2d(depot_pos))
        return self.greedy._dist(drone_pos, depot_pos)

    def _truck_position(self, truck: Any, state: Any) -> Any:
        ga_pos = self._read_field(truck, "_ga_position")
        if ga_pos is not None:
            return ga_pos
        if hasattr(truck, "get_location"):
            return truck.get_location(self._current_time(state))
        return self._read_field(truck, "current_loc")

    def _truck_time(self, truck: Any, state: Any) -> float:
        return float(self._read_field(truck, "_ga_time", self._current_time(state)) or 0.0)

    def _drone_energy(self, drone: Any) -> float:
        return float(self._read_field(drone, "_ga_energy", self._read_field(drone, "battery_current", 0.0)) or 0.0)

    def _drone_time(self, drone: Any, state: Any) -> float:
        return float(self._read_field(drone, "_ga_time", self._current_time(state)) or 0.0)

    def _service_time_customer(self) -> float:
        return float(self._read_field(self.greedy, "SERVICE_TIME_CUSTOMER", 60.0) or 60.0)

    def _drone_service_time(self) -> float:
        return float(self._read_field(self.greedy, "delivery_service_time", 30.0) or 30.0)

    def _launch_time(self) -> float:
        return float(self._read_field(self.greedy, "TRUCK_DRONE_LAUNCH_TIME", 10.0) or 10.0)

    def _recover_time(self) -> float:
        return float(self._read_field(self.greedy, "TRUCK_DRONE_RECOVER_TIME", 10.0) or 10.0)

    def get_node_position(self, node_id: str, state: Any | None = None):
        state = state if state is not None else self.entity_mgr
        if self._is_depot_node_id(node_id):
            depots = self._mapping(state, "depots")
            if node_id in depots:
                return depots[node_id].location
            _, depot = self._first_depot(state)
            return depot.location

        stations = self._mapping(state, "stations")
        if node_id in stations:
            return stations[node_id].location

        depots = self._mapping(state, "depots")
        if node_id in depots:
            return depots[node_id].location

        raise KeyError(f"unknown depot/station node_id: {node_id}")

    def is_legal_rendezvous_node(self, node_id: str, state: Any | None = None) -> bool:
        state = state if state is not None else self.entity_mgr
        return (
            self._is_depot_node_id(node_id)
            or node_id in self._mapping(state, "depots")
            or node_id in self._mapping(state, "stations")
        )

    def validate_drone_for_mode(self, state, mode: str, drone_id: str) -> tuple[bool, str]:
        drones = self._mapping(state, "drones")
        drone = drones.get(drone_id)
        if drone is None:
            return False, "drone_not_found"
        force_available = bool(self._read_field(drone, "_ga_force_available", False))
        if not force_available:
            if not self._is_idle(drone):
                return False, "drone_not_idle"
            if self._read_field(drone, "carrying_order_id"):
                return False, "drone_busy"
            if self._read_field(drone, "waiting_recovery_station_id"):
                return False, "drone_waiting_recovery"
            if self._has_pending_route(drone):
                return False, "drone_has_pending_route"

        mode = mode.upper()
        if mode == "B":
            if self._is_drone_on_ga_truck(state, drone, drone_id):
                return True, ""
            if self._read_field(drone, "transport_truck_id"):
                return True, ""
            return False, "drone_not_on_truck"

        if mode == "C":
            if self._is_drone_on_ga_truck(state, drone, drone_id):
                return False, "drone_on_truck"
            if self._read_field(drone, "transport_truck_id"):
                return False, "drone_on_truck"
            if self._is_drone_at_ga_depot(state, drone, drone_id):
                return True, ""
            return False, "drone_not_at_independent_node"

        return False, f"unknown_mode:{mode}"

    def _check_fixed_recovery_energy(self, drone, launch_pos, delivery_pos, recover_pos, payload) -> tuple[bool, float, str]:
        e1 = self.greedy._flight_energy(drone, launch_pos, delivery_pos, payload)
        e2 = self.greedy._flight_energy(drone, delivery_pos, recover_pos, 0.0)
        need = (e1 + e2) * float(self._read_field(self.greedy, "ENERGY_SAFETY_FACTOR", 1.0) or 1.0)
        if need > self._drone_energy(drone):
            return False, need, "energy_not_enough"
        return True, need, ""

    def _uav_path_distance(self, from_pos: Any, to_pos: Any) -> float:
        altitude = float(self._read_field(self.greedy, "UAV_CRUISE_ALTITUDE_M", 80.0) or 80.0)
        return float(self.greedy._uav_path_distance(from_pos, to_pos, altitude=altitude))

    def _uav_route_path(self, launch_pos: Any, delivery_pos: Any, recover_pos: Any) -> list[Any]:
        planner = self._read_field(self.greedy, "_path_planner")
        if planner is None:
            return [launch_pos, delivery_pos, recover_pos]

        altitude = float(self._read_field(self.greedy, "UAV_CRUISE_ALTITUDE_M", 80.0) or 80.0)
        outbound = list(planner.plan(launch_pos, delivery_pos, altitude) or [])
        inbound = list(planner.plan(delivery_pos, recover_pos, altitude) or [])
        if not outbound:
            outbound = [launch_pos, delivery_pos]
        if not inbound:
            inbound = [delivery_pos, recover_pos]
        return outbound + inbound[1:]

    def evaluate_fixed_mode_a(self, state, order_id: str, truck_id: str) -> GACandidate:
        orders = self._mapping(state, "orders")
        trucks = self._mapping(state, "trucks")
        order = orders.get(order_id)
        truck = trucks.get(truck_id)
        if order is None:
            return GACandidate(order_id, "A", False, reason="order_not_found")
        if truck is None:
            return GACandidate(order_id, "A", False, reason="truck_not_found")

        truck_pos = self._truck_position(truck, state)
        start_time = self._truck_time(truck, state)
        dist = self.greedy._road_dist(truck_pos, order.delivery_loc)
        speed = max(1e-6, float(self._read_field(truck, "speed", 0.0) or 0.0))
        arrival = start_time + dist / speed
        completion = arrival + self._service_time_customer()
        lateness = max(0.0, completion - float(self._read_field(order, "deadline", math.inf)))

        candidate = GACandidate(
            order_id=order_id,
            mode="A",
            feasible=True,
            truck_id=truck_id,
            completion_time=completion,
            truck_distance=dist,
            truck_energy=self.greedy._truck_energy_wh(dist),
            lateness=lateness,
            truck_stops=[{
                "node_id": order_id,
                "node_type": "customer",
                "position": order.delivery_loc,
                "arrival_time": arrival,
                "departure_time": completion,
                "order_id": order_id,
            }],
            truck_final_node_id=order_id,
            truck_final_time=completion,
            delivered_order_ids=[order_id],
        )
        return self.score_candidate(candidate, order)

    def evaluate_fixed_mode_b(
        self,
        state,
        order_id: str,
        truck_id: str,
        drone_id: str,
        launch_node_id: str,
        recover_node_id: str,
    ) -> GACandidate:
        orders = self._mapping(state, "orders")
        trucks = self._mapping(state, "trucks")
        drones = self._mapping(state, "drones")
        order = orders.get(order_id)
        truck = trucks.get(truck_id)
        drone = drones.get(drone_id)

        if order is None:
            return GACandidate(order_id, "B", False, reason="order_not_found")
        if truck is None:
            return GACandidate(order_id, "B", False, reason="truck_not_found")
        if drone is None:
            return GACandidate(order_id, "B", False, reason="drone_not_found")

        ok, reason = self.validate_drone_for_mode(state, "B", drone_id)
        if not ok:
            return GACandidate(order_id, "B", False, drone_id=drone_id, reason=reason)
        if not self.is_legal_rendezvous_node(launch_node_id, state):
            return GACandidate(order_id, "B", False, drone_id=drone_id, reason="illegal_launch_node")
        if not self.is_legal_rendezvous_node(recover_node_id, state):
            return GACandidate(order_id, "B", False, drone_id=drone_id, reason="illegal_recover_node")
        if float(self._read_field(order, "payload_weight", 0.0) or 0.0) > float(self._read_field(drone, "payload_capacity", 0.0) or 0.0):
            return GACandidate(order_id, "B", False, drone_id=drone_id, reason="payload_exceed")

        launch_pos = self.get_node_position(launch_node_id, state)
        recover_pos = self.get_node_position(recover_node_id, state)
        payload = float(self._read_field(order, "payload_weight", 0.0) or 0.0)
        ok, energy_need, reason = self._check_fixed_recovery_energy(drone, launch_pos, order.delivery_loc, recover_pos, payload)
        if not ok:
            return GACandidate(order_id, "B", False, drone_id=drone_id, reason=reason)

        uav_dist_out = self._uav_path_distance(launch_pos, order.delivery_loc)
        uav_dist_back = self._uav_path_distance(order.delivery_loc, recover_pos)
        uav_distance = uav_dist_out + uav_dist_back
        uav_energy = (
            self.greedy._uav_energy_wh(drone, launch_pos, order.delivery_loc, payload)
            + self.greedy._uav_energy_wh(drone, order.delivery_loc, recover_pos, 0.0)
        )

        truck_start_pos = self._truck_position(truck, state)
        truck_start_time = self._truck_time(truck, state)
        truck_dist_to_launch = self.greedy._road_dist(truck_start_pos, launch_pos)
        truck_dist_launch_to_recover = self.greedy._road_dist(launch_pos, recover_pos)
        truck_distance = truck_dist_to_launch + truck_dist_launch_to_recover
        truck_speed = max(1e-6, float(self._read_field(truck, "speed", 0.0) or 0.0))
        launch_arrival_time = truck_start_time + truck_dist_to_launch / truck_speed
        truck_depart_launch_time = launch_arrival_time + self._launch_time()
        truck_recover_arrival = truck_depart_launch_time + truck_dist_launch_to_recover / truck_speed

        drone_speed = max(1e-6, float(self._read_field(drone, "cruise_speed", 0.0) or 0.0))
        drone_launch_time = max(truck_depart_launch_time, self._drone_time(drone, state))
        delivery_arrival = drone_launch_time + uav_dist_out / drone_speed
        delivery_done = delivery_arrival + self._drone_service_time()
        drone_recover_arrival = delivery_done + uav_dist_back / drone_speed

        truck_wait = max(0.0, drone_recover_arrival - truck_recover_arrival)
        uav_wait = max(0.0, truck_recover_arrival - drone_recover_arrival)
        if truck_wait > float(self._read_field(self.config, "truck_wait_max_s", 10.0) or 10.0):
            if not bool(self._read_field(self.config, "soft_rendezvous_violation", True)):
                return GACandidate(order_id, "B", False, drone_id=drone_id, reason="rendezvous_wait_timeout")

        completion = delivery_done
        truck_final_time = max(truck_recover_arrival, drone_recover_arrival) + self._recover_time()
        drone_final_time = drone_recover_arrival + self._recover_time()
        lateness = max(0.0, completion - float(self._read_field(order, "deadline", math.inf)))

        candidate = GACandidate(
            order_id=order_id,
            mode="B",
            feasible=True,
            truck_id=truck_id,
            drone_id=drone_id,
            launch_node_id=launch_node_id,
            recover_node_id=recover_node_id,
            completion_time=completion,
            truck_distance=truck_distance,
            uav_distance=uav_distance,
            truck_energy=self.greedy._truck_energy_wh(truck_distance),
            uav_energy=uav_energy,
            waiting_time=truck_wait + uav_wait,
            lateness=lateness,
            truck_stops=[
                {
                    "node_id": launch_node_id,
                    "node_type": "station" if launch_node_id in self._mapping(state, "stations") else "depot",
                    "position": launch_pos,
                    "arrival_time": launch_arrival_time,
                    "departure_time": truck_depart_launch_time,
                    "order_id": order_id,
                },
                {
                    "node_id": recover_node_id,
                    "node_type": "recovery",
                    "position": recover_pos,
                    "arrival_time": truck_recover_arrival,
                    "departure_time": truck_final_time,
                    "order_id": order_id,
                },
            ],
            drone_route_fragment={
                "drone_id": drone_id,
                "order_id": order_id,
                "mode": "B",
                "launch_node_id": launch_node_id,
                "recover_node_id": recover_node_id,
                "launch_loc": launch_pos,
                "delivery_loc": order.delivery_loc,
                "recovery_loc": recover_pos,
                "path": self._uav_route_path(launch_pos, order.delivery_loc, recover_pos),
            },
            truck_final_node_id=recover_node_id,
            truck_final_time=truck_final_time,
            drone_final_node_id=recover_node_id,
            drone_final_time=drone_final_time,
            drone_energy_used=energy_need,
            drone_energy_after=self._drone_energy(drone) - energy_need,
            delivered_order_ids=[order_id],
        )
        return self.score_candidate(candidate, order)

    def evaluate_fixed_mode_c(self, state, order_id: str, drone_id: str, recover_node_id: str) -> GACandidate:
        orders = self._mapping(state, "orders")
        drones = self._mapping(state, "drones")
        order = orders.get(order_id)
        drone = drones.get(drone_id)
        if order is None:
            return GACandidate(order_id, "C", False, reason="order_not_found")
        if drone is None:
            return GACandidate(order_id, "C", False, reason="drone_not_found")

        ok, reason = self.validate_drone_for_mode(state, "C", drone_id)
        if not ok:
            return GACandidate(order_id, "C", False, drone_id=drone_id, reason=reason)
        if not self.is_legal_rendezvous_node(recover_node_id, state):
            return GACandidate(order_id, "C", False, drone_id=drone_id, reason="illegal_recover_node")
        if not bool(self._read_field(self.config, "allow_depot_drone_recover_at_station", True)):
            if not self._is_depot_node_id(recover_node_id):
                return GACandidate(order_id, "C", False, drone_id=drone_id, reason="station_recover_not_allowed")
        if float(self._read_field(order, "payload_weight", 0.0) or 0.0) > float(self._read_field(drone, "payload_capacity", 0.0) or 0.0):
            return GACandidate(order_id, "C", False, drone_id=drone_id, reason="payload_exceed")

        launch_node_id, launch_pos = self._independent_launch_node(state, drone, drone_id)
        if launch_node_id is None or launch_pos is None:
            return GACandidate(order_id, "C", False, drone_id=drone_id, reason="missing_launch_node")
        recover_pos = self.get_node_position(recover_node_id, state)
        payload = float(self._read_field(order, "payload_weight", 0.0) or 0.0)
        ok, energy_need, reason = self._check_fixed_recovery_energy(drone, launch_pos, order.delivery_loc, recover_pos, payload)
        if not ok:
            return GACandidate(order_id, "C", False, drone_id=drone_id, reason=reason)

        uav_dist_out = self._uav_path_distance(launch_pos, order.delivery_loc)
        uav_dist_back = self._uav_path_distance(order.delivery_loc, recover_pos)
        drone_speed = max(1e-6, float(self._read_field(drone, "cruise_speed", 0.0) or 0.0))
        start_time = self._drone_time(drone, state)
        delivery_arrival = start_time + uav_dist_out / drone_speed
        delivery_done = delivery_arrival + self._drone_service_time()
        recover_arrival = delivery_done + uav_dist_back / drone_speed
        lateness = max(0.0, delivery_done - float(self._read_field(order, "deadline", math.inf)))

        candidate = GACandidate(
            order_id=order_id,
            mode="C",
            feasible=True,
            drone_id=drone_id,
            launch_node_id=launch_node_id,
            recover_node_id=recover_node_id,
            completion_time=delivery_done,
            uav_distance=uav_dist_out + uav_dist_back,
            uav_energy=(
                self.greedy._uav_energy_wh(drone, launch_pos, order.delivery_loc, payload)
                + self.greedy._uav_energy_wh(drone, order.delivery_loc, recover_pos, 0.0)
            ),
            lateness=lateness,
            drone_route_fragment={
                "drone_id": drone_id,
                "order_id": order_id,
                "mode": "C",
                "launch_node_id": launch_node_id,
                "recover_node_id": recover_node_id,
                "launch_loc": launch_pos,
                "delivery_loc": order.delivery_loc,
                "recovery_loc": recover_pos,
                "path": self._uav_route_path(launch_pos, order.delivery_loc, recover_pos),
            },
            drone_final_node_id=recover_node_id,
            drone_final_time=recover_arrival,
            drone_energy_used=energy_need,
            drone_energy_after=self._drone_energy(drone) - energy_need,
            delivered_order_ids=[order_id],
        )
        return self.score_candidate(candidate, order)

    def score_candidate(self, candidate: GACandidate, order: Any) -> GACandidate:
        # This is a local action score. Decoder/fitness owns full Individual fitness aggregation.
        if not candidate.feasible:
            candidate.score_total = math.inf
            return candidate

        candidate.cost_dist = 0.0
        candidate.cost_energy = (
            candidate.truck_energy + candidate.uav_energy
        ) * self._config_float("weight_energy", 0.02)
        candidate.cost_penalty = (
            candidate.lateness * self._config_float("weight_delay", 10.0)
            + candidate.waiting_time * self._config_float("weight_waiting", 0.5)
        )
        candidate.mode_reward = (
            self._config_float("air_ground_mode_reward", 0.0)
            if candidate.mode == "B"
            else 0.0
        )
        candidate.score_total = (
            candidate.cost_energy
            + candidate.cost_penalty
            - candidate.mode_reward
        )
        return candidate

    def apply_candidate(self, state, candidate: GACandidate) -> None:
        """Advance GA-only state on a decoded state copy.

        The caller must pass a deepcopy/clone owned by the current Individual.
        This method intentionally writes only _ga_* fields and does not mutate
        canonical runtime fields such as current_loc, docked_drones, or idle_drones.
        """
        if not candidate.feasible:
            return

        trucks = self._mapping(state, "trucks")
        drones = self._mapping(state, "drones")
        orders = self._mapping(state, "orders")

        if candidate.truck_id and candidate.truck_id in trucks:
            truck = trucks[candidate.truck_id]
            final_pos = None
            if candidate.truck_stops:
                final_pos = candidate.truck_stops[-1].get("position")
            if final_pos is not None:
                self._write_field(truck, "_ga_position", final_pos)
            self._write_field(truck, "_ga_node_id", candidate.truck_final_node_id)
            self._write_field(truck, "_ga_time", candidate.truck_final_time)

        if candidate.drone_id and candidate.drone_id in drones:
            drone = drones[candidate.drone_id]
            if candidate.recover_node_id:
                self._write_field(drone, "_ga_position", self.get_node_position(candidate.recover_node_id, state))
            self._write_field(drone, "_ga_node_id", candidate.drone_final_node_id)
            self._write_field(drone, "_ga_time", candidate.drone_final_time)
            self._write_field(drone, "_ga_energy", max(0.0, candidate.drone_energy_after))
            if candidate.mode == "B":
                self._set_drone_host(
                    state,
                    drone,
                    candidate.drone_id,
                    "TRUCK",
                    candidate.recover_node_id,
                    truck_id=candidate.truck_id,
                    candidate=candidate,
                )
            elif candidate.mode == "C":
                recover_host_type = self._node_host_type(state, candidate.recover_node_id)
                if recover_host_type in {"DEPOT", "STATION"}:
                    self._set_drone_host(
                        state,
                        drone,
                        candidate.drone_id,
                        recover_host_type,
                        candidate.recover_node_id,
                        truck_id=None,
                        candidate=candidate,
                    )

        for order_id in candidate.delivered_order_ids:
            order = orders.get(order_id)
            if order is None:
                continue
            self._write_field(order, "_ga_completed", True)
            self._write_field(order, "_ga_completion_time", candidate.completion_time)

        fragments = self._read_field(state, "_ga_plan_fragments")
        if fragments is None:
            fragments = []
            self._write_field(state, "_ga_plan_fragments", fragments)
        fragments.append(candidate)
