from __future__ import annotations
from dataclasses import dataclass, field
from typing import Protocol
from ..models.order import Order
from ..models.rider import Rider, haversine_distance


@dataclass
class AllocationResult:
    assignments: dict[str, list[str]] = field(default_factory=dict)
    routes: dict[str, list[tuple[float, float]]] = field(default_factory=dict)
    total_distance: float = 0.0
    total_delivery_time: float = 0.0
    on_time_rate: float = 0.0
    avg_delivery_time: float = 0.0
    max_load_imbalance: float = 0.0
    algorithm_name: str = ""
    solve_time_seconds: float = 0.0
    metadata: dict = field(default_factory=dict)

    @property
    def num_assigned(self) -> int:
        return sum(len(v) for v in self.assignments.values())

    @property
    def summary(self) -> str:
        return (
            f"Algorithm: {self.algorithm_name}\n"
            f"Assigned: {self.num_assigned} orders\n"
            f"Total Distance: {self.total_distance:.1f}m\n"
            f"Avg Delivery Time: {self.avg_delivery_time:.1f}min\n"
            f"On-time Rate: {self.on_time_rate:.1%}\n"
            f"Load Imbalance: {self.max_load_imbalance:.2f}\n"
            f"Solve Time: {self.solve_time_seconds:.3f}s"
        )


class AllocationAlgorithm(Protocol):
    name: str

    def solve(
        self,
        orders: list[Order],
        riders: list[Rider],
    ) -> AllocationResult: ...


class GreedyAllocator:
    name = "greedy"

    def solve(
        self,
        orders: list[Order],
        riders: list[Rider],
    ) -> AllocationResult:
        import time

        start = time.time()
        assignments: dict[str, list[str]] = {r.rider_id: [] for r in riders}
        routes: dict[str, list[tuple[float, float]]] = {}
        total_distance = 0.0
        on_time_count = 0
        total_delivery_time = 0.0

        sorted_orders = sorted(
            orders,
            key=lambda o: (o.priority.value, o.remaining_time),
        )

        rider_positions = {r.rider_id: r.location for r in riders}
        rider_loads = {r.rider_id: len(r.current_orders) for r in riders}

        for order in sorted_orders:
            best_rider_id = None
            best_score = float("inf")

            for rider in riders:
                if rider_loads.get(rider.rider_id, 0) >= rider.max_orders:
                    continue

                rider_pos = rider_positions[rider.rider_id]
                dist_to_rest = haversine_distance(rider_pos, order.restaurant_location)
                dist_to_delivery = haversine_distance(
                    order.restaurant_location, order.delivery_location
                )
                total_dist = dist_to_rest + dist_to_delivery
                est_time = total_dist / rider.speed_m_per_min
                time_margin = order.remaining_time - est_time

                load_penalty = rider_loads.get(rider.rider_id, 0) * 500
                urgency_bonus = -2000 if order.is_urgent else 0
                score = total_dist + load_penalty + urgency_bonus

                if time_margin < 0:
                    score += 10000

                if score < best_score:
                    best_score = score
                    best_rider_id = rider.rider_id

            if best_rider_id:
                assignments[best_rider_id].append(order.order_id)
                rider_loads[best_rider_id] = rider_loads.get(best_rider_id, 0) + 1
                rider_positions[best_rider_id] = order.delivery_location
                total_distance += haversine_distance(
                    order.restaurant_location, order.delivery_location
                )
                est_time = total_distance / 400.0
                total_delivery_time += est_time
                if est_time <= order.remaining_time:
                    on_time_count += 1

        for rider_id, order_ids in assignments.items():
            rider = next(r for r in riders if r.rider_id == rider_id)
            route = [rider.location]
            for oid in order_ids:
                order = next(o for o in orders if o.order_id == oid)
                route.extend([order.restaurant_location, order.delivery_location])
            routes[rider_id] = route

        num_assigned = sum(len(v) for v in assignments.values())
        loads = [len(v) for v in assignments.values()]
        avg_load = sum(loads) / max(1, len(loads))
        max_imbalance = max(abs(l - avg_load) for l in loads) if loads else 0

        return AllocationResult(
            assignments=assignments,
            routes=routes,
            total_distance=total_distance,
            total_delivery_time=total_delivery_time,
            on_time_rate=on_time_count / max(1, num_assigned),
            avg_delivery_time=total_delivery_time / max(1, num_assigned),
            max_load_imbalance=max_imbalance,
            algorithm_name=self.name,
            solve_time_seconds=time.time() - start,
        )
