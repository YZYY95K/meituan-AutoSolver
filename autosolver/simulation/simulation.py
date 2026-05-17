from __future__ import annotations
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
import numpy as np

from ..models.scenario import DeliveryScenario
from ..models.order import Order, OrderStatus, OrderPriority
from ..models.rider import Rider, RiderStatus, haversine_distance
from ..agent.agent import AutoSolverAgent, AgentResult


@dataclass
class SimulationConfig:
    total_time_minutes: int = 120
    order_generation_rate: float = 2.0
    batch_interval_minutes: int = 5
    dynamic_orders: bool = True
    seed: int = 42


@dataclass
class SimulationMetrics:
    total_orders_generated: int = 0
    total_orders_delivered: int = 0
    total_orders_overdue: int = 0
    avg_delivery_time: float = 0.0
    on_time_rate: float = 0.0
    rider_utilization: float = 0.0
    total_distance: float = 0.0
    timeline: list[dict] = field(default_factory=list)


class DeliverySimulation:
    def __init__(
        self,
        scenario: DeliveryScenario,
        config: SimulationConfig | None = None,
        agent: AutoSolverAgent | None = None,
    ):
        self.scenario = scenario
        self.config = config or SimulationConfig()
        self.agent = agent or AutoSolverAgent(verbose=False)
        self.rng = np.random.default_rng(self.config.seed)
        self.metrics = SimulationMetrics()
        self._current_orders: list[Order] = list(scenario.orders)
        self._riders: list[Rider] = list(scenario.riders)
        self._delivered_orders: list[Order] = []
        self._overdue_orders: list[Order] = []

    def run(self) -> SimulationMetrics:
        total_steps = self.config.total_time_minutes // self.config.batch_interval_minutes

        for step in range(total_steps):
            current_time = datetime.now() + timedelta(minutes=step * self.config.batch_interval_minutes)

            if self.config.dynamic_orders:
                new_orders = self._generate_orders(current_time)
                self._current_orders.extend(new_orders)
                self.metrics.total_orders_generated += len(new_orders)

            pending_orders = [o for o in self._current_orders if o.status == OrderStatus.PENDING]

            if pending_orders:
                batch_scenario = DeliveryScenario(
                    scenario_id=f"{self.scenario.scenario_id}_batch_{step}",
                    name=f"Batch {step}",
                    description=f"Simulation batch at step {step}",
                    area_bounds=self.scenario.area_bounds,
                    restaurants=self.scenario.restaurants,
                    orders=pending_orders,
                    riders=[r for r in self._riders if r.is_available],
                )

                result = self.agent.solve(batch_scenario)

                for rider_id, order_ids in result.final_assignments.items():
                    for oid in order_ids:
                        order = next((o for o in self._current_orders if o.order_id == oid), None)
                        if order:
                            order.status = OrderStatus.PICKED_UP
                            order.assigned_rider_id = rider_id

            self._simulate_delivery_step(current_time)

            self.metrics.timeline.append({
                "time": current_time.isoformat(),
                "step": step,
                "pending": len([o for o in self._current_orders if o.status == OrderStatus.PENDING]),
                "delivering": len([o for o in self._current_orders if o.status == OrderStatus.DELIVERING]),
                "delivered": len(self._delivered_orders),
                "overdue": len(self._overdue_orders),
            })

        self._compute_final_metrics()
        return self.metrics

    def _generate_orders(self, current_time: datetime) -> list[Order]:
        num_new = self.rng.poisson(self.config.order_generation_rate)
        orders = []
        for i in range(num_new):
            order_id = f"DYN_{self.metrics.total_orders_generated + i:04d}"
            order = Order.generate_random(
                order_id=order_id,
                restaurants=self.scenario.restaurants,
                area_bounds=self.scenario.area_bounds,
                created_time=current_time,
                rng=self.rng,
            )
            orders.append(order)
        return orders

    def _simulate_delivery_step(self, current_time: datetime) -> None:
        for order in self._current_orders:
            if order.status == OrderStatus.PICKED_UP and order.assigned_rider_id:
                rider = next(
                    (r for r in self._riders if r.rider_id == order.assigned_rider_id), None
                )
                if rider:
                    dist = haversine_distance(order.restaurant_location, order.delivery_location)
                    est_time = dist / rider.speed_m_per_min
                    if est_time < self.config.batch_interval_minutes:
                        order.status = OrderStatus.DELIVERED
                        order.delivered_time = current_time + timedelta(minutes=est_time)
                        self._delivered_orders.append(order)
                        rider.completed_orders += 1
                        rider.location = order.delivery_location
                        rider.current_orders = [
                            oid for oid in rider.current_orders if oid != order.order_id
                        ]
                    else:
                        order.status = OrderStatus.DELIVERING

            elif order.status == OrderStatus.DELIVERING and order.assigned_rider_id:
                rider = next(
                    (r for r in self._riders if r.rider_id == order.assigned_rider_id), None
                )
                if rider:
                    dist = haversine_distance(rider.location, order.delivery_location)
                    est_time = dist / rider.speed_m_per_min
                    if est_time < self.config.batch_interval_minutes:
                        order.status = OrderStatus.DELIVERED
                        order.delivered_time = current_time + timedelta(minutes=est_time)
                        self._delivered_orders.append(order)
                        rider.completed_orders += 1
                        rider.location = order.delivery_location

            elif order.status == OrderStatus.PENDING:
                if current_time > order.deadline:
                    order.status = OrderStatus.OVERDUE
                    self._overdue_orders.append(order)

    def _compute_final_metrics(self) -> None:
        total = len(self._delivered_orders) + len(self._overdue_orders)
        self.metrics.total_orders_delivered = len(self._delivered_orders)
        self.metrics.total_orders_overdue = len(self._overdue_orders)

        if total > 0:
            self.metrics.on_time_rate = len(self._delivered_orders) / total

        if self._delivered_orders:
            delivery_times = []
            for o in self._delivered_orders:
                if o.delivered_time and o.created_time:
                    dt = (o.delivered_time - o.created_time).total_seconds() / 60
                    delivery_times.append(dt)
            if delivery_times:
                self.metrics.avg_delivery_time = sum(delivery_times) / len(delivery_times)

        if self._riders:
            self.metrics.rider_utilization = sum(
                r.completed_orders for r in self._riders
            ) / max(1, sum(r.max_orders for r in self._riders))

    def export_results(self, output_path: str = "simulation_results.json") -> str:
        data = {
            "scenario": {
                "id": self.scenario.scenario_id,
                "name": self.scenario.name,
                "orders": self.scenario.num_orders,
                "riders": self.scenario.num_riders,
            },
            "config": {
                "total_time_minutes": self.config.total_time_minutes,
                "batch_interval_minutes": self.config.batch_interval_minutes,
                "dynamic_orders": self.config.dynamic_orders,
            },
            "metrics": {
                "total_orders_generated": self.metrics.total_orders_generated,
                "total_orders_delivered": self.metrics.total_orders_delivered,
                "total_orders_overdue": self.metrics.total_orders_overdue,
                "avg_delivery_time": self.metrics.avg_delivery_time,
                "on_time_rate": self.metrics.on_time_rate,
                "rider_utilization": self.metrics.rider_utilization,
            },
            "timeline": self.metrics.timeline,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        return output_path
