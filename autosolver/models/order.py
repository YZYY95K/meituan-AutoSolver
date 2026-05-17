from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np


class OrderStatus(str, Enum):
    PENDING = "pending"
    PREPARING = "preparing"
    READY = "ready"
    PICKED_UP = "picked_up"
    DELIVERING = "delivering"
    DELIVERED = "delivered"
    OVERDUE = "overdue"


class OrderPriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class Order:
    order_id: str
    restaurant_id: str
    restaurant_location: tuple[float, float]
    delivery_location: tuple[float, float]
    created_time: datetime
    prep_time_minutes: float
    deadline: datetime
    priority: OrderPriority = OrderPriority.NORMAL
    status: OrderStatus = OrderStatus.PENDING
    assigned_rider_id: str | None = None
    picked_up_time: datetime | None = None
    delivered_time: datetime | None = None
    weight: float = 1.0

    @property
    def ready_time(self) -> datetime:
        return self.created_time + timedelta(minutes=self.prep_time_minutes)

    @property
    def remaining_time(self) -> float:
        now = datetime.now()
        return max(0, (self.deadline - now).total_seconds() / 60)

    @property
    def is_urgent(self) -> bool:
        return self.remaining_time < 10 or self.priority == OrderPriority.URGENT

    @staticmethod
    def generate_random(
        order_id: str,
        restaurants: list,
        area_bounds: tuple[float, float, float, float],
        created_time: datetime,
        rng: np.random.Generator | None = None,
    ) -> "Order":
        rng = rng or np.random.default_rng()
        restaurant = rng.choice(restaurants)
        lat_min, lon_min, lat_max, lon_max = area_bounds
        delivery_lat = rng.uniform(lat_min, lat_max)
        delivery_lon = rng.uniform(lon_min, lon_max)
        prep_time = rng.uniform(10, 30)
        deadline = created_time + timedelta(minutes=rng.uniform(30, 60))
        priority = rng.choice(
            list(OrderPriority),
            p=[0.1, 0.6, 0.25, 0.05],
        )
        return Order(
            order_id=order_id,
            restaurant_id=restaurant.restaurant_id,
            restaurant_location=restaurant.location,
            delivery_location=(delivery_lat, delivery_lon),
            created_time=created_time,
            prep_time_minutes=prep_time,
            deadline=deadline,
            priority=priority,
        )
