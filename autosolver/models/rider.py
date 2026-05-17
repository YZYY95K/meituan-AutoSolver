from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np


class RiderStatus(str, Enum):
    IDLE = "idle"
    HEADING_TO_RESTAURANT = "heading_to_restaurant"
    WAITING_AT_RESTAURANT = "waiting_at_restaurant"
    DELIVERING = "delivering"
    RETURNING = "returning"


@dataclass
class Rider:
    rider_id: str
    location: tuple[float, float]
    status: RiderStatus = RiderStatus.IDLE
    speed_m_per_min: float = 400.0
    max_orders: int = 5
    current_orders: list[str] = field(default_factory=list)
    completed_orders: int = 0
    total_distance: float = 0.0
    total_delivery_time: float = 0.0
    on_time_rate: float = 1.0

    @property
    def available_capacity(self) -> int:
        return self.max_orders - len(self.current_orders)

    @property
    def is_available(self) -> bool:
        return self.available_capacity > 0

    @property
    def load_factor(self) -> float:
        return len(self.current_orders) / self.max_orders

    def estimate_travel_time(self, target: tuple[float, float]) -> float:
        dist = haversine_distance(self.location, target)
        return dist / self.speed_m_per_min * 60

    @staticmethod
    def generate_random(
        rider_id: str,
        area_bounds: tuple[float, float, float, float],
        rng: np.random.Generator | None = None,
    ) -> "Rider":
        rng = rng or np.random.default_rng()
        lat_min, lon_min, lat_max, lon_max = area_bounds
        lat = rng.uniform(lat_min, lat_max)
        lon = rng.uniform(lon_min, lon_max)
        speed = rng.uniform(300, 500)
        max_orders = rng.choice([3, 4, 5, 6], p=[0.2, 0.3, 0.3, 0.2])
        return Rider(
            rider_id=rider_id,
            location=(lat, lon),
            speed_m_per_min=speed,
            max_orders=max_orders,
        )


def haversine_distance(
    point1: tuple[float, float],
    point2: tuple[float, float],
) -> float:
    lat1, lon1 = np.radians(point1)
    lat2, lon2 = np.radians(point2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371000
    return c * r
