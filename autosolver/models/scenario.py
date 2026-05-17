from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from .order import Order
from .rider import Rider
from .restaurant import Restaurant


@dataclass
class DeliveryScenario:
    scenario_id: str
    name: str
    description: str
    area_bounds: tuple[float, float, float, float]
    restaurants: list[Restaurant] = field(default_factory=list)
    orders: list[Order] = field(default_factory=list)
    riders: list[Rider] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def num_orders(self) -> int:
        return len(self.orders)

    @property
    def num_riders(self) -> int:
        return len(self.riders)

    @property
    def num_restaurants(self) -> int:
        return len(self.restaurants)

    @property
    def order_rider_ratio(self) -> float:
        return len(self.orders) / max(1, len(self.riders))

    @property
    def difficulty_level(self) -> str:
        ratio = self.order_rider_ratio
        if ratio <= 2:
            return "easy"
        elif ratio <= 4:
            return "medium"
        elif ratio <= 6:
            return "hard"
        else:
            return "extreme"

    @staticmethod
    def create_scenario(
        scenario_id: str,
        name: str,
        description: str,
        area_bounds: tuple[float, float, float, float],
        num_restaurants: int = 10,
        num_orders: int = 50,
        num_riders: int = 15,
        seed: int = 42,
    ) -> "DeliveryScenario":
        rng = np.random.default_rng(seed)
        restaurants = [
            Restaurant.generate_random(f"R{i:03d}", area_bounds, rng)
            for i in range(num_restaurants)
        ]
        base_time = datetime.now()
        orders = [
            Order.generate_random(
                f"O{i:04d}",
                restaurants,
                area_bounds,
                base_time + timedelta(seconds=int(rng.uniform(0, 600))),
                rng,
            )
            for i in range(num_orders)
        ]
        riders = [
            Rider.generate_random(f"D{i:03d}", area_bounds, rng)
            for i in range(num_riders)
        ]
        return DeliveryScenario(
            scenario_id=scenario_id,
            name=name,
            description=description,
            area_bounds=area_bounds,
            restaurants=restaurants,
            orders=orders,
            riders=riders,
        )


from datetime import timedelta
