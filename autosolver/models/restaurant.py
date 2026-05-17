from dataclasses import dataclass
import numpy as np


@dataclass
class Restaurant:
    restaurant_id: str
    name: str
    location: tuple[float, float]
    avg_prep_time: float = 20.0
    order_count: int = 0

    @staticmethod
    def generate_random(
        restaurant_id: str,
        area_bounds: tuple[float, float, float, float],
        rng: np.random.Generator | None = None,
    ) -> "Restaurant":
        rng = rng or np.random.default_rng()
        lat_min, lon_min, lat_max, lon_max = area_bounds
        lat = rng.uniform(lat_min, lat_max)
        lon = rng.uniform(lon_min, lon_max)
        avg_prep = rng.uniform(10, 30)
        return Restaurant(
            restaurant_id=restaurant_id,
            name=f"Restaurant_{restaurant_id}",
            location=(lat, lon),
            avg_prep_time=avg_prep,
        )
