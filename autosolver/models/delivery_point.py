from dataclasses import dataclass


@dataclass
class DeliveryPoint:
    point_id: str
    location: tuple[float, float]
    point_type: str
    estimated_service_time: float = 3.0
