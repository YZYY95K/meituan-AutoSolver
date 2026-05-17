from __future__ import annotations
import json
from typing import Any
from ..models.scenario import DeliveryScenario
from ..algorithms.greedy import AllocationResult


class DeliveryVisualizer:
    @staticmethod
    def generate_map_html(
        scenario: DeliveryScenario,
        result: AllocationResult | None = None,
        output_path: str = "delivery_map.html",
    ) -> str:
        center_lat = sum(scenario.area_bounds[i] for i in [0, 2]) / 2
        center_lon = sum(scenario.area_bounds[i] for i in [1, 3]) / 2

        restaurants_json = json.dumps([
            {"id": r.restaurant_id, "lat": r.location[0], "lon": r.location[1], "name": r.name}
            for r in scenario.restaurants
        ])

        orders_json = json.dumps([
            {
                "id": o.order_id,
                "rest_lat": o.restaurant_location[0],
                "rest_lon": o.restaurant_location[1],
                "del_lat": o.delivery_location[0],
                "del_lon": o.delivery_location[1],
                "priority": o.priority.value,
                "urgent": o.is_urgent,
            }
            for o in scenario.orders
        ])

        riders_json = json.dumps([
            {"id": r.rider_id, "lat": r.location[0], "lon": r.location[1]}
            for r in scenario.riders
        ])

        routes_json = "{}"
        assignments_json = "{}"
        if result:
            routes_json = json.dumps({
                k: [{"lat": p[0], "lon": p[1]} for p in v]
                for k, v in result.routes.items()
            })
            assignments_json = json.dumps(result.assignments)

        colors = [
            "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
            "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9",
            "#F0B27A", "#82E0AA", "#F1948A", "#85929E", "#73C6B6",
        ]
        colors_json = json.dumps(colors)

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>AutoSolver 配送分配可视化</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        body {{ margin: 0; padding: 0; font-family: 'Microsoft YaHei', sans-serif; }}
        #map {{ width: 100%; height: 80vh; }}
        #info {{ padding: 15px; background: #f8f9fa; }}
        .stat {{ display: inline-block; margin-right: 20px; padding: 8px 15px; background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .stat-value {{ font-size: 20px; font-weight: bold; color: #2c3e50; }}
        .stat-label {{ font-size: 12px; color: #7f8c8d; }}
        .legend {{ position: absolute; bottom: 30px; right: 10px; z-index: 1000; background: white; padding: 10px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.2); }}
        .legend-item {{ margin: 5px 0; }}
        .legend-color {{ display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 5px; }}
    </style>
</head>
<body>
    <div id="info">
        <h2>🛵 AutoSolver 配送分配可视化</h2>
        <div class="stat"><div class="stat-value">{scenario.num_orders}</div><div class="stat-label">订单数</div></div>
        <div class="stat"><div class="stat-value">{scenario.num_riders}</div><div class="stat-label">骑手数</div></div>
        <div class="stat"><div class="stat-value">{scenario.num_restaurants}</div><div class="stat-label">餐厅数</div></div>
        <div class="stat"><div class="stat-value">{scenario.difficulty_level}</div><div class="stat-label">难度</div></div>
    </div>
    <div id="map"></div>
    <script>
        var map = L.map('map').setView([{center_lat}, {center_lon}], 13);
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '© OpenStreetMap'
        }}).addTo(map);

        var restaurants = {restaurants_json};
        var orders = {orders_json};
        var riders = {riders_json};
        var routes = {routes_json};
        var assignments = {assignments_json};
        var colors = {colors_json};

        var restIcon = L.divIcon({{html: '🏪', iconSize: [24, 24], className: ''}});
        var riderIcon = L.divIcon({{html: '🛵', iconSize: [24, 24], className: ''}});

        restaurants.forEach(function(r) {{
            L.marker([r.lat, r.lon], {{icon: restIcon}}).addTo(map).bindPopup(r.name);
        }});

        riders.forEach(function(r) {{
            L.marker([r.lat, r.lon], {{icon: riderIcon}}).addTo(map).bindPopup('骑手: ' + r.id);
        }});

        orders.forEach(function(o) {{
            var color = o.urgent ? '#e74c3c' : (o.priority === 'high' ? '#f39c12' : '#3498db');
            L.circleMarker([o.rest_lat, o.rest_lon], {{
                radius: 4, fillColor: color, color: '#fff', weight: 1, fillOpacity: 0.8
            }}).addTo(map);
            L.circleMarker([o.del_lat, o.del_lon], {{
                radius: 3, fillColor: color, color: '#fff', weight: 1, fillOpacity: 0.6
            }}).addTo(map);
        }});

        var riderIdx = 0;
        Object.keys(routes).forEach(function(riderId) {{
            var points = routes[riderId].map(function(p) {{ return [p.lat, p.lon]; }});
            if (points.length > 1) {{
                L.polyline(points, {{
                    color: colors[riderIdx % colors.length],
                    weight: 3,
                    opacity: 0.7,
                    dashArray: '5, 10'
                }}).addTo(map).bindPopup('骑手: ' + riderId + ' (' + (assignments[riderId] ? assignments[riderId].length : 0) + '单)');
            }}
            riderIdx++;
        }});
    </script>
</body>
</html>"""

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

        return output_path

    @staticmethod
    def generate_comparison_chart(
        results: dict[str, AllocationResult],
        output_path: str = "comparison.png",
    ) -> str:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

        algorithms = list(results.keys())
        metrics = ["on_time_rate", "avg_delivery_time", "max_load_imbalance", "solve_time_seconds"]
        labels = ["准时率", "平均配送时间(min)", "负载偏差", "求解时间(s)"]

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]

        for i, (metric, label) in enumerate(zip(metrics, labels)):
            values = [getattr(results[algo], metric) for algo in algorithms]
            if metric == "on_time_rate":
                values = [v * 100 for v in values]
                label = "准时率(%)"
            axes[i].bar(algorithms, values, color=colors[:len(algorithms)])
            axes[i].set_title(label)
            axes[i].set_ylabel(label)

        plt.suptitle("AutoSolver 算法对比", fontsize=16)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        return output_path
