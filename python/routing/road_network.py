from pathlib import Path

import networkx as nx
import osmnx as ox


class RoadNetworkService:
    """
    Quản lý mạng đường phục vụ routing.

    Trách nhiệm:
    - tải graph từ OpenStreetMap;
    - lưu/load graph bằng GraphML;
    - ánh xạ tọa độ GPS sang road node;
    - tính shortest road distance;
    - tính estimated travel time.

    Không chứa logic Nearest Neighbor,
    Clarke-Wright hoặc Reinforcement Learning.
    """

    def __init__(
        self,
        graph: nx.MultiDiGraph,
    ) -> None:
        if graph.number_of_nodes() == 0:
            raise ValueError(
                "Road graph không được rỗng."
            )

        self.graph = graph

    @classmethod
    def from_place(
        cls,
        place_name: str,
        network_type: str = "drive",
    ) -> "RoadNetworkService":
        """
        Tải mạng đường từ OpenStreetMap.

        Ví dụ:
        Ho Chi Minh City, Vietnam
        """

        if not place_name.strip():
            raise ValueError(
                "place_name không được rỗng."
            )

        graph = ox.graph.graph_from_place(
            place_name,
            network_type=network_type,
            simplify=True,
            retain_all=False,
        )

        return cls(graph)

    @classmethod
    def load_graphml(
        cls,
        file_path: str | Path,
    ) -> "RoadNetworkService":
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy road graph: {path}"
            )

        graph = ox.io.load_graphml(
            filepath=path
        )

        return cls(graph)

    def save_graphml(
        self,
        file_path: str | Path,
    ) -> None:
        path = Path(file_path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        ox.io.save_graphml(
            self.graph,
            filepath=path,
        )

    def add_estimated_travel_times(
        self,
        highway_speeds: dict[str, float] | None = None,
        fallback_kph: float = 30.0,
    ) -> None:
        """
        Bổ sung:
        - speed_kph
        - travel_time (seconds)

        Đây là estimated/free-flow travel time,
        KHÔNG phải realtime traffic.
        """

        if fallback_kph <= 0:
            raise ValueError(
                "fallback_kph phải lớn hơn 0."
            )

        self.graph = ox.routing.add_edge_speeds(
            self.graph,
            hwy_speeds=highway_speeds,
            fallback=fallback_kph,
        )

        self.graph = ox.routing.add_edge_travel_times(
            self.graph
        )

    def nearest_node(
        self,
        latitude: float,
        longitude: float,
    ) -> int:
        """
        Tìm road node gần nhất với tọa độ GPS.

        OSMnx yêu cầu:
        X = longitude
        Y = latitude
        """

        node_id = ox.distance.nearest_nodes(
            self.graph,
            X=longitude,
            Y=latitude,
        )

        return int(node_id)

    def shortest_distance_m(
        self,
        origin_node: int,
        destination_node: int,
    ) -> float:
        """
        Shortest-path theo chiều dài đường.

        Kết quả: mét.
        """

        distance = nx.shortest_path_length(
            self.graph,
            source=origin_node,
            target=destination_node,
            weight="length",
        )

        return float(distance)

    def shortest_distance_km(
        self,
        origin_node: int,
        destination_node: int,
    ) -> float:
        return (
            self.shortest_distance_m(
                origin_node,
                destination_node,
            )
            / 1000.0
        )

    def shortest_travel_time_seconds(
        self,
        origin_node: int,
        destination_node: int,
    ) -> float:
        """
        Shortest-path theo estimated travel_time.
        """

        self._validate_travel_time()

        travel_time = nx.shortest_path_length(
            self.graph,
            source=origin_node,
            target=destination_node,
            weight="travel_time",
        )

        return float(travel_time)

    def shortest_travel_time_minutes(
        self,
        origin_node: int,
        destination_node: int,
    ) -> float:
        return (
            self.shortest_travel_time_seconds(
                origin_node,
                destination_node,
            )
            / 60.0
        )

    def shortest_path_nodes(
        self,
        origin_node: int,
        destination_node: int,
        weight: str = "length",
    ) -> list[int]:
        """
        Trả về danh sách road nodes trên tuyến tối ưu.
        """

        path = nx.shortest_path(
            self.graph,
            source=origin_node,
            target=destination_node,
            weight=weight,
        )

        return [
            int(node_id)
            for node_id in path
        ]

    def _validate_travel_time(self) -> None:
        """
        Đảm bảo graph đã được bổ sung travel_time.
        """

        for _, _, data in self.graph.edges(
            data=True
        ):
            if "travel_time" not in data:
                raise ValueError(
                    "Graph chưa có travel_time. "
                    "Hãy gọi "
                    "add_estimated_travel_times() trước."
                )

            return

        raise ValueError(
            "Road graph không có edge."
        )