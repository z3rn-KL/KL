from dataclasses import (
    dataclass,
)

from datetime import (
    datetime,
    timezone,
)

from functools import (
    lru_cache,
)

import pandas as pd

from api.config import (
    DATASET_WORKLOADS_PATH,
    ROAD_GRAPH_PATH,
    SNAPPED_NODES_PATH,
    XEDU_DATASET_PATH,
)

from data import (
    DataLoader,
    DataPreprocessor,
    DeliveryMapper,
)

from domain import (
    Delivery,
)

from evaluation import (
    RouteEvaluator,
)

from evaluation.operating_cost import (
    OperatingCostConfig,
)

from evaluation.route_evaluation_service import (
    RouteEvaluationService,
)

from routing import (
    NearestNeighborRouter,
    RoadNetworkService,
)

from routing.clarke_wright import (
    ClarkeWrightRouter,
)

from simulation.scenario import (
    ScenarioFactory,
)


@dataclass(frozen=True)
class DeliveryContext:
    dataframe: pd.DataFrame

    delivery_lookup: dict[
        str,
        Delivery,
    ]

    road_node_lookup: dict[
        str,
        int,
    ]


@dataclass(frozen=True)
class DepotContext:
    depot_id: str

    name: str

    latitude: float

    longitude: float

    road_node: int


def normalize_datetime(
    value: datetime,
) -> datetime:
    if value.tzinfo is None:
        return value.replace(
            tzinfo=timezone.utc
        )

    return value.astimezone(
        timezone.utc
    )


# ============================================================
# DELIVERY DATA
# ============================================================

@lru_cache(maxsize=1)
def get_delivery_context(
) -> DeliveryContext:
    dataframe = (
        DataLoader.load_csv(
            XEDU_DATASET_PATH
        )
    )

    dataframe = (
        DataPreprocessor
        .prepare_deliveries(
            dataframe
        )
    )

    deliveries = (
        DeliveryMapper
        .from_dataframe(
            dataframe
        )
    )

    delivery_lookup = {
        str(
            delivery.delivery_id
        ):
            delivery

        for delivery
        in deliveries
    }

    snapped = pd.read_csv(
        SNAPPED_NODES_PATH
    )

    snapped[
        "delivery_id"
    ] = (
        snapped[
            "delivery_id"
        ]
        .astype(
            str
        )
        .str
        .strip()
    )

    snapped[
        "road_node"
    ] = pd.to_numeric(
        snapped[
            "road_node"
        ],
        errors="raise",
    ).astype(
        int
    )

    road_node_lookup = dict(
        zip(
            snapped[
                "delivery_id"
            ],
            snapped[
                "road_node"
            ],
            strict=False,
        )
    )

    missing = [
        delivery_id

        for delivery_id
        in delivery_lookup

        if delivery_id
        not in road_node_lookup
    ]

    if missing:
        raise ValueError(
            f"{len(missing)} deliveries "
            "do not have SCC road nodes."
        )

    return DeliveryContext(
        dataframe=dataframe,
        delivery_lookup=(
            delivery_lookup
        ),
        road_node_lookup=(
            road_node_lookup
        ),
    )


def resolve_deliveries(
    delivery_ids: (
        list[str]
        | tuple[str, ...]
    ),
) -> tuple[
    list[Delivery],
    list[int],
]:
    normalized = [
        str(
            value
        ).strip()

        for value
        in delivery_ids
    ]

    if (
        not normalized
        or any(
            not value
            for value
            in normalized
        )
    ):
        raise ValueError(
            "delivery_ids cannot be empty."
        )

    if (
        len(
            normalized
        )
        != len(
            set(
                normalized
            )
        )
    ):
        raise ValueError(
            "delivery_ids must be unique."
        )

    context = (
        get_delivery_context()
    )

    missing = [
        value

        for value
        in normalized

        if value
        not in context.delivery_lookup
    ]

    if missing:
        raise ValueError(
            "Unknown delivery_id(s): "
            + ", ".join(
                missing[
                    :5
                ]
            )
        )

    deliveries = [
        context.delivery_lookup[
            value
        ]

        for value
        in normalized
    ]

    nodes = [
        int(
            context
            .road_node_lookup[
                value
            ]
        )

        for value
        in normalized
    ]

    return (
        deliveries,
        nodes,
    )


# ============================================================
# ROAD NETWORK / DEPOT
# ============================================================

@lru_cache(maxsize=1)
def get_road_network(
) -> RoadNetworkService:
    return (
        RoadNetworkService
        .load_graphml(
            ROAD_GRAPH_PATH
        )
    )


@lru_cache(maxsize=1)
def get_depot_context(
) -> DepotContext:
    scenario = (
        ScenarioFactory
        .create_single_depot_scenario()
    )

    depot = (
        scenario.depots[
            0
        ]
    )

    service = (
        get_road_network()
    )

    road_node = (
        service.nearest_node(
            latitude=(
                depot.latitude
            ),
            longitude=(
                depot.longitude
            ),
        )
    )

    return DepotContext(
        depot_id=str(
            depot.depot_id
        ),
        name=str(
            depot.name
        ),
        latitude=float(
            depot.latitude
        ),
        longitude=float(
            depot.longitude
        ),
        road_node=int(
            road_node
        ),
    )


# ============================================================
# WORKLOAD MANIFEST
# ============================================================

@lru_cache(maxsize=1)
def get_workload_manifest(
) -> pd.DataFrame:
    dataframe = pd.read_csv(
        DATASET_WORKLOADS_PATH
    )

    dataframe[
        "workload_id"
    ] = pd.to_numeric(
        dataframe[
            "workload_id"
        ],
        errors="raise",
    ).astype(
        int
    )

    return dataframe


def parse_manifest_delivery_ids(
    value,
) -> list[str]:
    return [
        item.strip()

        for item
        in str(
            value
        ).split(
            "->"
        )

        if item.strip()
    ]


def get_workload_row(
    workload_id: int,
) -> pd.Series:
    dataframe = (
        get_workload_manifest()
    )

    matched = dataframe[
        dataframe[
            "workload_id"
        ]
        == int(
            workload_id
        )
    ]

    if (
        len(
            matched
        )
        != 1
    ):
        raise ValueError(
            f"Unknown workload_id: "
            f"{workload_id}"
        )

    return matched.iloc[
        0
    ]


def find_exact_workload(
    delivery_ids: (
        list[str]
        | tuple[str, ...]
    ),
) -> pd.Series | None:
    normalized = [
        str(
            value
        ).strip()

        for value
        in delivery_ids
    ]

    target = set(
        normalized
    )

    if (
        len(
            target
        )
        != len(
            normalized
        )
    ):
        return None

    for _, row in (
        get_workload_manifest()
        .iterrows()
    ):
        ids = (
            parse_manifest_delivery_ids(
                row[
                    "delivery_ids"
                ]
            )
        )

        if (
            len(
                ids
            )
            == len(
                normalized
            )
            and set(
                ids
            )
            == target
        ):
            return row

    return None


def resolve_route_start_time(
    deliveries: list[Delivery],
    delivery_ids: (
        list[str]
        | tuple[str, ...]
    ),
    explicit_start_time: (
        datetime
        | None
    ),
) -> tuple[
    datetime,
    str,
]:
    if (
        explicit_start_time
        is not None
    ):
        return (
            normalize_datetime(
                explicit_start_time
            ),
            "request",
        )

    workload = (
        find_exact_workload(
            delivery_ids
        )
    )

    if workload is not None:
        value = (
            pd.Timestamp(
                workload[
                    "wave_end"
                ]
            )
            .to_pydatetime()
        )

        return (
            normalize_datetime(
                value
            ),
            "benchmark_wave_end",
        )

    created_times = [
        delivery.created_at

        for delivery
        in deliveries

        if delivery.created_at
        is not None
    ]

    if not created_times:
        raise ValueError(
            "start_time is required because "
            "it cannot be inferred from the "
            "selected deliveries."
        )

    return (
        max(
            normalize_datetime(
                value
            )
            for value
            in created_times
        ),
        "latest_created_at",
    )


# ============================================================
# COST MODEL
# ============================================================

@lru_cache(maxsize=1)
def get_default_cost_config(
) -> OperatingCostConfig:
    """
    Scenario proxy only.

    The scenario currently defines:
        cost_per_km = 4000
        cost_per_hour = 40000

    These must not be presented as verified
    XeDu accounting data.
    """

    scenario = (
        ScenarioFactory
        .create_single_depot_scenario()
    )

    vehicle = (
        scenario.vehicles[
            0
        ]
    )

    return OperatingCostConfig(
        distance_cost_per_km=float(
            vehicle.cost_per_km
        ),
        travel_time_cost_per_minute=(
            float(
                vehicle.cost_per_hour
            )
            / 60.0
        ),
        late_delivery_penalty=0.0,
        lateness_cost_per_minute=0.0,
        cost_unit=(
            "VND_proxy_scenario"
        ),
    )


# ============================================================
# ROUTERS / EVALUATORS
# ============================================================

@lru_cache(maxsize=1)
def get_route_evaluator(
) -> RouteEvaluator:
    return RouteEvaluator(
        road_network=(
            get_road_network()
        )
    )


def get_route_evaluation_service(
    cost_config: (
        OperatingCostConfig
    ),
) -> RouteEvaluationService:
    return RouteEvaluationService(
        route_evaluator=(
            get_route_evaluator()
        ),
        road_network=(
            get_road_network()
        ),
        cost_config=(
            cost_config
        ),
    )


@lru_cache(maxsize=2)
def get_nearest_neighbor_router(
    metric: str,
) -> NearestNeighborRouter:
    return NearestNeighborRouter(
        road_network=(
            get_road_network()
        ),
        metric=metric,
    )


@lru_cache(maxsize=1)
def get_clarke_wright_router(
) -> ClarkeWrightRouter:
    return ClarkeWrightRouter(
        road_network=(
            get_road_network()
        )
    )