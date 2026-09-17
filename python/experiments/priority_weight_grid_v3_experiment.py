from pathlib import Path
from time import perf_counter

import networkx as nx
import numpy as np
import pandas as pd

from data import (
    DataLoader,
    DataPreprocessor,
    DeliveryMapper,
)

from prioritization import (
    DeliveryPriorityModel,
    PriorityWeights,
)

from routing import (
    RoadNetworkService,
    TravelEfficiencyScorer,
)

from simulation.scenario import ScenarioFactory


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "xedu"
    / "xedu_cleaned.csv"
)

GRAPH_PATH = (
    PROJECT_ROOT
    / "maps"
    / "xedu_drive.graphml"
)

SNAPPED_PATH = (
    PROJECT_ROOT
    / "results"
    / "xedu_delivery_road_nodes.csv"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)

SUMMARY_PATH = (
    RESULTS_DIR
    / "priority_weight_grid_v3_summary.csv"
)

DETAIL_PATH = (
    RESULTS_DIR
    / "priority_weight_grid_v3_details.csv"
)

WAVE_SUMMARY_PATH = (
    RESULTS_DIR
    / "priority_weight_grid_v3_wave_summary.csv"
)


WAVE_MINUTES = 120

MIN_WAVE_SIZE = 2


# ---------------------------------------------------------
# Weight-grid configuration
# ---------------------------------------------------------

DEADLINE_VALUES = [
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
]

SERVICE_VALUES = [
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
]

WAITING_VALUES = [
    0.10,
    0.15,
    0.20,
    0.25,
]

TRAVEL_VALUES = [
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
]


def load_deliveries() -> pd.DataFrame:
    dataframe = DataLoader.load_csv(
        DATASET_PATH
    )

    dataframe = (
        DataPreprocessor.prepare_deliveries(
            dataframe
        )
    )

    deliveries = (
        DeliveryMapper.from_dataframe(
            dataframe
        )
    )

    rows = []

    for delivery in deliveries:
        rows.append(
            {
                "delivery_id":
                    delivery.delivery_id,

                "latitude":
                    delivery.latitude,

                "longitude":
                    delivery.longitude,

                "weight":
                    delivery.weight,

                "created_at":
                    delivery.created_at,

                "expected_delivery_time":
                    delivery.expected_delivery_time,

                "delivered_at":
                    delivery.delivered_at,

                "service_type":
                    delivery.service_type,
            }
        )

    result = pd.DataFrame(
        rows
    )

    result = result.dropna(
        subset=[
            "created_at",
        ]
    ).reset_index(
        drop=True
    )

    return result


def load_snapped_nodes() -> pd.DataFrame:
    if not SNAPPED_PATH.exists():
        raise FileNotFoundError(
            f"Missing snapped node file: "
            f"{SNAPPED_PATH}"
        )

    dataframe = pd.read_csv(
        SNAPPED_PATH
    )

    dataframe[
        "delivery_id"
    ] = dataframe[
        "delivery_id"
    ].astype(
        str
    )

    dataframe[
        "road_node"
    ] = dataframe[
        "road_node"
    ].astype(
        int
    )

    return dataframe[
        [
            "delivery_id",
            "road_node",
        ]
    ]


def merge_road_nodes(
    deliveries: pd.DataFrame,
    snapped: pd.DataFrame,
) -> pd.DataFrame:
    result = deliveries.merge(
        snapped,
        on="delivery_id",
        how="left",
    )

    missing = int(
        result[
            "road_node"
        ].isna().sum()
    )

    if missing > 0:
        raise ValueError(
            f"{missing} deliveries "
            "do not have road nodes."
        )

    result[
        "road_node"
    ] = result[
        "road_node"
    ].astype(
        int
    )

    return result


def assign_waves(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    dataframe = dataframe.copy()

    frequency = (
        f"{WAVE_MINUTES}min"
    )

    dataframe[
        "wave_start"
    ] = dataframe[
        "created_at"
    ].dt.floor(
        frequency
    )

    dataframe[
        "wave_end"
    ] = (
        dataframe[
            "wave_start"
        ]
        + pd.Timedelta(
            minutes=WAVE_MINUTES
        )
    )

    return dataframe


def load_road_network():
    if not GRAPH_PATH.exists():
        raise FileNotFoundError(
            f"Missing graph: "
            f"{GRAPH_PATH}"
        )

    print()
    print(
        "Loading cached road graph..."
    )

    service = (
        RoadNetworkService.load_graphml(
            GRAPH_PATH
        )
    )

    print(
        f"Road nodes: "
        f"{service.graph.number_of_nodes()}"
    )

    print(
        f"Road edges: "
        f"{service.graph.number_of_edges()}"
    )

    return service


def get_depot_node(
    road_service: RoadNetworkService,
) -> tuple:
    """
    Use the simulated operational depot.

    XeDu does not provide an authoritative
    enterprise depot, so this remains a
    simulation assumption.
    """

    scenario = (
        ScenarioFactory
        .create_single_depot_scenario()
    )

    depot = scenario.depots[
        0
    ]

    depot_node = (
        road_service.nearest_node(
            latitude=depot.latitude,
            longitude=depot.longitude,
        )
    )

    return (
        depot,
        depot_node,
    )


def calculate_origin_road_metrics(
    graph,
    origin_node: int,
) -> tuple[
    dict,
    dict,
]:
    """
    Run Dijkstra only once for distance and
    once for travel time from the depot.

    This is much faster than repeatedly
    solving individual origin-destination paths.
    """

    distances = (
        nx.single_source_dijkstra_path_length(
            graph,
            source=origin_node,
            weight="length",
        )
    )

    travel_times = (
        nx.single_source_dijkstra_path_length(
            graph,
            source=origin_node,
            weight="travel_time",
        )
    )

    return (
        distances,
        travel_times,
    )


def calculate_wave_travel_efficiency(
    wave_df: pd.DataFrame,
    distance_lookup: dict,
    travel_lookup: dict,
) -> pd.DataFrame:
    """
    Calculate road-based travel efficiency
    from the simulated depot to candidates
    inside one planning wave.
    """

    wave_df = wave_df.copy()

    distances = []
    travel_times = []

    for road_node in wave_df[
        "road_node"
    ]:
        distance_m = (
            distance_lookup.get(
                int(
                    road_node
                ),
                np.inf,
            )
        )

        travel_seconds = (
            travel_lookup.get(
                int(
                    road_node
                ),
                np.inf,
            )
        )

        distances.append(
            (
                float(
                    distance_m
                )
                / 1000.0
            )
            if np.isfinite(
                distance_m
            )
            else np.inf
        )

        travel_times.append(
            (
                float(
                    travel_seconds
                )
                / 60.0
            )
            if np.isfinite(
                travel_seconds
            )
            else np.inf
        )

    scorer = (
        TravelEfficiencyScorer()
    )

    efficiency = (
        scorer.score_candidates(
            distance_km=distances,
            travel_time_minutes=(
                travel_times
            ),
        )
    )

    wave_df[
        "road_distance_km"
    ] = distances

    wave_df[
        "road_travel_time_minutes"
    ] = travel_times

    wave_df[
        "distance_efficiency"
    ] = (
        efficiency
        .distance_efficiency
    )

    wave_df[
        "time_efficiency"
    ] = (
        efficiency
        .time_efficiency
    )

    wave_df[
        "travel_efficiency"
    ] = (
        efficiency
        .travel_efficiency
    )

    return wave_df


def prepare_wave_data(
    dataframe: pd.DataFrame,
    distance_lookup: dict,
    travel_lookup: dict,
) -> pd.DataFrame:
    """
    Calculate real road features once.

    Weight-grid experiments reuse these
    values and therefore do not rerun
    Dijkstra for every configuration.
    """

    frames = []

    grouped = dataframe.groupby(
        "wave_start",
        sort=True,
    )

    total_waves = dataframe[
        "wave_start"
    ].nunique()

    processed = 0
    skipped = 0

    for wave_start, wave_df in grouped:
        if len(
            wave_df
        ) < MIN_WAVE_SIZE:
            skipped += 1
            continue

        processed += 1

        if (
            processed == 1
            or processed % 100 == 0
        ):
            print(
                f"Preparing road features "
                f"for wave {processed}: "
                f"{wave_start}"
            )

        prepared = (
            calculate_wave_travel_efficiency(
                wave_df=wave_df,
                distance_lookup=(
                    distance_lookup
                ),
                travel_lookup=(
                    travel_lookup
                ),
            )
        )

        frames.append(
            prepared
        )

    if not frames:
        raise ValueError(
            "No valid multi-order waves."
        )

    print()
    print(
        f"Total original waves: "
        f"{total_waves}"
    )

    print(
        f"Evaluated waves: "
        f"{processed}"
    )

    print(
        f"Skipped singleton waves: "
        f"{skipped}"
    )

    return pd.concat(
        frames,
        ignore_index=True,
    )


def create_delivery_from_row(
    row: pd.Series,
):
    from domain import Delivery

    return Delivery(
        delivery_id=str(
            row[
                "delivery_id"
            ]
        ),

        latitude=float(
            row[
                "latitude"
            ]
        ),

        longitude=float(
            row[
                "longitude"
            ]
        ),

        weight=(
            None
            if pd.isna(
                row[
                    "weight"
                ]
            )
            else float(
                row[
                    "weight"
                ]
            )
        ),

        created_at=row[
            "created_at"
        ],

        expected_delivery_time=row[
            "expected_delivery_time"
        ],

        delivered_at=row[
            "delivered_at"
        ],

        service_type=row[
            "service_type"
        ],
    )


def generate_weight_grid():
    """
    Generate valid 4-factor weight combinations.

    Requirements:
        all weights > 0
        sum = 1.0
    """

    configurations = []

    for deadline in DEADLINE_VALUES:
        for service in SERVICE_VALUES:
            for waiting in WAITING_VALUES:
                for travel in TRAVEL_VALUES:

                    total = (
                        deadline
                        + service
                        + waiting
                        + travel
                    )

                    if not np.isclose(
                        total,
                        1.0,
                    ):
                        continue

                    name = (
                        f"D{int(deadline * 100):02d}_"
                        f"S{int(service * 100):02d}_"
                        f"W{int(waiting * 100):02d}_"
                        f"T{int(travel * 100):02d}"
                    )

                    weights = PriorityWeights(
                        deadline_urgency=(
                            deadline
                        ),
                        service_priority=(
                            service
                        ),
                        waiting_time=(
                            waiting
                        ),
                        travel_efficiency=(
                            travel
                        ),
                    )

                    configurations.append(
                        (
                            name,
                            weights,
                        )
                    )

    return configurations


def safe_spearman(
    left: pd.Series,
    right: pd.Series,
) -> float:
    dataframe = pd.DataFrame(
        {
            "left": left,
            "right": right,
        }
    ).dropna()

    if len(
        dataframe
    ) < 2:
        return 0.0

    if (
        dataframe[
            "left"
        ].nunique()
        < 2
    ):
        return 0.0

    if (
        dataframe[
            "right"
        ].nunique()
        < 2
    ):
        return 0.0

    value = dataframe[
        "left"
    ].corr(
        dataframe[
            "right"
        ],
        method="spearman",
    )

    if pd.isna(
        value
    ):
        return 0.0

    return float(
        value
    )


def evaluate_configuration(
    dataframe: pd.DataFrame,
    configuration_name: str,
    weights: PriorityWeights,
) -> tuple[
    dict,
    pd.DataFrame,
]:
    model = DeliveryPriorityModel(
        weights=weights
    )

    rows = []

    grouped = dataframe.groupby(
        "wave_start",
        sort=False,
    )

    for wave_start, wave_df in grouped:

        wave_end = wave_df[
            "wave_end"
        ].iloc[
            0
        ]

        wave_rows = []

        for _, row in wave_df.iterrows():
            delivery = (
                create_delivery_from_row(
                    row
                )
            )

            result = model.calculate(
                delivery=delivery,
                current_time=wave_end,
                travel_efficiency_score=(
                    float(
                        row[
                            "travel_efficiency"
                        ]
                    )
                ),
            )

            wave_rows.append(
                {
                    "configuration":
                        configuration_name,

                    "delivery_id":
                        result.delivery_id,

                    "wave_start":
                        wave_start,

                    "wave_end":
                        wave_end,

                    "wave_size":
                        len(
                            wave_df
                        ),

                    "service_type":
                        delivery.service_type,

                    "deadline_urgency":
                        result.deadline_urgency,

                    "service_priority":
                        result.service_priority,

                    "waiting_time_score":
                        result.waiting_time_score,

                    "road_distance_km":
                        row[
                            "road_distance_km"
                        ],

                    "road_travel_time_minutes":
                        row[
                            "road_travel_time_minutes"
                        ],

                    "travel_efficiency":
                        row[
                            "travel_efficiency"
                        ],

                    "information_coverage":
                        result.information_coverage,

                    "priority_score":
                        result.priority_score,
                }
            )

        wave_result = (
            pd.DataFrame(
                wave_rows
            )
            .sort_values(
                by=[
                    "priority_score",
                    "deadline_urgency",
                    "travel_efficiency",
                    "delivery_id",
                ],
                ascending=[
                    False,
                    False,
                    False,
                    True,
                ],
                na_position="last",
            )
            .reset_index(
                drop=True
            )
        )

        wave_result[
            "priority_rank"
        ] = (
            wave_result.index
            + 1
        )

        rows.extend(
            wave_result.to_dict(
                orient="records"
            )
        )

    details_df = pd.DataFrame(
        rows
    )

    deadline_alignment = (
        safe_spearman(
            details_df[
                "priority_score"
            ],
            details_df[
                "deadline_urgency"
            ],
        )
    )

    service_alignment = (
        safe_spearman(
            details_df[
                "priority_score"
            ],
            details_df[
                "service_priority"
            ],
        )
    )

    waiting_alignment = (
        safe_spearman(
            details_df[
                "priority_score"
            ],
            details_df[
                "waiting_time_score"
            ],
        )
    )

    travel_alignment = (
        safe_spearman(
            details_df[
                "priority_score"
            ],
            details_df[
                "travel_efficiency"
            ],
        )
    )

    alignments = np.array(
        [
            deadline_alignment,
            service_alignment,
            waiting_alignment,
            travel_alignment,
        ],
        dtype=float,
    )

    summary = {
        "configuration":
            configuration_name,

        "deadline_weight":
            weights.deadline_urgency,

        "service_weight":
            weights.service_priority,

        "waiting_weight":
            weights.waiting_time,

        "travel_weight":
            weights.travel_efficiency,

        "deadline_alignment":
            deadline_alignment,

        "service_alignment":
            service_alignment,

        "waiting_alignment":
            waiting_alignment,

        "travel_alignment":
            travel_alignment,

        "mean_alignment":
            float(
                alignments.mean()
            ),

        "alignment_std":
            float(
                alignments.std()
            ),

        "alignment_range":
            float(
                alignments.max()
                - alignments.min()
            ),

        "priority_score_std":
            float(
                details_df[
                    "priority_score"
                ].std()
            ),

        "priority_score_mean":
            float(
                details_df[
                    "priority_score"
                ].mean()
            ),

        "mean_information_coverage":
            float(
                details_df[
                    "information_coverage"
                ].mean()
            ),
    }

    return (
        summary,
        details_df,
    )


def normalize(
    series: pd.Series,
) -> pd.Series:
    minimum = float(
        series.min()
    )

    maximum = float(
        series.max()
    )

    if np.isclose(
        minimum,
        maximum,
    ):
        return pd.Series(
            np.ones(
                len(
                    series
                )
            ),
            index=series.index,
        )

    return (
        series
        - minimum
    ) / (
        maximum
        - minimum
    )


def add_balance_score(
    summary_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Experimental candidate-selection score.

    Rewards:
        high average alignment
        low imbalance among components
        useful score spread

    This is not treated as ground truth.
    """

    summary_df = summary_df.copy()

    summary_df[
        "normalized_alignment"
    ] = normalize(
        summary_df[
            "mean_alignment"
        ]
    )

    summary_df[
        "normalized_balance"
    ] = (
        1.0
        - normalize(
            summary_df[
                "alignment_range"
            ]
        )
    )

    summary_df[
        "normalized_spread"
    ] = normalize(
        summary_df[
            "priority_score_std"
        ]
    )

    summary_df[
        "balance_score"
    ] = (
        0.40
        * summary_df[
            "normalized_alignment"
        ]
        + 0.40
        * summary_df[
            "normalized_balance"
        ]
        + 0.20
        * summary_df[
            "normalized_spread"
        ]
    )

    return summary_df


def build_wave_summary(
    prepared_df: pd.DataFrame,
) -> pd.DataFrame:
    summary = (
        prepared_df
        .groupby(
            "wave_start"
        )
        .agg(
            wave_size=(
                "delivery_id",
                "size",
            ),

            mean_road_distance_km=(
                "road_distance_km",
                "mean",
            ),

            mean_travel_time_minutes=(
                "road_travel_time_minutes",
                "mean",
            ),

            mean_travel_efficiency=(
                "travel_efficiency",
                "mean",
            ),
        )
        .reset_index()
    )

    return summary


def run_experiment():
    print()
    print(
        "========================================"
    )
    print(
        "Priority Weight Grid V3"
    )
    print(
        "========================================"
    )

    start_time = perf_counter()

    deliveries = (
        load_deliveries()
    )

    snapped = (
        load_snapped_nodes()
    )

    dataframe = (
        merge_road_nodes(
            deliveries,
            snapped,
        )
    )

    dataframe = (
        assign_waves(
            dataframe
        )
    )

    print(
        f"Deliveries: "
        f"{len(dataframe)}"
    )

    print(
        f"Wave duration: "
        f"{WAVE_MINUTES} minutes"
    )

    road_service = (
        load_road_network()
    )

    (
        depot,
        depot_node,
    ) = get_depot_node(
        road_service
    )

    print()
    print(
        "Simulated depot:"
    )

    print(
        f"  ID: "
        f"{depot.depot_id}"
    )

    print(
        f"  Latitude: "
        f"{depot.latitude}"
    )

    print(
        f"  Longitude: "
        f"{depot.longitude}"
    )

    print(
        f"  Road node: "
        f"{depot_node}"
    )

    print()
    print(
        "Calculating road metrics "
        "from depot..."
    )

    (
        distance_lookup,
        travel_lookup,
    ) = calculate_origin_road_metrics(
        road_service.graph,
        depot_node,
    )

    prepared_df = (
        prepare_wave_data(
            dataframe=dataframe,
            distance_lookup=(
                distance_lookup
            ),
            travel_lookup=(
                travel_lookup
            ),
        )
    )

    configurations = (
        generate_weight_grid()
    )

    print()
    print(
        f"Weight configurations: "
        f"{len(configurations)}"
    )

    summaries = []
    detail_frames = []

    for index, (
        configuration_name,
        weights,
    ) in enumerate(
        configurations,
        start=1,
    ):
        print(
            f"Processing "
            f"{index}/"
            f"{len(configurations)}: "
            f"{configuration_name}"
        )

        (
            summary,
            details_df,
        ) = evaluate_configuration(
            dataframe=prepared_df,
            configuration_name=(
                configuration_name
            ),
            weights=weights,
        )

        summaries.append(
            summary
        )

        detail_frames.append(
            details_df
        )

    summary_df = pd.DataFrame(
        summaries
    )

    summary_df = add_balance_score(
        summary_df
    )

    summary_df = (
        summary_df
        .sort_values(
            by=[
                "balance_score",
                "alignment_range",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    summary_df[
        "candidate_rank"
    ] = (
        summary_df.index
        + 1
    )

    details_df = pd.concat(
        detail_frames,
        ignore_index=True,
    )

    wave_summary_df = (
        build_wave_summary(
            prepared_df
        )
    )

    runtime = (
        perf_counter()
        - start_time
    )

    print()
    print(
        f"Total runtime: "
        f"{runtime:.2f} seconds"
    )

    return (
        summary_df,
        details_df,
        wave_summary_df,
    )


def print_results(
    summary_df: pd.DataFrame,
) -> None:
    print()
    print(
        "========================================"
    )
    print(
        "Top 15 Final Weight Candidates"
    )
    print(
        "========================================"
    )

    columns = [
        "candidate_rank",
        "configuration",
        "deadline_weight",
        "service_weight",
        "waiting_weight",
        "travel_weight",
        "deadline_alignment",
        "service_alignment",
        "waiting_alignment",
        "travel_alignment",
        "mean_alignment",
        "alignment_range",
        "priority_score_std",
        "balance_score",
    ]

    print(
        summary_df[
            columns
        ]
        .head(
            15
        )
        .to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    best = summary_df.iloc[
        0
    ]

    print()
    print(
        "========================================"
    )
    print(
        "Top Final Candidate"
    )
    print(
        "========================================"
    )

    print(
        f"Configuration: "
        f"{best['configuration']}"
    )

    print(
        f"Deadline: "
        f"{best['deadline_weight']:.2f}"
    )

    print(
        f"Service: "
        f"{best['service_weight']:.2f}"
    )

    print(
        f"Waiting: "
        f"{best['waiting_weight']:.2f}"
    )

    print(
        f"Travel: "
        f"{best['travel_weight']:.2f}"
    )

    print(
        f"Balance score: "
        f"{best['balance_score']:.4f}"
    )

    print()
    print(
        "NOTE:"
    )

    print(
        "The selected candidate is an "
        "experimental result, not an "
        "official XeDu business rule."
    )


def save_results(
    summary_df: pd.DataFrame,
    details_df: pd.DataFrame,
    wave_summary_df: pd.DataFrame,
) -> None:
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_df.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    details_df.to_csv(
        DETAIL_PATH,
        index=False,
    )

    wave_summary_df.to_csv(
        WAVE_SUMMARY_PATH,
        index=False,
    )

    print()
    print(
        "Saved:"
    )

    print(
        SUMMARY_PATH
    )

    print(
        DETAIL_PATH
    )

    print(
        WAVE_SUMMARY_PATH
    )


def main():
    (
        summary_df,
        details_df,
        wave_summary_df,
    ) = run_experiment()

    print_results(
        summary_df
    )

    save_results(
        summary_df,
        details_df,
        wave_summary_df,
    )


if __name__ == "__main__":
    main()