from fastapi import (
    FastAPI,
)

from fastapi.middleware.cors import (
    CORSMiddleware,
)

from api.config import (
    API_DESCRIPTION,
    API_TITLE,
    API_VERSION,
    CORS_ORIGINS,
    DATASET_BENCHMARK_PATH,
    FINAL_VALIDATION_SUMMARY_PATH,
    KMEANS_RESULT_PATH,
    ROAD_GRAPH_PATH,
    SNAPPED_NODES_PATH,
    XEDU_DATASET_PATH,
)

from api.routers.adverse_deliveries import (
    router as adverse_deliveries_router,
)

from api.routers.clustering import (
    router as clustering_router,
)

from api.routers.deliveries import (
    router as deliveries_router,
)

from api.routers.evaluation import (
    router as evaluation_router,
)

from api.routers.results import (
    router as results_router,
)

from api.routers.routing import (
    router as routing_router,
)


app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=[
        "*",
    ],
    allow_headers=[
        "*",
    ],
)


app.include_router(
    deliveries_router
)

app.include_router(
    clustering_router
)

app.include_router(
    adverse_deliveries_router
)

app.include_router(
    routing_router
)

app.include_router(
    evaluation_router
)

app.include_router(
    results_router
)


@app.get(
    "/",
    tags=[
        "System",
    ],
)
def root() -> dict:
    return {
        "name":
            API_TITLE,

        "version":
            API_VERSION,

        "status":
            "running",

        "docs":
            "/docs",

        "health":
            "/api/health",
    }


@app.get(
    "/api/health",
    tags=[
        "System",
    ],
)
def health() -> dict:
    artifacts = {
        "dataset":
            XEDU_DATASET_PATH.exists(),

        "road_graph":
            ROAD_GRAPH_PATH.exists(),

        "snapped_nodes":
            SNAPPED_NODES_PATH.exists(),

        "kmeans_results":
            KMEANS_RESULT_PATH.exists(),

        "dataset_benchmark":
            DATASET_BENCHMARK_PATH.exists(),

        "final_validation":
            FINAL_VALIDATION_SUMMARY_PATH.exists(),
    }

    core_artifacts_ready = all(
        artifacts.values()
    )

    return {
        "status": (
            "ready"
            if core_artifacts_ready
            else "degraded"
        ),

        "api_version":
            API_VERSION,

        "artifacts":
            artifacts,

        "core_artifacts_ready":
            core_artifacts_ready,

        "ready_for_frontend":
            core_artifacts_ready,

        "available_modules": {
            "system":
                True,

            "deliveries":
                True,

            "clustering":
                True,

            "adverse_delivery":
                True,

            "routing":
                True,

            "evaluation":
                True,

            "results":
                True,
        },
    }