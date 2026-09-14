import numpy as np

from clustering import LocalGeoProjection


def test_reference_point_projects_to_origin():
    coordinates = np.array(
        [
            [10.77, 106.70],
        ],
        dtype=np.float64,
    )

    projection = LocalGeoProjection(
        reference_latitude=10.77,
        reference_longitude=106.70,
    )

    projected = projection.transform(
        coordinates
    )

    assert projected.shape == (1, 2)

    x, y = projected[0]

    assert abs(x) < 1e-9
    assert abs(y) < 1e-9


def test_from_coordinates():
    coordinates = np.array(
        [
            [10.77, 106.70],
            [10.78, 106.71],
            [10.79, 106.72],
        ],
        dtype=np.float64,
    )

    projection = LocalGeoProjection.from_coordinates(
        coordinates
    )

    assert isinstance(
        projection,
        LocalGeoProjection,
    )

    assert np.isclose(
        projection.reference_latitude,
        np.mean(coordinates[:, 0]),
    )

    assert np.isclose(
        projection.reference_longitude,
        np.mean(coordinates[:, 1]),
    )


def test_transform_multiple_coordinates():
    coordinates = np.array(
        [
            [10.77, 106.70],
            [10.78, 106.71],
            [10.79, 106.72],
        ],
        dtype=np.float64,
    )

    projection = LocalGeoProjection.from_coordinates(
        coordinates
    )

    projected = projection.transform(
        coordinates
    )

    assert isinstance(
        projected,
        np.ndarray,
    )

    assert projected.shape == (3, 2)

    assert np.isfinite(
        projected
    ).all()


def test_projection_has_positive_distance():
    projection = LocalGeoProjection(
        reference_latitude=10.77,
        reference_longitude=106.70,
    )

    coordinates = np.array(
        [
            [10.78, 106.71],
        ],
        dtype=np.float64,
    )

    projected = projection.transform(
        coordinates
    )

    x, y = projected[0]

    distance_km = np.sqrt(
        x ** 2 + y ** 2
    )

    assert distance_km > 0


def test_empty_coordinates_raise_error():
    coordinates = np.empty(
        (0, 2),
        dtype=np.float64,
    )

    try:
        LocalGeoProjection.from_coordinates(
            coordinates
        )

        assert False

    except ValueError:
        assert True