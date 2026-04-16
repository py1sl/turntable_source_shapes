import numpy as np
import pytest

from turntable_source_shapes import (
    compute_density_map,
    multi_source_trace,
    random_sources_in_drum,
    rotate_point,
    rotation_matrix_2d,
    source_trace,
    volume_source_trace,
)


def test_rotation_matrix_2d_quarter_turn():
    matrix = rotation_matrix_2d(np.pi / 2)
    expected = np.array([[0.0, -1.0], [1.0, 0.0]])
    np.testing.assert_allclose(matrix, expected, atol=1e-12)


def test_rotate_point_quarter_turn():
    rotated = rotate_point([1.0, 0.0], np.pi / 2)
    np.testing.assert_allclose(rotated, np.array([0.0, 1.0]), atol=1e-12)


def test_source_trace_shapes_and_weight_sum():
    xs, ys, weights = source_trace([0.0, 0.0], [0.0, 0.0], n_steps=12, intensity=3.0)
    assert xs.shape == (12,)
    assert ys.shape == (12,)
    assert weights.shape == (12,)
    assert np.allclose(xs, 0.0)
    assert np.allclose(ys, 0.0)
    assert np.isclose(np.sum(weights), 3.0)


def test_source_trace_circle_radius_with_offset():
    xs, ys, _ = source_trace([0.2, 0.0], [0.1, 0.0], n_steps=36, intensity=1.0)
    radii = np.sqrt(xs**2 + ys**2)
    assert np.allclose(radii, 0.3, atol=1e-12)


def test_random_sources_in_drum_reproducible_and_bounded():
    positions_a, intensities_a = random_sources_in_drum(5, 0.3, total_intensity=2.0, seed=7)
    positions_b, intensities_b = random_sources_in_drum(5, 0.3, total_intensity=2.0, seed=7)
    np.testing.assert_allclose(positions_a, positions_b)
    np.testing.assert_allclose(intensities_a, intensities_b)
    radii = np.linalg.norm(positions_a, axis=1)
    assert np.all(radii <= 0.3 + 1e-12)
    assert np.isclose(np.sum(intensities_a), 2.0)
    assert np.all(intensities_a >= 0.0)


@pytest.mark.parametrize(
    "args, error_message",
    [
        ((0, 0.3, 1.0, None), "n_sources must be a positive integer."),
        ((1, 0.0, 1.0, None), "drum_radius must be positive."),
        ((1, 0.3, 0.0, None), "total_intensity must be positive."),
    ],
)
def test_random_sources_in_drum_validates_inputs(args, error_message):
    with pytest.raises(ValueError, match=error_message):
        random_sources_in_drum(*args)


def test_multi_source_trace_shapes_and_weight_sum():
    sources = np.array([[0.0, 0.0], [0.2, 0.0]])
    intensities = np.array([0.3, 0.7])
    xs, ys, weights = multi_source_trace(sources, [0.1, 0.0], intensities, n_steps=10)
    assert xs.shape == (20,)
    assert ys.shape == (20,)
    assert weights.shape == (20,)
    assert np.isclose(np.sum(weights), 1.0)
    weights_by_source = np.reshape(weights, (10, 2))
    assert np.allclose(weights_by_source[:, 0], 0.03)
    assert np.allclose(weights_by_source[:, 1], 0.07)
    assert not np.allclose(xs[:10], xs[10:])


def test_multi_source_trace_validates_inputs():
    with pytest.raises(ValueError, match="sources_in_drum must have shape"):
        multi_source_trace([1.0, 2.0], [0.0, 0.0], [1.0], n_steps=10)
    with pytest.raises(ValueError, match="intensities must have one entry per source"):
        multi_source_trace([[0.0, 0.0]], [0.0, 0.0], [1.0, 2.0], n_steps=10)
    with pytest.raises(ValueError, match="intensities must be non-negative"):
        multi_source_trace([[0.0, 0.0]], [0.0, 0.0], [-1.0], n_steps=10)
    with pytest.raises(ValueError, match="at least one source intensity must be positive"):
        multi_source_trace([[0.0, 0.0]], [0.0, 0.0], [0.0], n_steps=10)
    with pytest.raises(ValueError, match="n_steps must be a positive integer"):
        multi_source_trace([[0.0, 0.0]], [0.0, 0.0], [1.0], n_steps=0)


def test_volume_source_trace_shapes_and_weight_sum():
    xs, ys, weights = volume_source_trace(
        drum_radius=0.3,
        drum_offset=[0.1, 0.0],
        n_steps=8,
        intensity=2.5,
        n_radial=3,
        n_angular=4,
    )
    expected_size = 8 * 3 * 4
    assert xs.shape == (expected_size,)
    assert ys.shape == (expected_size,)
    assert weights.shape == (expected_size,)
    assert np.isclose(np.sum(weights), 2.5)


def test_volume_source_trace_validates_inputs():
    with pytest.raises(ValueError, match="drum_radius must be greater than zero"):
        volume_source_trace(0.0, [0.0, 0.0])
    with pytest.raises(ValueError, match="n_radial and n_angular must be positive"):
        volume_source_trace(0.3, [0.0, 0.0], n_radial=0)


def test_compute_density_map_preserves_total_weight():
    xs = np.array([0.0, 0.1, -0.1])
    ys = np.array([0.0, 0.1, -0.1])
    weights = np.array([0.2, 0.3, 0.5])
    hist, xedges, yedges = compute_density_map(xs, ys, weights, grid_size=20)
    assert hist.shape == (20, 20)
    assert xedges.shape == (21,)
    assert yedges.shape == (21,)
    assert np.isclose(np.sum(hist), np.sum(weights))
