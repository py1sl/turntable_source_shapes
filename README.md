# turntable_source_shapes

Derivation of source shapes for containers on a turntable, designed for
gamma-spectrometry of waste drums.

## Background

When a waste container is placed on a rotating turntable and measured with a
gamma-ray detector, point sources inside the container trace different paths in
the top-down (bird's-eye) view depending on their position:

| Configuration | Top-down shape |
|---|---|
| Source on drum axis, drum centred on turntable | Stationary point |
| Source *off* drum axis, drum centred on turntable | Circular ring (radius = source offset) |
| Source on drum axis, drum *offset* from turntable axis | Circular ring (radius = drum offset) |
| Source *off* drum axis, drum *offset* from turntable axis | Circular ring (radius = vector sum of offsets) |
| Uniform volume source in drum, drum centred on turntable | Filled disk |
| Uniform volume source in drum, drum *offset* from turntable axis | Thick annulus-like band |

The module uses a 2-D rotation matrix to compute the locus and bins the
result into a weighted intensity map to account for emission probability.

Container geometry is handled by subclasses of :class:`WasteContainer`:
* **`CylindricalDrum`** — circular cross-section defined by a radius.
* **`CuboidContainer`** — rectangular cross-section defined by width and length.

## Installation

Install from the repository root:

```bash
pip install .
```

For development tools as well:

```bash
pip install ".[dev]"
```

## Usage

```python
from turntable_source_shapes import (
    CylindricalDrum,
    CuboidContainer,
    rotation_matrix_2d,
    source_trace,
    random_sources_in_container,
    random_sources_in_drum,
    multi_source_trace,
    volume_source_trace,
    plot_cases,
)
import matplotlib.pyplot as plt

# Compute the locus for a source 15 cm off the drum axis
xs, ys, weights = source_trace(
    source_in_drum=[0.15, 0.0],   # source position relative to drum centre (m)
    drum_offset=[0.0, 0.0],       # drum centre relative to turntable axis (m)
    n_steps=3600,
    intensity=1.0,
)

# Or generate six comparison plots directly (four point-source + two volume-source)
fig = plot_cases(
    drum_radius=0.30,            # drum radius in metres (builds CylindricalDrum)
    source_offset_in_drum=0.15,  # source radial offset in drum (Case 2)
    drum_offset=0.20,            # drum offset from turntable axis (Case 3)
    n_volume_radial=18,          # volume-source radial sampling bins
    n_volume_angular=72,         # volume-source angular sampling bins
)
plt.savefig("my_plot.png", dpi=150, bbox_inches="tight")
plt.show()

# Use a cuboid container instead of a cylindrical drum
box = CuboidContainer(width=0.50, length=0.40)
fig = plot_cases(container=box, source_offset_in_drum=0.15, drum_offset=0.20)

# Volume source trace for a cylindrical drum
drum = CylindricalDrum(radius=0.30)
xs, ys, weights = volume_source_trace(
    drum,
    drum_offset=[0.20, 0.0],
    n_steps=3600,
    n_radial=18,
    n_angular=72,
)

# Volume source trace for a cuboid container
xs, ys, weights = volume_source_trace(
    box,
    drum_offset=[0.20, 0.0],
    n_steps=3600,
    n_radial=20,   # columns along width
    n_angular=16,  # rows along length
)

# Random sources in any container
source_positions, source_intensities = random_sources_in_container(
    drum, n_sources=25, total_intensity=1.0, seed=42,
)
xs, ys, weights = multi_source_trace(
    sources_in_drum=source_positions,
    drum_offset=[0.0, 0.0],
    intensities=source_intensities,
    n_steps=3600,
)

# Legacy helper for cylindrical drums (backward-compatible)
source_positions, source_intensities = random_sources_in_drum(
    n_sources=25,
    drum_radius=0.30,
    total_intensity=1.0,
    seed=42,
)

# Or append an automatic random-source case to the comparison plot
fig = plot_cases(random_n_sources=25, random_seed=42)
```

Run the script directly to produce `turntable_source_shapes.png`:

```bash
python turntable_source_shapes.py
```

## API

### Container classes

#### `WasteContainer` (abstract base class)
Defines the interface for all container geometry.  Subclasses must implement:
* `sample_volume_points(n1, n2)` — representative cross-section sample grid.
* `sample_random_points(n, rng)` — uniformly random points within the cross-section.
* `characteristic_radius` — bounding radius for display-extent calculations.
* `description` — short label string for plot legends.
* `outline_patch(offset, **kwargs)` — matplotlib `Patch` for the container outline.

#### `CylindricalDrum(radius)`
Circular cross-section drum.  `n1` = radial bins, `n2` = angular bins in
`sample_volume_points`.

#### `CuboidContainer(width, length)`
Rectangular cross-section container.  `n1` = columns along width, `n2` = rows
along length in `sample_volume_points`.

### Rotation helpers

#### `rotation_matrix_2d(theta)`
Returns the 2×2 rotation matrix for angle `theta` (radians).

#### `rotate_point(point, theta)`
Rotates a 2-D point `(x, y)` by `theta` radians about the origin.

### Turntable simulation

#### `source_trace(source_in_drum, drum_offset, n_steps=360, intensity=1.0)`
Computes the `(xs, ys, weights)` locus of a source through one full
revolution of the turntable.

#### `random_sources_in_container(container, n_sources, total_intensity=1.0, seed=None)`
Generates `n_sources` random point locations uniformly within any
`WasteContainer` cross-section and assigns random intensities that sum to
`total_intensity`.

#### `random_sources_in_drum(n_sources, drum_radius, total_intensity=1.0, seed=None)`
Convenience wrapper around `random_sources_in_container` for a
`CylindricalDrum`.

#### `multi_source_trace(sources_in_drum, drum_offset, intensities, n_steps=360)`
Computes the combined `(xs, ys, weights)` locus for multiple point sources
in one container.

#### `volume_source_trace(container, drum_offset, n_steps=360, intensity=1.0, n_radial=18, n_angular=72)`
Computes the `(xs, ys, weights)` locus for a uniformly distributed volume
source.  `container` is any `WasteContainer` instance (`CylindricalDrum` or
`CuboidContainer`).  `n_radial` and `n_angular` map to the container's two
sampling dimensions.

#### `compute_density_map(xs, ys, weights, grid_size=200, extent=None)`
Bins a source-trace locus into a 2-D weighted intensity histogram.

#### `plot_cases(..., container=None)`
Generates canonical comparison plots (four point-source + two volume-source),
and can optionally append a random multi-source case with
`plot_cases(random_n_sources=<n>, random_seed=<seed>)`.  Pass
`container=CuboidContainer(w, l)` (or any `WasteContainer`) to use a
non-cylindrical geometry; otherwise a `CylindricalDrum(drum_radius)` is
created automatically.

## Requirements

- Python ≥ 3.9
- NumPy
- Matplotlib
- pytest (for tests)
- flake8 (for linting)

## Development checks

Run linting:

```bash
flake8 .
```

Run tests:

```bash
pytest
```
