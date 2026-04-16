# turntable_source_shapes

Derivation of source shapes for containers on a turntable, designed for
gamma-spectrometry of waste drums.

## Background

When a waste drum is placed on a rotating turntable and measured with a
gamma-ray detector, point sources inside the drum trace different paths in
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

## Usage

```python
from turntable_source_shapes import (
    rotation_matrix_2d,
    source_trace,
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
    drum_radius=0.30,            # drum radius in metres
    source_offset_in_drum=0.15,  # source radial offset in drum (Case 2)
    drum_offset=0.20,            # drum offset from turntable axis (Case 3)
    n_volume_radial=18,          # volume-source radial sampling bins
    n_volume_angular=72,         # volume-source angular sampling bins
)
plt.savefig("my_plot.png", dpi=150, bbox_inches="tight")
plt.show()

# Optional: random n-source distribution throughout the drum
source_positions, source_intensities = random_sources_in_drum(
    n_sources=25,
    drum_radius=0.30,
    total_intensity=1.0,
    seed=42,
)
xs, ys, weights = multi_source_trace(
    sources_in_drum=source_positions,
    drum_offset=[0.0, 0.0],
    intensities=source_intensities,
    n_steps=3600,
)

# Or append an automatic random-source case to the comparison plot
fig = plot_cases(random_n_sources=25, random_seed=42)
```

Run the script directly to produce `turntable_source_shapes.png`:

```bash
python turntable_source_shapes.py
```

## API

### `rotation_matrix_2d(theta)`
Returns the 2×2 rotation matrix for angle `theta` (radians).

### `rotate_point(point, theta)`
Rotates a 2-D point `(x, y)` by `theta` radians about the origin.

### `source_trace(source_in_drum, drum_offset, n_steps=360, intensity=1.0)`
Computes the `(xs, ys, weights)` locus of a source through one full
revolution of the turntable.

### `random_sources_in_drum(n_sources, drum_radius, total_intensity=1.0, seed=None)`
Generates `n_sources` random point locations uniformly within the drum
cross-section and assigns different random intensities that sum to
`total_intensity`.

### `multi_source_trace(sources_in_drum, drum_offset, intensities, n_steps=360)`
Computes the combined `(xs, ys, weights)` locus for multiple point sources
in one drum.

### `volume_source_trace(drum_radius, drum_offset, n_steps=360, intensity=1.0, n_radial=18, n_angular=72)`
Computes the `(xs, ys, weights)` locus for a uniformly distributed drum
volume source using equal-area sampling across the drum cross-section.

### `compute_density_map(xs, ys, weights, grid_size=200, extent=None)`
Bins a source-trace locus into a 2-D weighted intensity histogram.

### `plot_cases(...)`
Generates canonical comparison plots and can optionally append a random
multi-source case with `plot_cases(random_n_sources=<n>, random_seed=<seed>)`.
Generates canonical comparison plots (four point-source + two volume-source),
and can optionally append a random multi-source case with
`plot_cases(random_n_sources=<n>, random_seed=<seed>)`.

## Requirements

- Python ≥ 3.9
- NumPy
- Matplotlib
