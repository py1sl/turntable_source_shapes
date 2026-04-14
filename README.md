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
| Source *off* drum axis, drum centred on turntable | Circular ring |
| Source on drum axis, drum *offset* from turntable axis | Circular ring |

The module uses a 2-D rotation matrix to compute the locus and bins the
result into a weighted intensity map to account for emission probability.

## Usage

```python
from turntable_source_shapes import rotation_matrix_2d, source_trace, plot_cases
import matplotlib.pyplot as plt

# Compute the locus for a source 15 cm off the drum axis
xs, ys, weights = source_trace(
    source_in_drum=[0.15, 0.0],   # source position relative to drum centre (m)
    drum_offset=[0.0, 0.0],       # drum centre relative to turntable axis (m)
    n_steps=3600,
    intensity=1.0,
)

# Or generate the three canonical comparison plots directly
fig = plot_cases(
    drum_radius=0.30,            # drum radius in metres
    source_offset_in_drum=0.15,  # source radial offset in drum (Case 2)
    drum_offset=0.20,            # drum offset from turntable axis (Case 3)
)
plt.savefig("my_plot.png", dpi=150, bbox_inches="tight")
plt.show()
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

### `compute_density_map(xs, ys, weights, grid_size=200, extent=None)`
Bins a source-trace locus into a 2-D weighted intensity histogram.

### `plot_cases(...)`
Generates a side-by-side matplotlib figure of the three initial cases.

## Requirements

- Python ≥ 3.9
- NumPy
- Matplotlib
