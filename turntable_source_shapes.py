"""turntable_source_shapes.py

Simulation of source shapes for gamma-spectrometry containers on a turntable.

As the turntable revolves, different source configurations trace different
paths when viewed from above (top-down view).  The module:

  * provides 2-D rotation-matrix helpers,
  * computes the locus traced by a source during one full turn,
  * bins that locus into a weighted density/intensity map, and
  * plots four canonical cases with matplotlib.

Cases (all viewed from directly above):
  1. Point source on the drum axis, drum centred on the turntable axis
     → source appears stationary (single point).
  2. Point source off the drum axis, drum centred on the turntable axis
     → source traces a circular ring.
  3. Point source on the drum axis, but drum NOT centred on the turntable
     → source traces a circular ring of radius equal to the drum offset.
  4. Point source off the drum axis AND drum offset from the turntable axis
     → source traces a circular ring of radius equal to the vector sum of
     the two offsets.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle


# ---------------------------------------------------------------------------
# Core rotation helpers
# ---------------------------------------------------------------------------

def rotation_matrix_2d(theta: float) -> np.ndarray:
    """Return the 2-D rotation matrix for angle *theta* (radians).

    .. math::

        R(\\theta) = \\begin{pmatrix}
            \\cos\\theta & -\\sin\\theta \\\\
            \\sin\\theta &  \\cos\\theta
        \\end{pmatrix}

    Parameters
    ----------
    theta:
        Rotation angle in radians (counter-clockwise positive).

    Returns
    -------
    numpy.ndarray of shape (2, 2)
    """
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s],
                     [s,  c]])


def rotate_point(point: np.ndarray | list[float], theta: float) -> np.ndarray:
    """Rotate a 2-D point by *theta* radians about the origin.

    Parameters
    ----------
    point:
        The (x, y) coordinates to rotate.
    theta:
        Rotation angle in radians.

    Returns
    -------
    numpy.ndarray of shape (2,)
        Rotated coordinates.
    """
    R = rotation_matrix_2d(theta)
    return R @ np.asarray(point, dtype=float)


# ---------------------------------------------------------------------------
# Turntable simulation
# ---------------------------------------------------------------------------

def source_trace(
    source_in_drum: np.ndarray | list[float],
    drum_offset: np.ndarray | list[float],
    n_steps: int = 360,
    intensity: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the locus of a source during one full turntable revolution.

    The geometry (all coordinates in the turntable/lab frame):

    * The turntable rotates about the *origin* of the lab frame.
    * The drum centre is offset from the turntable rotation axis by
      ``drum_offset``.
    * The source is located at ``source_in_drum`` *relative to the drum
      centre*.

    At rotation angle *θ* the source position in the lab frame is::

        pos(θ) = R(θ) · (drum_offset + source_in_drum)

    Parameters
    ----------
    source_in_drum:
        (x, y) position of the source relative to the drum centre (metres).
    drum_offset:
        (x, y) offset of the drum centre from the turntable rotation axis
        (metres).
    n_steps:
        Number of angular steps (angular resolution of the simulation).
    intensity:
        Relative emission intensity / probability of the source.

    Returns
    -------
    xs : numpy.ndarray of shape (n_steps,)
        x-coordinates of the source trace.
    ys : numpy.ndarray of shape (n_steps,)
        y-coordinates of the source trace.
    weights : numpy.ndarray of shape (n_steps,)
        Intensity weight at each position; the sum equals *intensity*.
    """
    thetas = np.linspace(0, 2 * np.pi, n_steps, endpoint=False)
    source_in_drum = np.asarray(source_in_drum, dtype=float)
    drum_offset = np.asarray(drum_offset, dtype=float)

    # Position of the source relative to the turntable axis (fixed in drum frame)
    relative_pos = drum_offset + source_in_drum

    # Apply rotation matrix to each angle
    c = np.cos(thetas)
    s = np.sin(thetas)
    xs = c * relative_pos[0] - s * relative_pos[1]
    ys = s * relative_pos[0] + c * relative_pos[1]

    # Each step contributes equally; total weight = intensity
    weights = np.full(n_steps, intensity / n_steps)

    return xs, ys, weights


def compute_density_map(
    xs: np.ndarray,
    ys: np.ndarray,
    weights: np.ndarray,
    grid_size: int = 200,
    extent: tuple[float, float, float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin a source trace into a 2-D weighted intensity / density map.

    Parameters
    ----------
    xs, ys:
        Source trace coordinates (metres).
    weights:
        Intensity weight per sample point.
    grid_size:
        Number of bins in each spatial dimension.
    extent:
        ``(xmin, xmax, ymin, ymax)`` spatial extent of the map.  If *None*,
        a margin of 10 % beyond the maximum radial extent is used.

    Returns
    -------
    H : numpy.ndarray of shape (grid_size, grid_size)
        Weighted 2-D intensity map (x is the first axis).
    xedges : numpy.ndarray of shape (grid_size + 1,)
        Bin edges along x.
    yedges : numpy.ndarray of shape (grid_size + 1,)
        Bin edges along y.
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    weights = np.asarray(weights, dtype=float)

    if extent is None:
        r_max = np.sqrt(np.max(xs ** 2 + ys ** 2))
        if r_max < 1e-12:
            r_max = 0.1
        r_max *= 1.1
        extent = (-r_max, r_max, -r_max, r_max)

    H, xedges, yedges = np.histogram2d(
        xs, ys,
        bins=grid_size,
        range=[[extent[0], extent[1]], [extent[2], extent[3]]],
        weights=weights,
    )
    return H, xedges, yedges


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_cases(
    drum_radius: float = 0.30,
    source_offset_in_drum: float = 0.15,
    drum_offset: float = 0.20,
    n_steps: int = 3600,
    grid_size: int = 400,
) -> plt.Figure:
    """Render four canonical source-on-turntable cases.

    Parameters
    ----------
    drum_radius:
        Radius of the waste drum in metres (used only for the drum outline).
    source_offset_in_drum:
        Radial offset of the source inside the drum for Cases 2 and 4 (metres).
    drum_offset:
        Offset of the drum centre from the turntable rotation axis for
        Cases 3 and 4 (metres).
    n_steps:
        Angular resolution – number of steps per full revolution.
    grid_size:
        Resolution of the 2-D density map (number of bins per axis).

    Returns
    -------
    matplotlib.figure.Figure
    """
    cases = [
        {
            "title": (
                "Case 1: Source on drum axis\n"
                "Drum centred on turntable axis\n"
                "→ stationary point"
            ),
            "source_in_drum": [0.0, 0.0],
            "drum_offset": [0.0, 0.0],
        },
        {
            "title": (
                f"Case 2: Source {source_offset_in_drum:.2f} m off drum axis\n"
                "Drum centred on turntable axis\n"
                "→ circular ring"
            ),
            "source_in_drum": [source_offset_in_drum, 0.0],
            "drum_offset": [0.0, 0.0],
        },
        {
            "title": (
                "Case 3: Source on drum axis\n"
                f"Drum offset {drum_offset:.2f} m from turntable axis\n"
                "→ circular ring"
            ),
            "source_in_drum": [0.0, 0.0],
            "drum_offset": [drum_offset, 0.0],
        },
        {
            "title": (
                f"Case 4: Source {source_offset_in_drum:.2f} m off drum axis\n"
                f"Drum offset {drum_offset:.2f} m from turntable axis\n"
                "→ circular ring (vector-sum radius)"
            ),
            "source_in_drum": [source_offset_in_drum, 0.0],
            "drum_offset": [drum_offset, 0.0],
        },
    ]

    # Shared spatial extent – large enough for all four cases
    r_max = max(
        drum_radius,
        source_offset_in_drum,
        drum_offset + drum_radius,
        drum_offset + source_offset_in_drum + drum_radius,
    ) * 1.5
    extent = (-r_max, r_max, -r_max, r_max)

    fig = plt.figure(figsize=(14, 12))
    gs = GridSpec(2, 2, figure=fig, wspace=0.40, hspace=0.55)

    for idx, case in enumerate(cases):
        row, col = divmod(idx, 2)
        xs, ys, weights = source_trace(
            case["source_in_drum"],
            case["drum_offset"],
            n_steps=n_steps,
        )

        ax = fig.add_subplot(gs[row, col])
        ax.set_facecolor("black")

        is_stationary = (np.std(xs) < 1e-9 and np.std(ys) < 1e-9)

        if is_stationary:
            # Source does not move – render as a bright dot
            ax.scatter(
                [xs[0]], [ys[0]],
                s=200, c="yellow", zorder=5, label="Source",
            )
        else:
            H, xedges, yedges = compute_density_map(
                xs, ys, weights,
                grid_size=grid_size,
                extent=extent,
            )
            im = ax.imshow(
                H.T,
                origin="lower",
                extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                cmap="hot",
                aspect="equal",
                interpolation="bilinear",
            )
            fig.colorbar(im, ax=ax, shrink=0.75, label="Relative intensity")

        # Drum outline – shown at initial position (θ=0) as a reference
        drum_circle = Circle(
            case["drum_offset"],
            drum_radius,
            fill=False,
            edgecolor="cyan",
            linewidth=1.5,
            linestyle="--",
            label=f"Drum @ θ=0 (r={drum_radius:.2f} m)",
            zorder=4,
        )
        ax.add_patch(drum_circle)

        # Turntable rotation axis (blue cross at origin)
        ax.plot(
            0, 0,
            "b+", markersize=14, markeredgewidth=2,
            label="Turntable axis",
            zorder=6,
        )

        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title(case["title"], fontsize=9)
        ax.set_aspect("equal")
        ax.legend(fontsize=7, loc="upper right")

    fig.suptitle(
        "Top-down view: source traces on a turntable\n"
        "(gamma-spectrometry waste drum)",
        fontsize=12,
        y=1.01,
    )
    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    output_file = "turntable_source_shapes.png"
    fig = plot_cases()
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_file}")
    plt.show()
