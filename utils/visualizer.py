import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap


DEFAULT_CMAP = LinearSegmentedColormap.from_list(
    "Pearl Earring - 10 colors", ["#ffffff", "#4393c4"], N=10
)


class SoccerVisualizer:
    """
    Cleaner, composable visualizer.

    Conventions:
    - Field coordinates: x in [0, FL], y in [0, FW]
    - Internal "map" tensors are expected shape (FL, FW) in `layout="x_rows"`:
        row index = x cell, col index = y cell
      `layout="y_rows"` is the transpose convention.
    """

    def __init__(self, pitch_length=105, pitch_width=68, layout="x_rows"):
        self.FL = int(pitch_length)
        self.FW = int(pitch_width)
        if layout not in ("x_rows", "y_rows"):
            raise ValueError("layout must be 'x_rows' or 'y_rows'")
        self.layout = layout

    # ---------------------------------------------------------------------
    # Core small utilities
    # ---------------------------------------------------------------------
    @staticmethod
    def _to_numpy(arr):
        if isinstance(arr, torch.Tensor):
            return arr.detach().cpu().numpy()
        return np.asarray(arr)

    def _require_shape(self, A, shape):
        if tuple(A.shape) != tuple(shape):
            raise ValueError(f"Expected shape {shape}, got {A.shape}")

    def _prepare_grid_for_mpl(self, mat):
        """
        Convert a (FL, FW) map in the user's layout into Z with shape (FW, FL)
        where rows=y and cols=x (mplsoccer's natural grid orientation).
        """
        A = self._to_numpy(mat)
        self._require_shape(A, (self.FL, self.FW))
        return A.T if self.layout == "x_rows" else A

    def _indices_to_xy_centers(self, onehot):
        """
        Convert a one-hot occupancy map to (x, y) coordinates of cell centers.
        Accepts torch/numpy. Expects shape (FL, FW) in the user's layout.
        """
        if not isinstance(onehot, torch.Tensor):
            onehot = torch.as_tensor(onehot)

        if self.layout == "x_rows":
            xs_idx, ys_idx = torch.nonzero(onehot > 0, as_tuple=True)
        else:  # "y_rows"
            ys_idx, xs_idx = torch.nonzero(onehot > 0, as_tuple=True)

        xs = xs_idx.float().cpu().numpy() + 0.5
        ys = ys_idx.float().cpu().numpy() + 0.5
        return xs, ys

    # ---------------------------------------------------------------------
    # Pitch creation
    # ---------------------------------------------------------------------
    def _build_pitch(self, *, ax=None, plain=False, pitch_kwargs=None):
        """
        Create/reuse an mplsoccer.Pitch and draw it on ax if provided.
        Returns (pitch, fig, ax).
        """
        try:
            from mplsoccer import Pitch
        except ImportError as e:
            raise RuntimeError(
                "mplsoccer is required. Install with `pip install mplsoccer`."
            ) from e

        pitch_kwargs = pitch_kwargs or {}

        if plain:
            pitch = Pitch(
                pitch_type="custom",
                pitch_width=self.FW,
                pitch_length=self.FL,
                pitch_color="#ffffff7f",
                stripe_color="#ffffff",
                stripe=True,
                axis=False,
                label=False,
                **pitch_kwargs,
            )
        else:
            pitch = Pitch(
                pitch_type="custom",
                pitch_width=self.FW,
                pitch_length=self.FL,
                pitch_color="#aabb97",
                line_color="#6e6e6e",
                stripe_color="#c2d59d",
                stripe=True,
                axis=False,
                label=False,
                **pitch_kwargs,
            )

        if ax is None:
            fig, ax = pitch.draw()
        else:
            out = pitch.draw(ax=ax)
            fig = ax.figure if out is None else (out[0] if isinstance(out, tuple) else ax.figure)

        # Keep pitch lines above most overlays by default
        for line in ax.lines:
            line.set_zorder(10)

        ax.set_xlim(0, self.FL)
        ax.set_ylim(0, self.FW)
        ax.set_aspect("equal")
        ax.margins(0)

        return pitch, fig, ax

    # ---------------------------------------------------------------------
    # Overlay primitives (small + composable)
    # ---------------------------------------------------------------------
    def add_players(
        self,
        pitch,
        ax,
        in_possession,
        out_possession,
        *,
        in_pos_kwargs=None,
        out_pos_kwargs=None,
    ):
        in_pos_kwargs = dict(in_pos_kwargs or {})
        out_pos_kwargs = dict(out_pos_kwargs or {})

        in_defaults = dict(
            s=20, marker="s", edgecolors="black", zorder=20, c="dodgerblue",
            label="Attacking Players"
        )
        out_defaults = dict(
            s=30, ec="k", zorder=20, c="orange",
            label="Defending Players"
        )

        xs_tm, ys_tm = self._indices_to_xy_centers(in_possession)
        xs_op, ys_op = self._indices_to_xy_centers(out_possession)

        artists = {}
        if len(xs_tm):
            artists["in_pos_scatter"] = pitch.scatter(xs_tm, ys_tm, ax=ax, **{**in_defaults, **in_pos_kwargs})
        if len(xs_op):
            artists["out_pos_scatter"] = pitch.scatter(xs_op, ys_op, ax=ax, **{**out_defaults, **out_pos_kwargs})

        return artists

    def add_heatmap(
        self,
        pitch,
        fig,
        ax,
        heatmap,
        *,
        cmap=DEFAULT_CMAP,
        edgecolors="none",
        heatmap_kwargs=None,
        add_colorbar=True,
        colorbar_kwargs=None,
    ):
        heatmap_kwargs = dict(heatmap_kwargs or {})
        colorbar_kwargs = dict(colorbar_kwargs or {})

        Z = self._prepare_grid_for_mpl(heatmap)  # (FW, FL)
        x_edges = np.linspace(0, self.FL, self.FL + 1)
        y_edges = np.linspace(0, self.FW, self.FW + 1)
        stats = {"statistic": Z, "x_grid": x_edges, "y_grid": y_edges}

        hm = pitch.heatmap(
            stats, ax=ax, cmap=cmap, edgecolors=edgecolors, zorder=.75, **heatmap_kwargs
        )

        cb = None
        if add_colorbar:
            try:
                from mpl_toolkits.axes_grid1 import make_axes_locatable

                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="3%", pad=0.05)
                cb = fig.colorbar(hm, cax=cax, **colorbar_kwargs)
            except Exception:
                cb = None

        return {"heatmap": hm, "colorbar": cb}

    def add_visible_area(
        self,
        pitch,
        ax,
        visible_area_xy,
        *,
        poly_kwargs=None,
    ):
        poly_kwargs = dict(poly_kwargs or {})
        vis = self._to_numpy(visible_area_xy).reshape(-1, 2)

        defaults = dict(
            color=("#adbde7", 0.5),
            zorder=8,
            ec="#524218",
            lw=1.0,
            label="Visible Area",
        )
        polys = pitch.polygon([vis], ax=ax, **{**defaults, **poly_kwargs})
        # mplsoccer returns list of patches; usually length 1
        patch = polys[0] if polys else None
        return {"visible_polygon": patch}

    def add_voronoi(
        self,
        pitch,
        ax,
        in_possession,
        out_possession,
        *,
        visible_clip_patch=None,
        team1_kwargs=None,
        team2_kwargs=None,
    ):
        team1_kwargs = dict(team1_kwargs or {})
        team2_kwargs = dict(team2_kwargs or {})

        xs_tm, ys_tm = self._indices_to_xy_centers(in_possession)
        xs_op, ys_op = self._indices_to_xy_centers(out_possession)

        if (len(xs_tm) + len(xs_op)) < 2:
            return {"voronoi_team1": None, "voronoi_team2": None}

        x_all = np.concatenate([xs_tm, xs_op])
        y_all = np.concatenate([ys_tm, ys_op])
        team_mask = np.concatenate([np.ones_like(xs_tm, dtype=bool), np.zeros_like(xs_op, dtype=bool)])

        team1_polys, team2_polys = pitch.voronoi(x_all, y_all, team_mask)

        t1_defaults = dict(fc="dodgerblue", ec="white", lw=2, alpha=0.25, zorder=7)
        t2_defaults = dict(fc="orange", ec="white", lw=2, alpha=0.25, zorder=7)

        t1 = pitch.polygon(team1_polys, ax=ax, **{**t1_defaults, **team1_kwargs})
        t2 = pitch.polygon(team2_polys, ax=ax, **{**t2_defaults, **team2_kwargs})

        if visible_clip_patch is not None:
            for poly in (t1 or []):
                poly.set_clip_path(visible_clip_patch)
            for poly in (t2 or []):
                poly.set_clip_path(visible_clip_patch)

        return {"voronoi_team1": t1, "voronoi_team2": t2}

    def add_quiver(
        self,
        ax,
        vx_map,
        vy_map,
        *,
        step=8,
        scale=None,
        color="red",
        alpha=0.9,
        quiver_kwargs=None,
    ):
        quiver_kwargs = dict(quiver_kwargs or {})

        Zx = self._prepare_grid_for_mpl(vx_map)  # (FW, FL)
        Zy = self._prepare_grid_for_mpl(vy_map)  # (FW, FL)

        xs = np.linspace(0.5, self.FL - 0.5, self.FL)
        ys = np.linspace(0.5, self.FW - 0.5, self.FW)
        X, Y = np.meshgrid(xs, ys)  # (FW, FL)

        Xs = X[::step, ::step]
        Ys = Y[::step, ::step]
        U = Zx[::step, ::step]
        V = Zy[::step, ::step]

        q = ax.quiver(
            Xs, Ys, U, V,
            color=color,
            alpha=alpha,
            scale=scale,
            angles="xy",
            scale_units="xy",
            width=0.005,
            headwidth=3,
            headlength=4,
            zorder=15,
            **quiver_kwargs,
        )
        return {"quiver": q}

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def plot_heatmap(
        self,
        heatmap,
        *,
        ax=None,
        pitch_kwargs=None,
        plain=False,
        cmap=DEFAULT_CMAP,
        edgecolors="none",
        heatmap_kwargs=None,
        add_colorbar=True,
        colorbar_kwargs=None,
    ):
        """
        Heatmap-only plot (no players).
        """
        pitch, fig, ax = self._build_pitch(ax=ax, plain=plain, pitch_kwargs=pitch_kwargs)
        artists = self.add_heatmap(
            pitch, fig, ax, heatmap,
            cmap=cmap,
            edgecolors=edgecolors,
            heatmap_kwargs=heatmap_kwargs,
            add_colorbar=add_colorbar,
            colorbar_kwargs=colorbar_kwargs,
        )
        return fig, ax, artists

    def plot_state(
        self,
        in_possession,
        out_possession,
        *,
        ax=None,
        pitch_kwargs=None,
        plain=False,
        # overlays
        heatmap=None,
        cmap=DEFAULT_CMAP,
        edgecolors="none",
        heatmap_kwargs=None,
        add_colorbar=True,
        colorbar_kwargs=None,
        visible_area=None,
        visible_poly_kwargs=None,
        show_voronoi=False,
        voronoi_team1_kwargs=None,
        voronoi_team2_kwargs=None,
        vx_map=None,
        vy_map=None,
        quiver_step=8,
        quiver_scale=None,
        quiver_color="red",
        quiver_alpha=0.9,
        quiver_kwargs=None,
        in_pos_kwargs=None,
        out_pos_kwargs=None,
    ):
        """
        One-stop: players + optional overlays, all on a single pitch axis.

        Typical uses:
        - plot_state(..., heatmap=succ_map)
        - plot_state(..., show_voronoi=True, visible_area=poly)
        - plot_state(..., heatmap=..., vx_map=..., vy_map=..., quiver_step=8)
        """
        pitch, fig, ax = self._build_pitch(ax=ax, plain=plain, pitch_kwargs=pitch_kwargs)

        artists = {}

        # Heatmap under everything
        if heatmap is not None:
            artists.update(self.add_heatmap(
                pitch, fig, ax, heatmap,
                cmap=cmap,
                edgecolors=edgecolors,
                heatmap_kwargs=heatmap_kwargs,
                add_colorbar=add_colorbar,
                colorbar_kwargs=colorbar_kwargs,
            ))

        # Visible area (also used as voronoi clip)
        clip_patch = None
        if visible_area is not None:
            vis_art = self.add_visible_area(pitch, ax, visible_area, poly_kwargs=visible_poly_kwargs)
            artists.update(vis_art)
            clip_patch = vis_art.get("visible_polygon")

        # Voronoi under players, above heatmap
        if show_voronoi:
            artists.update(self.add_voronoi(
                pitch, ax, in_possession, out_possession,
                visible_clip_patch=clip_patch,
                team1_kwargs=voronoi_team1_kwargs,
                team2_kwargs=voronoi_team2_kwargs,
            ))

        # Quiver (usually above heatmap, below players or above depending on zorder)
        if (vx_map is not None) and (vy_map is not None):
            artists.update(self.add_quiver(
                ax, vx_map, vy_map,
                step=quiver_step,
                scale=quiver_scale,
                color=quiver_color,
                alpha=quiver_alpha,
                quiver_kwargs=quiver_kwargs,
            ))

        # Players on top
        artists.update(self.add_players(
            pitch, ax, in_possession, out_possession,
            in_pos_kwargs=in_pos_kwargs,
            out_pos_kwargs=out_pos_kwargs,
        ))

        return fig, ax, artists
