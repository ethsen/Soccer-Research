import torch
import numpy as np
from matplotlib.colors import LinearSegmentedColormap



class SoccerVisualizer:
    def __init__(self, pitch_length=105, pitch_width=68, layout="x_rows"):
        """
        layout:
          - 'x_rows': tensor shape (105, 68) where row=i is x [0..104], col=j is y [0..67]
          - 'y_rows': tensor shape (68, 105) where row=i is y, col=j is x
        """
        self.FL = int(pitch_length)
        self.FW = int(pitch_width)
        self.layout = layout

    # ---------------------------------------------------------------------
    # Basic helpers
    # ---------------------------------------------------------------------
    @staticmethod
    def _to_numpy(arr):
        if isinstance(arr, torch.Tensor):
            return arr.detach().cpu().numpy()
        return np.asarray(arr)

    def _indices_to_xy_centers(self, mat):
        """
        Convert one-hot grid to x,y coordinates at cell centers.
        Expects mat to match self.layout. Returns xs, ys as 1D numpy arrays.
        """
        if not isinstance(mat, torch.Tensor):
            mat = torch.as_tensor(mat)

        if self.layout == "x_rows":
            # mat: (H=FL, W=FW) with row=x, col=y
            xs_idx, ys_idx = torch.nonzero(mat > 0, as_tuple=True)
            xs = xs_idx.float().numpy() + 0.5  # center of cell
            ys = ys_idx.float().numpy() + 0.5
            return xs, ys
        elif self.layout == "y_rows":
            # mat: (H=FW, W=FL) with row=y, col=x
            ys_idx, xs_idx = torch.nonzero(mat > 0, as_tuple=True)
            xs = xs_idx.float().numpy() + 0.5
            ys = ys_idx.float().numpy() + 0.5
            return xs, ys
        else:
            raise ValueError("layout must be 'x_rows' or 'y_rows'")

    def _build_pitch(self, *, ax=None, plain=False, pitch_kwargs=None):
        """
        Small utility to create an mplsoccer Pitch with consistent settings.
        plain=True -> white pitch (for Voronoi)
        plain=False -> grass pitch

        If ax is provided, we draw onto that axes and recover fig from ax.figure.
        If ax is None, we let mplsoccer create a new fig, ax.
        """
        try:
            from mplsoccer import Pitch
        except ImportError as e:
            raise RuntimeError("mplsoccer is required for visualization. Install with `pip install mplsoccer`.") from e

        pitch_kwargs = pitch_kwargs or {}

        if plain:
            pitch = Pitch(
                pitch_type="custom",
                pitch_width=self.FW,
                pitch_length=self.FL,
                pitch_color="#ffffff",
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
                line_color="white",
                stripe_color="#c2d59d",
                stripe=True,
                axis=False,
                label=False,
                **pitch_kwargs,
            )

        # Handle both "new fig/ax" and "draw on existing ax"
        if ax is None:
            # mplsoccer returns (fig, ax) here
            fig, ax = pitch.draw()
        else:
            # Draw on existing axes; draw() may return None or ax
            out = pitch.draw(ax=ax)
            if out is None:
                fig = ax.figure
            elif isinstance(out, tuple) and len(out) == 2:
                fig, ax = out
            else:
                # Sometimes draw(ax=...) returns just an Axes
                fig = ax.figure

        if not plain:
            fig.set_facecolor("#aabb97")
            ax.set_facecolor("#aabb97")

        return pitch, fig, ax


    def _prepare_grid_for_mpl(self, mat):
        """
        Take a (FL, FW) map in current layout and return Z (rows=y, cols=x)
        for mplsoccer heatmap / quiver.
        """
        A = self._to_numpy(mat)
        if A.shape != (self.FL, self.FW):
            raise ValueError(f"Expected map of shape {(self.FL, self.FW)}; got {A.shape}")

        if self.layout == "x_rows":
            Z = A.T  # (FW, FL) rows=y, cols=x
        elif self.layout == "y_rows":
            Z = A
        else:
            raise ValueError("layout must be 'x_rows' or 'y_rows'")
        return Z

    # ---------------------------------------------------------------------
    # Scatter plotting (teams only)
    # ---------------------------------------------------------------------
    def plot_possession_scatters(
        self,
        in_possession: torch.Tensor,
        out_possession: torch.Tensor,
        *,
        ax=None,
        pitch_kwargs=None,
        in_pos_kwargs=None,
        out_pos_kwargs=None,
        draw=True,
    ):
        """
        Plot two one-hot matrices (shape 105x68) as team scatter points.
        - in_possession: torch.Tensor with 1s for the team in possession
        - out_possession: torch.Tensor with 1s for the out-of-possession team
        Returns (fig, ax).
        """
        xs_tm, ys_tm = self._indices_to_xy_centers(in_possession)
        xs_op, ys_op = self._indices_to_xy_centers(out_possession)

        pitch, fig, ax = self._build_pitch(ax=ax, plain=False, pitch_kwargs=pitch_kwargs)

        in_pos_kwargs = in_pos_kwargs or {}
        out_pos_kwargs = out_pos_kwargs or {}

        in_pos_defaults = dict(s=20, marker="s", edgecolors="black", zorder=3, c="orange")
        out_pos_defaults = dict(c="dodgerblue", s=30, ec="k")

        in_plot = {**in_pos_defaults, **in_pos_kwargs}
        out_plot = {**out_pos_defaults, **out_pos_kwargs}

        if len(xs_tm):
            pitch.scatter(xs_tm, ys_tm, ax=ax, **in_plot)
        if len(xs_op):
            pitch.scatter(xs_op, ys_op, ax=ax, **out_plot)

        ax.set_xlim(0, self.FL)
        ax.set_ylim(0, self.FW)
        ax.set_aspect("equal")

        if draw:
            try:
                import matplotlib.pyplot as plt
                plt.tight_layout()
            except Exception:
                pass

        return fig, ax

    # ---------------------------------------------------------------------
    # Heatmap-only plotting
    # ---------------------------------------------------------------------
    def plot_heatmap(
        self,
        mat,
        *,
        ax=None,
        cmap=LinearSegmentedColormap.from_list("Pearl Earring - 10 colors",
                                                       ['#15242e', '#4393c4'], N=10),
        edgecolors="none",
        pitch_kwargs=None,
        heatmap_kwargs=None,
        draw=True,
    ):
        """
        Plot a (105, 68) map as an mplsoccer heatmap.
        - mat: torch.Tensor or np.ndarray shaped (105, 68) in this visualizer's layout.
        Returns: (fig, ax, artist)
        """
        try:
            from mplsoccer import Pitch  # noqa: F401
        except ImportError as e:
            raise RuntimeError("mplsoccer is required for visualization. Install with `pip install mplsoccer`.") from e

        pitch_kwargs = pitch_kwargs or {}
        heatmap_kwargs = heatmap_kwargs or {}

        Z = self._prepare_grid_for_mpl(mat)  # (FW, FL) rows=y, cols=x

        pitch = None
        # build pitch (reuse grass style for heatmaps)
        pitch, fig, ax = self._build_pitch(ax=ax, plain=False, pitch_kwargs=pitch_kwargs)

        x_edges = np.linspace(0, self.FL, self.FL + 1)  # 106 edges
        y_edges = np.linspace(0, self.FW, self.FW + 1)  # 69 edges
        stats = {"statistic": Z, "x_grid": x_edges, "y_grid": y_edges}
        artist = pitch.heatmap(
            stats, ax=ax, cmap=cmap, edgecolors=edgecolors, **heatmap_kwargs
        )

        ax.set_xlim(0, self.FL)
        ax.set_ylim(0, self.FW)
        ax.set_aspect("equal")

        if draw:
            try:
                import matplotlib.pyplot as plt
                fig.colorbar(artist, ax=ax)
                plt.tight_layout()
            except Exception:
                pass

        return fig, ax, artist

    # ---------------------------------------------------------------------
    # Full state plotting (players, optional heatmap, voronoi, visible area)
    # ---------------------------------------------------------------------
    def plot_state(
        self,
        in_possession: torch.Tensor,
        out_possession: torch.Tensor,
        *,
        heatmap: torch.Tensor | np.ndarray | None = None,
        cmap: str = LinearSegmentedColormap.from_list("Pearl Earring - 10 colors",
                                                       ['#15242e', '#4393c4'], N=10),
        edgecolors: str = "none",
        pitch_kwargs: dict | None = None,
        in_pos_kwargs: dict | None = None,
        out_pos_kwargs: dict | None = None,
        heatmap_kwargs: dict | None = None,
        add_colorbar: bool = True,
        ax=None,
        show_voronoi: bool = False,
        visible_area: torch.Tensor | np.ndarray | None = None,
    ):
        """
        Overlay (optional) heatmap + two team scatters on an mplsoccer pitch.

        in_possession / out_possession: (105, 68) one-hot tensors.
        heatmap: optional (105, 68) tensor/array to render under the scatters.
        If show_voronoi=True, draw a Voronoi diagram for the two teams.
        If visible_area is provided (Nx2 array of x,y coords), also draw it and
        clip Voronoi regions to that polygon (camera visible area).
        """
        pitch_kwargs = pitch_kwargs or {}
        in_pos_kwargs = in_pos_kwargs or {}
        out_pos_kwargs = out_pos_kwargs or {}
        heatmap_kwargs = heatmap_kwargs or {}

        # choose pitch style
        pitch, fig, ax = self._build_pitch(
            ax=ax, plain=show_voronoi, pitch_kwargs=pitch_kwargs
        )

        # Defaults for scatters
        in_pos_defaults = dict(
            s=20, marker="s", edgecolors="black", zorder=4,
            c="dodgerblue", label="Attacking Players"
        )
        out_pos_defaults = dict(
            c="orange", s=30, ec="k", zorder=4,
            label="Defending Players"
        )
        in_plot = {**in_pos_defaults, **in_pos_kwargs}
        out_plot = {**out_pos_defaults, **out_pos_kwargs}

        artists = {
            "heatmap": None,
            "in_pos_scatter": None,
            "out_pos_scatter": None,
            "voronoi_team1": None,
            "voronoi_team2": None,
            "visible_polygon": None,
        }

        # ---- Optional heatmap (beneath everything) ----
        if heatmap is not None:
            Z = self._prepare_grid_for_mpl(heatmap)
            x_edges = np.linspace(0, self.FL, self.FL + 1)
            y_edges = np.linspace(0, self.FW, self.FW + 1)
            stats = {"statistic": Z, "x_grid": x_edges, "y_grid": y_edges}
            hm = pitch.heatmap(
                stats, ax=ax, cmap=cmap, edgecolors=edgecolors, zorder=1, **heatmap_kwargs
            )
            artists["heatmap"] = hm
            if add_colorbar:
                try:
                    import matplotlib.pyplot as plt
                    fig.colorbar(hm, ax=ax)
                except Exception:
                    pass

        # ---- Player coordinates from masks ----
        xs_tm, ys_tm = self._indices_to_xy_centers(in_possession)
        xs_op, ys_op = self._indices_to_xy_centers(out_possession)

        # ---- Voronoi + visible area (under the players) ----
        visible_patch = None
        if show_voronoi and (len(xs_tm) + len(xs_op)) >= 2:
            x_all = np.concatenate([xs_tm, xs_op])
            y_all = np.concatenate([ys_tm, ys_op])
            team_mask = np.concatenate(
                [np.ones_like(xs_tm, dtype=bool), np.zeros_like(xs_op, dtype=bool)]
            )

            team1_polys, team2_polys = pitch.voronoi(x_all, y_all, team_mask)

            t1 = pitch.polygon(
                team1_polys, ax=ax, fc="dodgerblue", ec="white",
                lw=2, alpha=0.25, zorder=2
            )
            t2 = pitch.polygon(
                team2_polys, ax=ax, fc="orange", ec="white",
                lw=2, alpha=0.25, zorder=2
            )

            artists["voronoi_team1"] = t1
            artists["voronoi_team2"] = t2

            if visible_area is not None:
                vis = self._to_numpy(visible_area).reshape(-1, 2)
                visible_patch = pitch.polygon(
                    [vis], ax=ax, color="none", ec="k",
                    linestyle="--", lw=2, zorder=3
                )
                artists["visible_polygon"] = visible_patch
                vp = visible_patch[0]
                for poly in t1:
                    poly.set_clip_path(vp)
                for poly in t2:
                    poly.set_clip_path(vp)

        # ---- Scatters on top ----
        if len(xs_tm):
            artists["in_pos_scatter"] = pitch.scatter(xs_tm, ys_tm, ax=ax, **in_plot)
        if len(xs_op):
            artists["out_pos_scatter"] = pitch.scatter(xs_op, ys_op, ax=ax, **out_plot)

        # Safety
        ax.set_xlim(0, self.FL)
        ax.set_ylim(0, self.FW)
        ax.set_aspect("equal")

        return fig, ax, artists

    # ---------------------------------------------------------------------
    # NEW: Velocity visualization
    # ---------------------------------------------------------------------
    def plot_velocity_quiver(
        self,
        vx_map,
        vy_map,
        *,
        ax=None,
        pitch_kwargs=None,
        step: int = 8,
        scale: float | None = None,
        color: str = "red",
        alpha: float = 0.9,
        label: str | None = None,
    ):
        """
        Plot a velocity vector field (vx_map, vy_map) as arrows on a pitch.

        vx_map, vy_map: (FL, FW) in this visualizer's layout (same as heatmaps).
        - If the maps are constant (e.g. your glob velocities or ball velocity),
          you effectively get a uniform flow field.
        - step: downsampling factor to avoid drawing 7k arrows; step=8 is usually plenty.
        """
        pitch_kwargs = pitch_kwargs or {}

        # prepare data in mpl layout
        Zx = self._prepare_grid_for_mpl(vx_map)  # (FW, FL)
        Zy = self._prepare_grid_for_mpl(vy_map)  # (FW, FL)

        pitch, fig, ax = self._build_pitch(ax=ax, plain=False, pitch_kwargs=pitch_kwargs)

        # Grid of arrow positions: centers of cells
        xs = np.linspace(0.5, self.FL - 0.5, self.FL)  # length axis
        ys = np.linspace(0.5, self.FW - 0.5, self.FW)  # width axis
        X, Y = np.meshgrid(xs, ys)  # (FW, FL) each

        Xs = X[::step, ::step]
        Ys = Y[::step, ::step]
        U = Zx[::step, ::step]
        V = Zy[::step, ::step]

        import matplotlib.pyplot as plt  # noqa: F401

        q = ax.quiver(
            Xs, Ys, U, V,
            color=color,
            alpha=alpha,
            scale=scale,
            angles="xy",
            scale_units="xy",
            width=0.003,
            headwidth=3,
            headlength=4,
        )
        if label is not None:
            q.set_label(label)

        ax.set_xlim(0, self.FL)
        ax.set_ylim(0, self.FW)
        ax.set_aspect("equal")

        return fig, ax, q

    def plot_velocity_on_state(
        self,
        in_possession: torch.Tensor,
        out_possession: torch.Tensor,
        vx_map,
        vy_map,
        *,
        heatmap: torch.Tensor | np.ndarray | None = None,
        pitch_kwargs: dict | None = None,
        quiver_kwargs: dict | None = None,
        step: int = 8,
    ):
        """
        Convenience helper: plot a full state (players + optional heatmap) and overlay a
        velocity quiver field on top.

        You can use this for:
        - ball velocity maps (vx_ball_map, vy_ball_map)
        - att_glob_vx_map / att_glob_vy_map
        - def_glob_vx_map / def_glob_vy_map
        """
        pitch_kwargs = pitch_kwargs or {}
        quiver_kwargs = dict(quiver_kwargs or {})

        # If user didn't specify step in quiver_kwargs, use the function argument
        quiver_kwargs.setdefault("step", step)

        # First draw state (players + optional heatmap)
        fig, ax, _ = self.plot_state(
            in_possession=in_possession,
            out_possession=out_possession,
            heatmap=heatmap,
            pitch_kwargs=pitch_kwargs,
            add_colorbar=False,  # usually don't want colorbar for this combo
        )

        # Then overlay the vector field
        _, _, q = self.plot_velocity_quiver(
            vx_map=vx_map,
            vy_map=vy_map,
            ax=ax,
            pitch_kwargs=pitch_kwargs,
            **quiver_kwargs,
        )

        return fig, ax, q
