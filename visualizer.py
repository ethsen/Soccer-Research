import torch
import numpy as np

class SoccerVisualizer:
    def __init__(self, pitch_length=105, pitch_width=68, layout="x_rows"):
        """
        layout:
          - 'x_rows'  : tensor shape (105, 68) where row=i is x [0..104], col=j is y [0..67]
          - 'y_rows'  : tensor shape (68, 105) where row=i is y, col=j is x  (not used here but supported)
        """
        self.FL = int(pitch_length)
        self.FW = int(pitch_width)
        self.layout = layout

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

    def plot_possession_scatters(
        self,
        in_possession: torch.Tensor,
        out_possession: torch.Tensor,
        *,
        ax=None,
        pitch_kwargs=None,
        in_pos_kwargs=None,
        out_pos_kwargs=None,
        draw=True
    ):
        """
        Plot two one-hot matrices (shape 105x68) as team scatter points.
        - in_possession: torch.Tensor with 1s for the team in possession
        - out_possession: torch.Tensor with 1s for the out-of-possession team
        Returns (fig, ax).
        """
        # Extract x/y (meters) at cell centers
        xs_tm, ys_tm = self._indices_to_xy_centers(in_possession)
        xs_op, ys_op = self._indices_to_xy_centers(out_possession)

        # Lazy import so the class doesn't hard-depend on mplsoccer at import-time
        try:
            from mplsoccer import Pitch
        except ImportError as e:
            raise RuntimeError("mplsoccer is required for visualization. Install with `pip install mplsoccer`.") from e

        # Defaults
        pitch_kwargs = pitch_kwargs or {}
        in_pos_kwargs = in_pos_kwargs or {}
        out_pos_kwargs = out_pos_kwargs or {}

        # Default styles (your requested styling)
        in_pos_defaults = dict(s=20, marker="s", edgecolors="black", zorder=3, c="orange")
        out_pos_defaults = dict(c="dodgerblue", s=30, ec="k")

        # Merge user kwargs over defaults
        in_pos_plot = {**in_pos_defaults, **in_pos_kwargs}
        out_pos_plot = {**out_pos_defaults, **out_pos_kwargs}

        # Build/draw pitch
        pitch = Pitch(
            pitch_type="custom",
            pitch_width=self.FW,
            pitch_length=self.FL,
            axis=True,
            label=True,
            **pitch_kwargs
        )
        fig, ax = pitch.draw(ax=ax)

        # Scatter the teams
        if len(xs_tm):
            pitch.scatter(xs_tm, ys_tm, ax=ax, **in_pos_plot)
        if len(xs_op):
            pitch.scatter(xs_op, ys_op, ax=ax, **out_pos_plot)

        # Axis safety (should already be set by Pitch)
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
    
    def plot_heatmap(
            self,
            mat, *,
            ax=None,
            cmap="hot",
            edgecolors="none",
            pitch_kwargs=None,
            heatmap_kwargs=None,
            draw=True
        ):
            """
            Plot a (105, 68) map as an mplsoccer heatmap.
            - mat: torch.Tensor or np.ndarray shaped (105, 68) in this visualizer's layout.
            - layout='x_rows' means row=i is x in [0..104], col=j is y in [0..67].
            - Set use_imshow=True to use pitch.imshow instead of pitch.heatmap.
            Returns: (fig, ax, artist)
            """
            try:
                from mplsoccer import Pitch
            except ImportError as e:
                raise RuntimeError("mplsoccer is required for visualization. Install with `pip install mplsoccer`.") from e

            pitch_kwargs = pitch_kwargs or {}
            heatmap_kwargs = heatmap_kwargs or {}

            A = self._to_numpy(mat)
            if A.shape != (self.FL, self.FW):
                raise ValueError(f"Expected map of shape {(self.FL, self.FW)}; got {A.shape}")

            # mplsoccer expects rows=y, cols=x; transpose if we're in x_rows layout
            if self.layout == "x_rows":
                Z = A.T  # (68, 105) rows=y, cols=x
            elif self.layout == "y_rows":
                Z = A    # already rows=y, cols=x
            else:
                raise ValueError("layout must be 'x_rows' or 'y_rows'")

            # Build pitch
            pitch = Pitch(
                pitch_type="custom",
                pitch_width=self.FW,
                pitch_length=self.FL,
                axis=True,
                label=True,
                **pitch_kwargs
            )
            fig, ax = pitch.draw(ax=ax)

            
            # heatmap() needs a stats dict with bin edges
            x_edges = np.linspace(0, self.FL, self.FL + 1)  # 106 edges for 105 bins
            y_edges = np.linspace(0, self.FW, self.FW + 1)  # 69 edges for 68 bins
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
    
    def plot_state(
        self,
        in_possession: torch.Tensor,
        out_possession: torch.Tensor,*,
        heatmap: torch.Tensor | np.ndarray | None = None,
        cmap: str = "hot",
        edgecolors: str = "none",
        pitch_kwargs: dict | None = None,
        in_pos_kwargs: dict | None = None,
        out_pos_kwargs: dict | None = None,
        heatmap_kwargs: dict | None = None,
        draw: bool = True,
        add_colorbar: bool = True,
        ax=None,
    ):
        """
        Overlay (optional) heatmap + two team scatters on an mplsoccer pitch.

        in_possession / out_possession: (105, 68) one-hot tensors.
        heatmap: optional (105, 68) tensor/array to render under the scatters.
        use_imshow: if True, uses pitch.imshow; else pitch.heatmap with bin edges.
        Returns: (fig, ax, artists) where artists is a dict with keys:
                 {'heatmap', 'in_pos_scatter', 'out_pos_scatter'} (values may be None).
        """
        try:
            from mplsoccer import Pitch
        except ImportError as e:
            raise RuntimeError("mplsoccer is required. Install with `pip install mplsoccer`.") from e

        pitch_kwargs  = pitch_kwargs  or {}
        in_pos_kwargs = in_pos_kwargs or {}
        out_pos_kwargs= out_pos_kwargs or {}
        heatmap_kwargs= heatmap_kwargs or {}

        # Defaults for scatters
        in_pos_defaults  = dict(s=20, marker="s", edgecolors="black", zorder=3, c="orange")
        out_pos_defaults = dict(c="dodgerblue", s=30, ec="k", zorder=3)

        in_plot  = {**in_pos_defaults,  **in_pos_kwargs}
        out_plot = {**out_pos_defaults, **out_pos_kwargs}

        # Build pitch
        pitch = Pitch(
            pitch_type="custom",
            pitch_width=self.FW,
            pitch_length=self.FL,
            axis=True,
            label=True,
            **pitch_kwargs
        )
        fig, ax = pitch.draw(ax=ax)

        artists = {"heatmap": None, "in_pos_scatter": None, "out_pos_scatter": None}

        # ---- Optional heatmap (beneath everything) ----
        if heatmap is not None:
            A = self._to_numpy(heatmap)
            if A.shape != (self.FL, self.FW):
                raise ValueError(f"heatmap must be shape {(self.FL, self.FW)}, got {A.shape}")

            # mplsoccer expects rows=y, cols=x
            Z = A.T if self.layout == "x_rows" else A
            import numpy as np
            x_edges = np.linspace(0, self.FL, self.FL + 1)
            y_edges = np.linspace(0, self.FW, self.FW + 1)
            stats = {"statistic": Z, "x_grid": x_edges, "y_grid": y_edges}
            hm = pitch.heatmap(stats, ax=ax, cmap=cmap, edgecolors=edgecolors, zorder=1, **heatmap_kwargs)

            artists["heatmap"] = hm
            if add_colorbar:
                try:
                    import matplotlib.pyplot as plt
                    fig.colorbar(hm, ax=ax)
                except Exception:
                    pass

        # ---- Scatters on top ----
        xs_tm, ys_tm = self._indices_to_xy_centers(in_possession)
        xs_op, ys_op = self._indices_to_xy_centers(out_possession)

        if len(xs_tm):
            artists["in_pos_scatter"]  = pitch.scatter(xs_tm, ys_tm, ax=ax, **in_plot)
        if len(xs_op):
            artists["out_pos_scatter"] = pitch.scatter(xs_op, ys_op, ax=ax, **out_plot)

        # Safety
        ax.set_xlim(0, self.FL)
        ax.set_ylim(0, self.FW)
        ax.set_aspect("equal")

        return fig, ax, artists