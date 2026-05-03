import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import Slider

from core.data_io import read_lsv_data
from core.style import (
    BG, INK, SUBTLE, ACCENT, BORDER,
    LSV_CURVE_COLORS, LSV_DATA_COLOR,
    OVERPOT_COLORS, OVERPOT_LABELS,
    style_axes,
)
from lsv_decoupling.logic import iterative_tafel_fit, get_overpotential_at_current


class LsvUI(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)

        # --- Data holders ---
        self.filename = None
        self.V_data = None
        self.i_data = None
        self.E_rev = None
        self.fit_results = None
        self.eta_kin = self.Eta_ohm = self.Eta_RCL = self.Eta_res = None
        self.y1 = self.y2 = self.y3 = self.y4 = self.y5 = self.y6 = None
        self.x_log_scale = False

        # Interactive marker state on the decoupled plot
        self._target_marker = None
        self._target_line = None
        self._slider = None  # matplotlib Slider on column plot

        self.reference_options = {
            "RHE": 0.0,
            "Ag/AgCl (sat)": 0.197,
            "SCE (sat)": 0.242,
            "HgO (sat)": 0.098,
        }

        self._build_layout()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def _build_layout(self):
        self.configure(style="TFrame")

        # --- Left Panel: Controls ---
        left_panel = ttk.Frame(self)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(14, 8), pady=14)

        ttk.Label(
            left_panel, text="LSV Decoupling",
            font=("Segoe UI", 14, "bold"),
        ).pack(anchor="w", pady=(0, 2))
        ttk.Label(
            left_panel,
            text="Decompose polarization losses into kinetic, ohmic,\ncatalyst-layer and residual contributions.",
            style="Hint.TLabel", justify="left",
        ).pack(anchor="w", pady=(0, 10))

        # File Loading
        file_frame = ttk.LabelFrame(left_panel, text="Data")
        file_frame.pack(fill=tk.X, pady=(0, 8))
        ttk.Button(file_frame, text="Load LSV File", command=self.load_file,
                   style="Accent.TButton").pack(fill=tk.X)
        self.lbl_status = ttk.Label(file_frame, text="No file selected.",
                                    style="Hint.TLabel")
        self.lbl_status.pack(anchor="w", pady=(6, 0))

        # Ref & Electrolyte
        ref_frame = ttk.LabelFrame(left_panel, text="Electrolyte & Reference")
        ref_frame.pack(fill=tk.X, pady=8)
        ref_frame.columnconfigure(1, weight=1)

        ttk.Label(ref_frame, text="Reference").grid(row=0, column=0, sticky="w", padx=2, pady=3)
        self.ref_var = tk.StringVar(value="RHE")
        ttk.Combobox(
            ref_frame, textvariable=self.ref_var,
            values=list(self.reference_options.keys()),
            state="readonly", width=14,
        ).grid(row=0, column=1, sticky="ew", padx=2, pady=3)

        ttk.Label(ref_frame, text="pH").grid(row=1, column=0, sticky="w", padx=2, pady=3)
        self.pH_entry = ttk.Entry(ref_frame, width=14)
        self.pH_entry.insert(0, "0")
        self.pH_entry.grid(row=1, column=1, sticky="ew", padx=2, pady=3)

        ttk.Label(ref_frame, text="Temperature (K)").grid(row=2, column=0, sticky="w", padx=2, pady=3)
        self.temp_entry = ttk.Entry(ref_frame, width=14)
        self.temp_entry.insert(0, "378")
        self.temp_entry.grid(row=2, column=1, sticky="ew", padx=2, pady=3)

        # Resistances
        res_frame = ttk.LabelFrame(left_panel, text="Resistances")
        res_frame.pack(fill=tk.X, pady=8)
        res_frame.columnconfigure(1, weight=1)

        ttk.Label(res_frame, text="R_CL  (Ω·cm²)").grid(row=0, column=0, sticky="w", padx=2, pady=3)
        self.rcl_entry = ttk.Entry(res_frame, width=12)
        self.rcl_entry.insert(0, "0")
        self.rcl_entry.grid(row=0, column=1, sticky="ew", padx=2, pady=3)

        ttk.Label(res_frame, text="HFR   (Ω·cm²)").grid(row=1, column=0, sticky="w", padx=2, pady=3)
        self.hfr_entry = ttk.Entry(res_frame, width=12)
        self.hfr_entry.insert(0, "0.01")
        self.hfr_entry.grid(row=1, column=1, sticky="ew", padx=2, pady=3)

        # Tafel Window
        tafel_frame = ttk.LabelFrame(left_panel, text="Tafel Fit Range  (A/cm²)")
        tafel_frame.pack(fill=tk.X, pady=8)
        tafel_frame.columnconfigure((1, 3), weight=1)

        ttk.Label(tafel_frame, text="Lower").grid(row=0, column=0, sticky="w", padx=2, pady=3)
        self.tafel_lower_entry = ttk.Entry(tafel_frame, width=8)
        self.tafel_lower_entry.insert(0, "0.02")
        self.tafel_lower_entry.grid(row=0, column=1, sticky="ew", padx=2, pady=3)

        ttk.Label(tafel_frame, text="Upper").grid(row=0, column=2, sticky="w", padx=(10, 2), pady=3)
        self.tafel_upper_entry = ttk.Entry(tafel_frame, width=8)
        self.tafel_upper_entry.insert(0, "0.8")
        self.tafel_upper_entry.grid(row=0, column=3, sticky="ew", padx=2, pady=3)

        # Target current density for breakdown
        bar_frame = ttk.LabelFrame(left_panel, text="Breakdown Target")
        bar_frame.pack(fill=tk.X, pady=8)
        bar_frame.columnconfigure(1, weight=1)

        ttk.Label(bar_frame, text="i_target (A/cm²)").grid(row=0, column=0, sticky="w", padx=2, pady=3)
        self.target_i_entry = ttk.Entry(bar_frame, width=8)
        self.target_i_entry.insert(0, "1.0")
        self.target_i_entry.grid(row=0, column=1, sticky="ew", padx=2, pady=3)
        ttk.Button(bar_frame, text="Update", command=self.update_bar_plot).grid(
            row=0, column=2, padx=(6, 0), pady=3,
        )
        ttk.Label(
            bar_frame,
            text="Tip: click the LSV plot or drag the slider\nto explore other current densities.",
            style="Hint.TLabel", justify="left",
        ).grid(row=1, column=0, columnspan=3, sticky="w", padx=2, pady=(4, 0))

        # Actions
        ttk.Separator(left_panel).pack(fill=tk.X, pady=10)
        ttk.Button(left_panel, text="Fit & Plot", command=self.perform_fit,
                   style="Accent.TButton").pack(fill=tk.X, pady=2)
        ttk.Button(left_panel, text="Toggle Linear / Log X", command=self.toggle_x_scale).pack(
            fill=tk.X, pady=2,
        )
        ttk.Button(left_panel, text="Export Decoupled Data…", command=self.export_data).pack(
            fill=tk.X, pady=(10, 0),
        )

        # --- Right Panel: Notebook ---
        right_panel = ttk.Frame(self)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(8, 14), pady=14)

        self.notebook = ttk.Notebook(right_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # --- Tab 1: Decoupled Plot ---
        self.tab_plot = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_plot, text="Decoupled curves")

        self.fig = plt.Figure(figsize=(8, 6))
        self.fig.patch.set_facecolor(BG)
        self.ax = self.fig.add_subplot(111)
        style_axes(self.ax)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.tab_plot)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("button_press_event", self._on_curve_click)

        toolbar_frame = ttk.Frame(self.tab_plot)
        toolbar_frame.pack(fill=tk.X, side=tk.BOTTOM)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()

        # --- Tab 2: Overpotential breakdown (the redesigned column plot) ---
        self.tab_bar = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_bar, text="Overpotential breakdown")

        self.fig_bar = plt.Figure(figsize=(9, 5))
        self.fig_bar.patch.set_facecolor(BG)
        # Two areas: main bar on top, slider strip on the bottom
        self.ax_bar = self.fig_bar.add_axes([0.08, 0.32, 0.88, 0.55])
        self.ax_slider = self.fig_bar.add_axes([0.12, 0.10, 0.80, 0.05])
        self.canvas_bar = FigureCanvasTkAgg(self.fig_bar, master=self.tab_bar)
        self.canvas_bar.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar_frame_bar = ttk.Frame(self.tab_bar)
        toolbar_frame_bar.pack(fill=tk.X, side=tk.BOTTOM)
        self.toolbar_bar = NavigationToolbar2Tk(self.canvas_bar, toolbar_frame_bar)
        self.toolbar_bar.update()

        # Empty-state message until a fit is run.
        self._draw_breakdown_placeholder()

        # --- Tab 3: Diagnostics ---
        self.tab_diag = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_diag, text="Fit diagnostics")
        self.diag_text = tk.Text(
            self.tab_diag, font=("Consolas", 10), wrap=tk.WORD,
            background=BG, foreground=INK, relief="flat",
            highlightthickness=1, highlightbackground=BORDER,
        )
        self.diag_text.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

    # ------------------------------------------------------------------
    # File / diagnostics helpers
    # ------------------------------------------------------------------
    def load_file(self):
        filepath = filedialog.askopenfilename(filetypes=[("Data files", "*.xlsx *.txt *.csv")])
        if filepath:
            self.filename = filepath
            self.lbl_status.config(text=os.path.basename(filepath))

    def write_diag(self, text):
        self.diag_text.config(state=tk.NORMAL)
        self.diag_text.delete("1.0", tk.END)
        self.diag_text.insert(tk.END, text)
        self.diag_text.config(state=tk.DISABLED)

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def perform_fit(self):
        if not self.filename:
            messagebox.showerror("Error", "No file selected.")
            return

        try:
            T = float(self.temp_entry.get())
            R_CL = float(self.rcl_entry.get())
            HFR = float(self.hfr_entry.get())
            pH_val = float(self.pH_entry.get())
            i_lower = float(self.tafel_lower_entry.get())
            i_upper = float(self.tafel_upper_entry.get())
            if i_lower >= i_upper:
                messagebox.showerror("Input Error", "Lower limit must be < Upper limit.")
                return
        except ValueError:
            messagebox.showerror("Input Error", "Please enter numeric values for all parameters.")
            return

        data, err = read_lsv_data(self.filename)
        if err:
            messagebox.showerror("File Error", err)
            return

        V_raw, i_raw = data
        self.E_rev = 1.2291 - 0.0008456 * (T - 298.15)
        offset = self.reference_options[self.ref_var.get()]
        V_corr = V_raw if self.ref_var.get() == "RHE" else (V_raw + 0.0592 * pH_val + offset)

        self.V_data, self.i_data = V_corr, i_raw

        result, warn_err = iterative_tafel_fit(
            self.V_data, self.i_data, self.E_rev, HFR, R_CL, i_lower, i_upper,
        )

        if result is None:
            messagebox.showerror("Fitting Error", warn_err)
            return

        if warn_err:
            messagebox.showwarning("Fitting Warning", warn_err)

        fit_result, eta_ohm_all, eta_rcl_all, _ = result
        b_kin, i0 = fit_result["b_kin"], fit_result["i0"]
        self.fit_results = fit_result

        self.eta_kin = b_kin * np.log10(self.i_data / i0)
        self.Eta_ohm = eta_ohm_all
        self.Eta_RCL = eta_rcl_all
        self.Eta_res = np.maximum(
            (self.V_data - self.E_rev) - (self.eta_kin + self.Eta_ohm + self.Eta_RCL), 0,
        )

        self.y1 = np.full_like(self.i_data, self.E_rev)
        self.y2 = self.y1 + self.eta_kin
        self.y3 = self.y2 + self.Eta_ohm
        self.y4 = self.y3 + self.Eta_RCL
        self.y5 = self.y4 + self.Eta_res
        self.y6 = self.V_data

        d = fit_result["candidate_details"]
        diag = (
            f"Tafel fit (iterative)\n"
            f"  b_kin: {b_kin:.4f} V/dec\n"
            f"  i0:    {i0:.4e} A/cm²\n"
            f"  Intercept: {fit_result['intercept']:.4f} V\n"
            f"  R²:    {fit_result['r_squared']:.4f}\n"
            f"  N in [{i_lower:.4g}, {i_upper:.4g}] A/cm²: {fit_result['n_points']}\n\n"
            f"Best 60 mV window\n"
            f"  {d['window'][0]:.4f}–{d['window'][1]:.4f} V  (center {d['center']:.4f} V)\n"
            f"  20 mV: {d['subwindows']['20mV'][0]:.4f}–{d['subwindows']['20mV'][1]:.4f} (n={d['counts']['20mV']})\n"
            f"  40 mV: {d['subwindows']['40mV'][0]:.4f}–{d['subwindows']['40mV'][1]:.4f} (n={d['counts']['40mV']})\n"
            f"  60 mV: {d['subwindows']['60mV'][0]:.4f}–{d['subwindows']['60mV'][1]:.4f} (n={d['counts']['60mV']})\n"
            f"  Rel. SE(b):  {d['rel_err_b'] * 100:.2f}%   Rel. SE(i0): {d['rel_err_i0'] * 100:.2f}%\n"
            f"  Combined metric: {d['candidate_metric'] * 100:.2f}%\n"
        )
        self.write_diag(diag)

        self._plot_curves()
        self._build_slider()
        self.update_bar_plot(auto_focus=False)
        self.notebook.select(self.tab_plot)

    # ------------------------------------------------------------------
    # Decoupled plot (curves)
    # ------------------------------------------------------------------
    def _plot_curves(self):
        ax = self.ax
        ax.clear()
        style_axes(ax)
        ax.set_xscale("log" if self.x_log_scale else "linear")

        c0, c1, c2, c3, c4 = LSV_CURVE_COLORS

        # Filled bands between cumulative curves convey *which* overpotential
        # contributes how much at each current density.
        ax.fill_between(self.i_data, self.y1, self.y2, color=c1, alpha=0.18,
                        linewidth=0, label="_nolegend_")
        ax.fill_between(self.i_data, self.y2, self.y3, color=c2, alpha=0.18,
                        linewidth=0, label="_nolegend_")
        ax.fill_between(self.i_data, self.y3, self.y4, color=c3, alpha=0.18,
                        linewidth=0, label="_nolegend_")
        ax.fill_between(self.i_data, self.y4, self.y5, color=c4, alpha=0.18,
                        linewidth=0, label="_nolegend_")

        # Cumulative boundary curves
        ax.plot(self.i_data, self.y1, color=c0, linestyle=(0, (4, 3)),
                linewidth=1.4, label=OVERPOT_LABELS["E_rev"])
        ax.plot(self.i_data, self.y2, color=c1, linewidth=1.6,
                label=f"+ {OVERPOT_LABELS['eta_kin']}")
        ax.plot(self.i_data, self.y3, color=c2, linewidth=1.6,
                label=f"+ {OVERPOT_LABELS['eta_ohm']}")
        ax.plot(self.i_data, self.y4, color=c3, linewidth=1.6,
                label=f"+ {OVERPOT_LABELS['eta_rcl']}")
        ax.plot(self.i_data, self.y5, color=c4, linewidth=1.6,
                label=f"+ {OVERPOT_LABELS['eta_res']}")
        ax.plot(self.i_data, self.y6, color=LSV_DATA_COLOR, marker="o",
                linestyle="none", markersize=3.2, markerfacecolor="white",
                markeredgewidth=1.0, label="LSV (RHE)")

        ax.set_xlabel(r"Current density  $i$  (A cm$^{-2}$)")
        ax.set_ylabel(r"Potential  $E$  (V)")
        ax.set_title("LSV decoupled into cumulative contributions")

        legend = ax.legend(
            loc="upper left", ncol=2, fontsize=9, frameon=False,
            handlelength=1.6, columnspacing=1.0,
        )
        for text in legend.get_texts():
            text.set_color(INK)

        # Reset interactive marker, will be redrawn by the breakdown update.
        self._target_marker = None
        self._target_line = None

        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _on_curve_click(self, event):
        """Set the breakdown target current by clicking on the curve plot."""
        if event.inaxes is not self.ax or self.fit_results is None:
            return
        if event.xdata is None or event.xdata <= 0:
            return

        i_min, i_max = float(np.min(self.i_data)), float(np.max(self.i_data))
        target = float(np.clip(event.xdata, i_min, i_max))

        self.target_i_entry.delete(0, tk.END)
        self.target_i_entry.insert(0, f"{target:.4g}")

        if self._slider is not None:
            try:
                self._slider.set_val(target)  # also triggers update
                return
            except Exception:
                pass
        self.update_bar_plot(auto_focus=False)

    def _draw_target_marker(self, target_i):
        """Draw a dotted vertical line + dot on the curves plot."""
        if self.i_data is None or self.fit_results is None:
            return
        sort_idx = np.argsort(self.i_data)
        i_sorted = self.i_data[sort_idx]
        V_sorted = self.V_data[sort_idx]
        V_at_target = float(np.interp(target_i, i_sorted, V_sorted))

        if self._target_line is None:
            self._target_line = self.ax.axvline(
                target_i, color=INK, linestyle=":", linewidth=1.0, alpha=0.6,
            )
        else:
            self._target_line.set_xdata([target_i, target_i])

        if self._target_marker is None:
            (self._target_marker,) = self.ax.plot(
                [target_i], [V_at_target], marker="o", markersize=8,
                markerfacecolor="white", markeredgecolor=ACCENT, markeredgewidth=1.6,
                linestyle="none", zorder=10,
            )
        else:
            self._target_marker.set_data([target_i], [V_at_target])

        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Overpotential breakdown (the redesigned plot)
    # ------------------------------------------------------------------
    def _draw_breakdown_placeholder(self):
        self.ax_bar.clear()
        self.ax_bar.set_axis_off()
        self.ax_slider.clear()
        self.ax_slider.set_axis_off()
        self.ax_bar.text(
            0.5, 0.5,
            "Run a fit to populate the overpotential breakdown.",
            ha="center", va="center", color=SUBTLE,
            transform=self.ax_bar.transAxes, fontsize=11,
        )
        self.canvas_bar.draw_idle()

    def _build_slider(self):
        """Create or refresh the matplotlib Slider for live target selection."""
        if self.i_data is None:
            return

        i_min = float(np.min(self.i_data))
        i_max = float(np.max(self.i_data))
        try:
            initial = float(self.target_i_entry.get())
        except ValueError:
            initial = 0.5 * (i_min + i_max)
        initial = float(np.clip(initial, i_min, i_max))

        self.ax_slider.clear()
        self.ax_slider.set_axis_on()

        self._slider = Slider(
            self.ax_slider, "i (A/cm²)", i_min, i_max,
            valinit=initial, valfmt="%.3g", color=ACCENT,
            track_color=BORDER, initcolor="none",
        )
        self._slider.label.set_color(INK)
        self._slider.label.set_fontsize(9)
        self._slider.valtext.set_color(INK)
        self._slider.valtext.set_fontsize(9)
        self._slider.on_changed(self._on_slider_change)

    def _on_slider_change(self, val):
        target = float(val)
        self.target_i_entry.delete(0, tk.END)
        self.target_i_entry.insert(0, f"{target:.4g}")
        self._render_breakdown(target)
        self._draw_target_marker(target)

    def update_bar_plot(self, auto_focus=True):
        if self.fit_results is None:
            if auto_focus:
                messagebox.showerror("Error", "Perform a fit first.")
            return

        try:
            target_i = float(self.target_i_entry.get())
        except ValueError:
            messagebox.showerror("Input Error",
                                 "Please ensure the target current is a valid number.")
            return

        # Sync the slider silently (avoid feedback loop with on_changed).
        if self._slider is not None:
            try:
                self._slider.eventson = False
                clipped = float(np.clip(target_i, self._slider.valmin, self._slider.valmax))
                self._slider.set_val(clipped)
                target_i = clipped
            finally:
                self._slider.eventson = True

        self._render_breakdown(target_i)
        self._draw_target_marker(target_i)

        if auto_focus:
            self.notebook.select(self.tab_bar)

    def _render_breakdown(self, target_i):
        """Render the wide horizontal stacked bar for the given target current."""
        try:
            HFR = float(self.hfr_entry.get())
            R_CL = float(self.rcl_entry.get())
        except ValueError:
            return

        comps = get_overpotential_at_current(
            target_i, self.i_data, self.V_data, self.E_rev, HFR, R_CL, self.fit_results,
        )

        ax = self.ax_bar
        ax.clear()
        ax.set_axis_on()
        style_axes(ax)
        ax.spines["left"].set_visible(False)
        ax.tick_params(left=False, labelleft=False)

        keys = ["E_rev", "eta_kin", "eta_ohm", "eta_rcl", "eta_res"]
        # Clip negatives (rare numerical artefact for very small i) to keep
        # the stack visually monotone.
        values = [max(float(comps[k]), 0.0) for k in keys]
        total = sum(values) if sum(values) > 0 else 1e-9

        bar_y = 0.0
        bar_h = 0.55
        cumulative = 0.0
        for key, val in zip(keys, values):
            color = OVERPOT_COLORS[key]
            ax.barh(bar_y, val, left=cumulative, height=bar_h,
                    color=color, edgecolor="white", linewidth=1.2, zorder=2)

            # Inside-segment annotation (only when wide enough to read)
            if val > 0.06 * total:
                ax.text(
                    cumulative + val / 2, bar_y,
                    f"{OVERPOT_LABELS[key]}\n{val * 1000:.0f} mV",
                    ha="center", va="center", color="white",
                    fontsize=10, fontweight="bold", zorder=3,
                )
            cumulative += val

        # Outer "envelope" tick marks for E_rev and total
        ax.axvline(comps["E_rev"], color=INK, linewidth=0.6, alpha=0.6,
                   linestyle="--", zorder=1)
        ax.text(
            comps["E_rev"], bar_y + 0.5,
            r"$E_\mathrm{rev}$" + f"\n{comps['E_rev']:.3f} V",
            ha="center", va="bottom", fontsize=9, color=INK,
        )
        ax.text(
            total, bar_y + 0.5,
            f"Total\n{total:.3f} V",
            ha="center", va="bottom", fontsize=9, color=INK, fontweight="bold",
        )
        ax.plot([total], [bar_y], marker="|", markersize=18,
                markeredgewidth=2, color=INK, zorder=4)

        # Compact legend chips beneath the bar with numerical values.
        chip_y = -0.55  # axes-fraction coords (transAxes)
        n = len(keys)
        for idx, (key, val) in enumerate(zip(keys, values)):
            x_frac = (idx + 0.5) / n
            ax.scatter(
                x_frac, chip_y, s=70, marker="s",
                color=OVERPOT_COLORS[key], edgecolor="none",
                transform=ax.transAxes, clip_on=False, zorder=5,
            )
            ax.text(
                x_frac, chip_y - 0.18,
                f"{OVERPOT_LABELS[key]}\n{val * 1000:.0f} mV",
                transform=ax.transAxes,
                ha="center", va="top", fontsize=9, color=INK,
                clip_on=False,
            )

        ax.set_xlabel(r"Potential  $E$  (V vs RHE)")
        ax.set_title(
            f"Overpotential breakdown   ·   i = {target_i:.4g} A cm$^{{-2}}$",
            loc="left", fontweight="bold",
        )
        ax.set_xlim(0, total * 1.15)
        ax.set_ylim(-0.6, 0.6)
        ax.set_yticks([])
        ax.grid(True, axis="x", alpha=0.25, linewidth=0.4)

        self.canvas_bar.draw_idle()

    # ------------------------------------------------------------------
    # Misc actions
    # ------------------------------------------------------------------
    def toggle_x_scale(self):
        if self.i_data is None:
            return
        self.x_log_scale = not self.x_log_scale
        self._plot_curves()
        # Restore the click-target marker if any
        try:
            target = float(self.target_i_entry.get())
            self._draw_target_marker(target)
        except ValueError:
            pass

    def export_data(self):
        if self.i_data is None:
            return

        df_curves = pd.DataFrame({
            "Current Density (A/cm²)": self.i_data,
            "E_rev": self.y1,
            "E_rev + η_kin": self.y2,
            "+ η_ohm": self.y3,
            "+ η_RCL": self.y4,
            "+ η_res": self.y5,
            "Original LSV (RHE)": self.y6,
        })

        df_comp = pd.DataFrame({
            "Current Density (A/cm²)": self.i_data,
            "η_kin (V)": self.eta_kin,
            "η_ohm (V)": self.Eta_ohm,
            "η_RCL (V)": self.Eta_RCL,
            "η_res (V)": self.Eta_res,
        })

        df_info = pd.DataFrame({
            "Parameter": [
                "Tafel slope (V/dec)", "Exchange current density (A/cm²)",
                "Intercept (V)", "R²", "N points",
            ],
            "Value": [
                self.fit_results["b_kin"], self.fit_results["i0"],
                self.fit_results["intercept"], self.fit_results["r_squared"],
                self.fit_results["n_points"],
            ],
        }) if self.fit_results else pd.DataFrame()

        file_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx", filetypes=[("Excel", "*.xlsx")],
        )
        if not file_path:
            return

        try:
            with pd.ExcelWriter(file_path) as writer:
                df_curves.to_excel(writer, sheet_name="Curves (stacked)", index=False)
                df_comp.to_excel(writer, sheet_name="Overpotential Components", index=False)
                df_info.to_excel(writer, sheet_name="Fitting Info", index=False)
            messagebox.showinfo("Export", "Data successfully exported.")
        except Exception as e:
            messagebox.showerror("Export Error", str(e))
