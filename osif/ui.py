import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from core.data_io import read_eis_data
from core.style import BG, INK, SUBTLE, ACCENT, BORDER, style_axes
from osif.logic import EisLogic


# Palette for EIS plots (data vs fit) — shares the language of the LSV module.
COLOR_DATA = "#1D4ED8"     # indigo for measured points
COLOR_FIT = "#B91C1C"      # crimson for the model curve
COLOR_VALID = "#1D4ED8"    # KK valid points
COLOR_INVALID = "#B91C1C"  # KK invalid points
COLOR_AVG = INK            # averaged spectrum line
COLOR_KK_THRESH = "#B45309"


class OsifUI(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.logic = EisLogic()
        self.raw_data_list = []
        self.df_all_diagnostics = None
        self.df_avg = None
        self.processed_f = self.processed_zr = self.processed_zi = None

        self.entries = {}
        self.se_labels = {}
        self.last_fit_data = None

        # Hover annotation state on the Nyquist plot
        self._hover_ann = None
        self._fit_data_artist = None
        self._fit_freq_array = None
        self._fit_z_array = None

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
            left_panel, text="EIS Fitting (OSIF)",
            font=("Segoe UI", 14, "bold"),
        ).pack(anchor="w", pady=(0, 2))
        ttk.Label(
            left_panel,
            text="Validate, average and fit non-Faradaic\nimpedance to extract HFR and R_CL.",
            style="Hint.TLabel", justify="left",
        ).pack(anchor="w", pady=(0, 10))

        # File Management
        file_frame = ttk.LabelFrame(left_panel, text="Loaded EIS files")
        file_frame.pack(fill=tk.X, pady=(0, 8))

        listbox_wrap = ttk.Frame(file_frame)
        listbox_wrap.pack(fill=tk.X, padx=2, pady=(2, 6))
        self.listbox_files = tk.Listbox(
            listbox_wrap, height=5, selectmode=tk.EXTENDED,
            background=BG, foreground=INK, relief="flat",
            highlightthickness=1, highlightbackground=BORDER,
            selectbackground="#EEF2FF", selectforeground=INK,
            activestyle="none",
        )
        self.listbox_files.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        btn_box = ttk.Frame(file_frame)
        btn_box.pack(fill=tk.X)
        ttk.Button(btn_box, text="Add", command=self.add_files).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2),
        )
        ttk.Button(btn_box, text="Remove", command=self.remove_files).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=2,
        )
        ttk.Button(btn_box, text="Clear", command=self.clear_files).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=(2, 0),
        )

        # Validation & HFR Frame
        val_frame = ttk.LabelFrame(left_panel, text="1.  Validation & averaging")
        val_frame.pack(fill=tk.X, pady=8)

        ttk.Button(val_frame, text="Run preprocess & KK test",
                   command=self.run_preprocessing,
                   style="Accent.TButton").pack(fill=tk.X, pady=(0, 6))

        freq_f = ttk.Frame(val_frame)
        freq_f.pack(fill=tk.X, pady=2)
        freq_f.columnconfigure((1, 3), weight=1)
        ttk.Label(freq_f, text="f_max (Hz)").grid(row=0, column=0, sticky="w", padx=2)
        self.ent_fmax = ttk.Entry(freq_f, width=10)
        self.ent_fmax.grid(row=0, column=1, sticky="ew", padx=2)
        ttk.Label(freq_f, text="f_min (Hz)").grid(row=0, column=2, sticky="w", padx=(8, 2))
        self.ent_fmin = ttk.Entry(freq_f, width=10)
        self.ent_fmin.grid(row=0, column=3, sticky="ew", padx=2)

        hfr_f = ttk.Frame(val_frame)
        hfr_f.pack(fill=tk.X, pady=(8, 2))
        hfr_f.columnconfigure(1, weight=1)
        ttk.Label(hfr_f, text="HFR (Ω·cm²)").grid(row=0, column=0, sticky="w", padx=2)
        self.entries["HFR"] = ttk.Entry(hfr_f, width=12)
        self.entries["HFR"].insert(0, "0.2")
        self.entries["HFR"].grid(row=0, column=1, sticky="ew", padx=2)
        self.se_labels["HFR"] = ttk.Label(hfr_f, text="± —", style="Hint.TLabel")
        self.se_labels["HFR"].grid(row=0, column=2, sticky="w", padx=(6, 2))

        self.lbl_status = ttk.Label(val_frame, text="", style="Status.TLabel")
        self.lbl_status.pack(anchor="w", pady=(6, 0))

        # Impedance Fitting Frame
        self.fit_container = ttk.LabelFrame(left_panel, text="2.  Impedance fitting")
        self.fit_container.pack(fill=tk.X, pady=8)

        self.do_fit_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            self.fit_container, text="Enable impedance fitting (non-Faradaic)",
            variable=self.do_fit_var, command=self.toggle_fit_ui,
        ).pack(anchor="w", pady=(0, 6))

        self.param_frame = ttk.Frame(self.fit_container)
        self.param_frame.pack(fill=tk.X)
        self.param_frame.columnconfigure(1, weight=1)

        header_font = ("Segoe UI", 9, "italic")
        ttk.Label(self.param_frame, text="Parameter", font=header_font, foreground=SUBTLE).grid(
            row=0, column=0, pady=(0, 4), sticky="w",
        )
        ttk.Label(self.param_frame, text="Value", font=header_font, foreground=SUBTLE).grid(
            row=0, column=1, pady=(0, 4), sticky="w",
        )
        ttk.Label(self.param_frame, text="Std. error (%)", font=header_font, foreground=SUBTLE).grid(
            row=0, column=2, pady=(0, 4), sticky="w",
        )

        labels = ["Lwire (H·cm²)", "Rcl (Ω·cm²)", "Qdl (F)", "Phi (—)", "Theta (—)"]
        defaults = ["0", "0.2", "0.1", "0.9", "0"]

        for i, (lbl, df) in enumerate(zip(labels, defaults)):
            key = lbl.split()[0]
            ttk.Label(self.param_frame, text=lbl).grid(
                row=i + 1, column=0, sticky="w", padx=2, pady=2,
            )
            ent = ttk.Entry(self.param_frame, width=12)
            ent.insert(0, df)
            ent.grid(row=i + 1, column=1, sticky="ew", padx=2, pady=2)
            self.entries[key] = ent

            se_lbl = ttk.Label(self.param_frame, text="± —", style="Hint.TLabel")
            se_lbl.grid(row=i + 1, column=2, sticky="w", padx=(6, 2))
            self.se_labels[key] = se_lbl

            if key in ["Lwire", "Theta"]:
                ent.config(state="readonly")
                se_lbl.config(text="fixed at 0")

        self.model_var = tk.StringVar(value="Transmission Line")
        models = ["Transmission Line", "1-D Linear Diffusion", "1-D Spherical Diffusion"]
        ttk.Label(self.fit_container, text="Model").pack(anchor="w", pady=(8, 2))
        self.cmb_model = ttk.Combobox(
            self.fit_container, textvariable=self.model_var,
            values=models, state="readonly",
        )
        self.cmb_model.pack(fill=tk.X)

        self.btn_fit = ttk.Button(
            self.fit_container, text="Fit model", command=self.run_fitting,
            style="Accent.TButton",
        )
        self.btn_fit.pack(fill=tk.X, pady=(8, 0))

        ttk.Separator(left_panel).pack(fill=tk.X, pady=10)
        ttk.Button(left_panel, text="Export results…", command=self.export_results).pack(fill=tk.X)

        # --- Right Panel: Plotting Tabs ---
        right_panel = ttk.Frame(self)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(8, 14), pady=14)

        self.notebook = ttk.Notebook(right_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 1: KK diagnostics
        self.tab_diag = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_diag, text="KK diagnostics")
        self.fig_diag, self.ax_diag = plt.subplots(1, 2, figsize=(10, 4))
        self.fig_diag.patch.set_facecolor(BG)
        for ax in self.ax_diag:
            style_axes(ax)
        self.canvas_diag = FigureCanvasTkAgg(self.fig_diag, master=self.tab_diag)
        self.canvas_diag.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        toolbar_frame_diag = ttk.Frame(self.tab_diag)
        toolbar_frame_diag.pack(fill=tk.X, side=tk.BOTTOM)
        self.toolbar_diag = NavigationToolbar2Tk(self.canvas_diag, toolbar_frame_diag)
        self.toolbar_diag.update()

        # Tab 2: Fit
        self.tab_fit = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_fit, text="Fit results")
        self.fig_fit, self.ax_fit = plt.subplots(2, 2, figsize=(10, 6))
        self.fig_fit.patch.set_facecolor(BG)
        for ax in self.ax_fit.flat:
            style_axes(ax)
        self.canvas_fit = FigureCanvasTkAgg(self.fig_fit, master=self.tab_fit)
        self.canvas_fit.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas_fit.mpl_connect("motion_notify_event", self._on_fit_hover)

        toolbar_frame_fit = ttk.Frame(self.tab_fit)
        toolbar_frame_fit.pack(fill=tk.X, side=tk.BOTTOM)
        self.toolbar_fit = NavigationToolbar2Tk(self.canvas_fit, toolbar_frame_fit)
        self.toolbar_fit.update()

    def toggle_fit_ui(self):
        state = "normal" if self.do_fit_var.get() else "disabled"
        state_cmb = "readonly" if self.do_fit_var.get() else "disabled"

        for child in self.param_frame.winfo_children():
            if isinstance(child, ttk.Entry):
                if child in [self.entries["Lwire"], self.entries["Theta"]]:
                    continue
            child.configure(state=state)

        self.cmb_model.configure(state=state_cmb)
        self.btn_fit.configure(state=state)

    # ------------------------------------------------------------------
    # File management
    # ------------------------------------------------------------------
    def add_files(self):
        filepaths = filedialog.askopenfilenames(filetypes=[("EIS Files", "*.txt *.csv *.xlsx")])
        if not filepaths:
            return

        for fp in filepaths:
            data, err = read_eis_data(fp)
            if data:
                filename = os.path.basename(fp)
                self.raw_data_list.append(data)
                self.listbox_files.insert(tk.END, filename)
            else:
                messagebox.showwarning("Load Error", err)

    def remove_files(self):
        selected_indices = list(self.listbox_files.curselection())
        selected_indices.reverse()
        for idx in selected_indices:
            self.listbox_files.delete(idx)
            del self.raw_data_list[idx]

    def clear_files(self):
        self.listbox_files.delete(0, tk.END)
        self.raw_data_list.clear()
        self.df_all_diagnostics = None
        self.df_avg = None
        self.processed_f = self.processed_zr = self.processed_zi = None
        self.lbl_status.config(text="")

        for ax in self.ax_diag:
            ax.clear()
            style_axes(ax)
        for ax in self.ax_fit.flat:
            ax.clear()
            style_axes(ax)
        self.canvas_diag.draw_idle()
        self.canvas_fit.draw_idle()

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------
    def run_preprocessing(self):
        if not self.raw_data_list:
            messagebox.showwarning("Warning", "No files loaded. Please add files first.")
            return

        df_all, df_avg, (fmin, fmax), rsd = self.logic.process_spectra(self.raw_data_list)
        self.df_all_diagnostics = df_all
        self.df_avg = df_avg

        self.processed_f = df_avg["Freq(Hz)"].values
        self.processed_zr = df_avg["Z'(Ohm.cm²)"].values
        self.processed_zi = df_avg["Z''(Ohm.cm²)"].values

        self.ent_fmax.delete(0, tk.END)
        self.ent_fmax.insert(0, f"{fmax:.2f}")
        self.ent_fmin.delete(0, tk.END)
        self.ent_fmin.insert(0, f"{fmin:.2f}")

        est_hfr = float(np.min(self.processed_zr))
        self.entries["HFR"].delete(0, tk.END)
        self.entries["HFR"].insert(0, f"{est_hfr:.4f}")

        for k in self.se_labels:
            if k not in ["Lwire", "Theta"]:
                self.se_labels[k].config(text="± —")

        ax_n, ax_b = self.ax_diag
        ax_n.clear(); style_axes(ax_n)
        ax_b.clear(); style_axes(ax_b)

        mask_valid = df_all["kk_err"] <= self.logic.kk_threshold
        ax_n.scatter(
            df_all.loc[mask_valid, "zr"], -df_all.loc[mask_valid, "zi"],
            c=COLOR_VALID, alpha=0.45, s=18, edgecolor="none",
            label="KK valid",
        )
        ax_n.scatter(
            df_all.loc[~mask_valid, "zr"], -df_all.loc[~mask_valid, "zi"],
            c=COLOR_INVALID, marker="x", s=22, linewidth=1.2,
            label="KK invalid",
        )
        ax_n.plot(
            self.processed_zr, -self.processed_zi,
            color=COLOR_AVG, linewidth=1.8, label="Averaged",
        )
        ax_n.set_aspect("equal", adjustable="datalim")
        ax_n.set_xlabel(r"Z′  (Ω·cm²)")
        ax_n.set_ylabel(r"−Z″  (Ω·cm²)")
        ax_n.set_title("Nyquist  ·  KK validation")
        leg = ax_n.legend(loc="best", frameon=False)
        for t in leg.get_texts():
            t.set_color(INK)

        ax_b.scatter(
            df_all["f"], df_all["kk_err"] * 100,
            c=SUBTLE, alpha=0.6, s=12, edgecolor="none",
        )
        ax_b.axhline(
            self.logic.kk_threshold * 100, color=COLOR_KK_THRESH,
            linestyle="--", linewidth=1.2,
            label=f"threshold ({self.logic.kk_threshold*100:.0f}%)",
        )
        ax_b.set_xscale("log")
        ax_b.set_yscale("log")
        ax_b.set_xlabel(r"Frequency  (Hz)")
        ax_b.set_ylabel(r"KK residual  (%)")
        ax_b.set_title("Bode  ·  point-wise error")
        leg2 = ax_b.legend(loc="best", frameon=False)
        for t in leg2.get_texts():
            t.set_color(INK)

        self.fig_diag.tight_layout()
        self.canvas_diag.draw_idle()
        self.notebook.select(self.tab_diag)

        if len(self.raw_data_list) >= 3:
            self.lbl_status.config(text=f"{len(self.raw_data_list)} files averaged. RSD: {rsd:.2%}")
        else:
            self.lbl_status.config(text=f"{len(self.raw_data_list)} file(s) processed.")

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def run_fitting(self):
        if self.processed_f is None:
            messagebox.showerror("Error", "Run preprocessing first.")
            return

        fmax, fmin = float(self.ent_fmax.get()), float(self.ent_fmin.get())
        mask = (self.processed_f >= fmin) & (self.processed_f <= fmax)
        f_fit = self.processed_f[mask]
        z_exp = self.processed_zr[mask] + 1j * self.processed_zi[mask]

        free_keys = ["HFR", "Rcl", "Qdl", "Phi"]
        init_p = [float(self.entries[k].get()) for k in free_keys]

        results, error_msg = self.logic.fit_impedance(self.model_var.get(), init_p, f_fit, z_exp)

        if error_msg:
            messagebox.showerror("Fitting Error", error_msg)
            return

        params, se, residuals = results

        for i, k in enumerate(free_keys):
            self.entries[k].delete(0, tk.END)
            self.entries[k].insert(0, f"{params[i]:.6f}")
            pct_error = (se[i] / params[i] * 100) if params[i] != 0 else 0
            self.se_labels[k].config(text=f"± {se[i]:.3g} ({pct_error:.1f}%)")

        Z_model = self.logic.evaluate_model(self.model_var.get(), params, f_fit)

        self.last_fit_data = {
            "f_fit": f_fit,
            "z_exp": z_exp,
            "Z_model": Z_model,
            "keys": ["Lwire", "HFR", "Rcl", "Qdl", "Phi", "Theta"],
            "params": [0, params[0], params[1], params[2], params[3], 0],
            "se": [0, se[0], se[1], se[2], se[3], 0],
        }

        # Cache for hover-tooltip
        self._fit_freq_array = f_fit
        self._fit_z_array = z_exp

        for ax in self.ax_fit.flat:
            ax.clear()
            style_axes(ax)

        ax_ny, ax_bode, ax_re, ax_im = (
            self.ax_fit[0, 0], self.ax_fit[0, 1],
            self.ax_fit[1, 0], self.ax_fit[1, 1],
        )

        # Nyquist
        self._fit_data_artist, = ax_ny.plot(
            np.real(z_exp), -np.imag(z_exp),
            marker="o", linestyle="none", markersize=4,
            markerfacecolor="white", markeredgecolor=COLOR_DATA,
            markeredgewidth=1.2, label="Data",
        )
        ax_ny.plot(
            np.real(Z_model), -np.imag(Z_model),
            color=COLOR_FIT, linewidth=1.8, label="Fit",
        )
        ax_ny.set_aspect("equal", adjustable="datalim")
        ax_ny.set_xlabel(r"Z′  (Ω·cm²)")
        ax_ny.set_ylabel(r"−Z″  (Ω·cm²)")
        ax_ny.set_title("Nyquist", loc="left")

        # Inline parameter readout in the Nyquist axes corner
        param_text = (
            f"HFR  = {params[0]:.4f} Ω·cm²\n"
            f"R_CL = {params[1]:.4f} Ω·cm²\n"
            f"Q_dl = {params[2]:.3g} F\n"
            f"φ    = {params[3]:.3f}"
        )
        ax_ny.text(
            0.97, 0.05, param_text, transform=ax_ny.transAxes,
            ha="right", va="bottom", fontsize=9, color=INK,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", fc="white",
                      ec=BORDER, lw=0.6, alpha=0.9),
        )
        leg = ax_ny.legend(loc="upper left", frameon=False)
        for t in leg.get_texts():
            t.set_color(INK)

        # Bode magnitude
        ax_bode.loglog(
            f_fit, np.abs(z_exp), marker="o", linestyle="none",
            markersize=4, markerfacecolor="white",
            markeredgecolor=COLOR_DATA, markeredgewidth=1.2, label="Data",
        )
        ax_bode.loglog(f_fit, np.abs(Z_model), color=COLOR_FIT, linewidth=1.6, label="Fit")
        ax_bode.set_xlabel(r"Frequency  (Hz)")
        ax_bode.set_ylabel(r"|Z|  (Ω·cm²)")
        ax_bode.set_title("Bode  ·  |Z|", loc="left")

        # Real & Imag vs frequency
        ax_re.semilogx(
            f_fit, np.real(z_exp), marker="o", linestyle="none",
            markersize=4, markerfacecolor="white",
            markeredgecolor=COLOR_DATA, markeredgewidth=1.2,
        )
        ax_re.semilogx(f_fit, np.real(Z_model), color=COLOR_FIT, linewidth=1.6)
        ax_re.set_xlabel(r"Frequency  (Hz)")
        ax_re.set_ylabel(r"Re(Z)  (Ω·cm²)")
        ax_re.set_title("Re(Z)", loc="left")

        ax_im.semilogx(
            f_fit, -np.imag(z_exp), marker="o", linestyle="none",
            markersize=4, markerfacecolor="white",
            markeredgecolor=COLOR_DATA, markeredgewidth=1.2,
        )
        ax_im.semilogx(f_fit, -np.imag(Z_model), color=COLOR_FIT, linewidth=1.6)
        ax_im.set_xlabel(r"Frequency  (Hz)")
        ax_im.set_ylabel(r"−Im(Z)  (Ω·cm²)")
        ax_im.set_title("−Im(Z)", loc="left")

        # Hover annotation (lazy)
        self._hover_ann = ax_ny.annotate(
            "", xy=(0, 0), xytext=(12, 12), textcoords="offset points",
            fontsize=9, color=INK,
            bbox=dict(boxstyle="round,pad=0.4", fc="white",
                      ec=BORDER, lw=0.6, alpha=0.95),
            arrowprops=dict(arrowstyle="-", color=SUBTLE, lw=0.6),
        )
        self._hover_ann.set_visible(False)

        self.fig_fit.tight_layout()
        self.canvas_fit.draw_idle()
        self.notebook.select(self.tab_fit)

    def _on_fit_hover(self, event):
        if (self._fit_data_artist is None or self._fit_z_array is None
                or self._hover_ann is None):
            return
        if event.inaxes is not self.ax_fit[0, 0]:
            if self._hover_ann.get_visible():
                self._hover_ann.set_visible(False)
                self.canvas_fit.draw_idle()
            return

        contains, info = self._fit_data_artist.contains(event)
        if contains:
            idx = info["ind"][0]
            f = float(self._fit_freq_array[idx])
            z = self._fit_z_array[idx]
            x, y = float(np.real(z)), -float(np.imag(z))
            self._hover_ann.xy = (x, y)
            self._hover_ann.set_text(
                f"f = {f:.3g} Hz\nZ′ = {x:.4f} Ω·cm²\n−Z″ = {y:.4f} Ω·cm²",
            )
            self._hover_ann.set_visible(True)
            self.canvas_fit.draw_idle()
        elif self._hover_ann.get_visible():
            self._hover_ann.set_visible(False)
            self.canvas_fit.draw_idle()

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def export_results(self):
        if self.df_all_diagnostics is None:
            messagebox.showerror(
                "Export Error",
                "No data available to export. Run 'Preprocess & KK Test' first.",
            )
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx")],
            title="Save OSIF Results As",
        )
        if not file_path:
            return

        try:
            with pd.ExcelWriter(file_path) as writer:
                self.df_all_diagnostics.to_excel(writer, sheet_name="KK Diagnostics & Raw", index=False)

                if self.df_avg is not None:
                    self.df_avg.to_excel(writer, sheet_name="Averaged Data", index=False)

                if self.last_fit_data is not None and self.do_fit_var.get():
                    df_curve = pd.DataFrame({
                        "Frequency (Hz)": self.last_fit_data["f_fit"],
                        "Data Re(Z)": np.real(self.last_fit_data["z_exp"]),
                        "Data Im(Z)": np.imag(self.last_fit_data["z_exp"]),
                        "Fit Re(Z)": np.real(self.last_fit_data["Z_model"]),
                        "Fit Im(Z)": np.imag(self.last_fit_data["Z_model"]),
                        "Fit |Z|": np.abs(self.last_fit_data["Z_model"]),
                    })
                    df_curve.to_excel(writer, sheet_name="Fitted Curve", index=False)

                    df_params = pd.DataFrame({
                        "Parameter": self.last_fit_data["keys"],
                        "Fitted Value": self.last_fit_data["params"],
                        "Standard Error": self.last_fit_data["se"],
                    })
                    df_params.to_excel(writer, sheet_name="Fit Parameters", index=False)

            messagebox.showinfo("Export Successful", f"Results successfully exported to:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Export Error", f"An error occurred while exporting:\n{e}")
