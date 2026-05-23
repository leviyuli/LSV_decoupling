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


# Colour used to flag fit parameters whose relative standard error exceeds
# the high-uncertainty threshold from EisLogic.fit_high_uncertainty_pct.
COLOR_HIGH_SE = "#B91C1C"


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
        self.limit_kk_range_var = tk.BooleanVar(value=False)
        self.fit_max_nfev_var = tk.StringVar(value=str(self.logic.fit_max_nfev))
        self.fit_n_restarts_var = tk.StringVar(value=str(self.logic.fit_n_restarts))

        self.entries = {}
        self.se_labels = {}
        self.param_labels = {}
        self.last_fit_data = None

        # Hover annotation state on the Nyquist plot
        self._hover_ann = None
        self._fit_data_artist = None
        self._fit_freq_array = None
        self._fit_z_array = None

        # Simulate tab state (parallel to the fit grid, no SE labels).
        self.sim_entries = {}
        self.sim_param_labels = {}
        self.sim_freq_entries = {}
        self.lbl_sim_status = None
        self.lbl_sim_model = None

        self._build_layout()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def _build_layout(self):
        self.configure(style="TFrame")

        # --- Left Panel: Controls (scrollable so a short window can't clip
        # the Fit button or the Export button at the bottom). ---
        left_container = ttk.Frame(self)
        left_container.pack(side=tk.LEFT, fill=tk.Y, padx=(14, 8), pady=14)

        left_canvas = tk.Canvas(
            left_container, background=BG, borderwidth=0, highlightthickness=0,
        )
        left_scroll = ttk.Scrollbar(
            left_container, orient=tk.VERTICAL, command=left_canvas.yview,
        )
        left_canvas.configure(yscrollcommand=left_scroll.set)
        left_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        left_canvas.pack(side=tk.LEFT, fill=tk.Y, expand=False)

        left_panel = ttk.Frame(left_canvas, style="TFrame")
        left_canvas.create_window((0, 0), window=left_panel, anchor="nw")

        def _on_inner_configure(_event):
            left_canvas.configure(scrollregion=left_canvas.bbox("all"))
            # Match the canvas width to the inner content so widgets don't squeeze.
            req_w = left_panel.winfo_reqwidth()
            if left_canvas.winfo_width() != req_w:
                left_canvas.configure(width=req_w)

        left_panel.bind("<Configure>", _on_inner_configure)

        # Mouse-wheel scrolling, scoped to the canvas (Windows delta = ±120/notch).
        def _on_mousewheel(event):
            left_canvas.yview_scroll(int(-event.delta / 120), "units")

        left_canvas.bind(
            "<Enter>", lambda _e: left_canvas.bind_all("<MouseWheel>", _on_mousewheel),
        )
        left_canvas.bind(
            "<Leave>", lambda _e: left_canvas.unbind_all("<MouseWheel>"),
        )

        ttk.Label(
            left_panel, text="EIS Fitting (OSIF)",
            font=("Segoe UI", 14, "bold"),
        ).pack(anchor="w", pady=(0, 2))
        ttk.Label(
            left_panel,
            text="Validate, average and fit impedance\nto extract HFR and R_CL.",
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

        ttk.Checkbutton(
            val_frame,
            text="Limit KK to f_min/f_max",
            variable=self.limit_kk_range_var,
        ).pack(anchor="w", pady=(4, 2))

        hfr_f = ttk.Frame(val_frame)
        hfr_f.pack(fill=tk.X, pady=(8, 2))
        hfr_f.columnconfigure(1, weight=1)
        ttk.Label(hfr_f, text="HFR (Ω·cm²)").grid(row=0, column=0, sticky="w", padx=2)
        self.entries["HFR"] = ttk.Entry(hfr_f, width=12)
        self.entries["HFR"].insert(0, "0.2")
        self.entries["HFR"].grid(row=0, column=1, sticky="ew", padx=2)
        self.se_labels["HFR"] = ttk.Label(hfr_f, text="± —", style="Hint.TLabel")
        self.se_labels["HFR"].grid(row=0, column=2, sticky="w", padx=(6, 2))

        self.lbl_status = ttk.Label(
            val_frame, text="", style="Status.TLabel",
            wraplength=300, justify="left",
        )
        self.lbl_status.pack(anchor="w", fill=tk.X, pady=(6, 0))

        # Impedance Fitting Frame
        self.fit_container = ttk.LabelFrame(left_panel, text="2.  Impedance fitting")
        self.fit_container.pack(fill=tk.X, pady=8)

        self.do_fit_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            self.fit_container, text="Enable impedance fitting",
            variable=self.do_fit_var, command=self.toggle_fit_ui,
        ).pack(anchor="w", pady=(0, 6))

        self.fit_inductance_var = tk.BooleanVar(value=False)
        self.chk_fit_inductance = ttk.Checkbutton(
            self.fit_container,
            text="Fit wire inductance (L_wire, Θ)",
            variable=self.fit_inductance_var,
            command=self.toggle_inductance_ui,
        )
        self.chk_fit_inductance.pack(anchor="w", pady=(0, 6))

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

        labels = [
            "Lwire (H·cm²)",
            "Rcl (Ω·cm²)",
            "Qdl (F)",
            "Phi (—)",
            "Theta (—)",
            "Rk (Ω·cm²)",
        ]
        defaults = ["0", "0.2", "0.1", "0.9", "0", "1.0"]

        for i, (lbl, df) in enumerate(zip(labels, defaults)):
            key = lbl.split()[0]
            label = ttk.Label(self.param_frame, text=lbl)
            label.grid(
                row=i + 1, column=0, sticky="w", padx=2, pady=2,
            )
            self.param_labels[key] = label
            ent = ttk.Entry(self.param_frame, width=12)
            ent.insert(0, df)
            ent.grid(row=i + 1, column=1, sticky="ew", padx=2, pady=2)
            self.entries[key] = ent

            se_lbl = ttk.Label(self.param_frame, text="± —", style="Hint.TLabel")
            se_lbl.grid(row=i + 1, column=2, sticky="w", padx=(6, 2))
            self.se_labels[key] = se_lbl

            if key in ["Lwire", "Theta", "Rk"]:
                ent.config(state="readonly")
                se_lbl.config(text="unused" if key == "Rk" else "fixed at 0")

        self.model_var = tk.StringVar(value="Transmission Line")
        models = [
            "Transmission Line",
            self.logic.FARADAIC_TML_MODEL,
            "1-D Linear Diffusion",
            "1-D Spherical Diffusion",
        ]
        ttk.Label(self.fit_container, text="Model").pack(anchor="w", pady=(8, 2))
        self.cmb_model = ttk.Combobox(
            self.fit_container, textvariable=self.model_var,
            values=models, state="readonly",
        )
        self.cmb_model.pack(fill=tk.X)
        self.cmb_model.bind("<<ComboboxSelected>>", lambda _event: self.toggle_model_ui())

        eval_f = ttk.Frame(self.fit_container)
        eval_f.pack(fill=tk.X, pady=(8, 0))
        eval_f.columnconfigure(1, weight=1)
        ttk.Label(eval_f, text="Max evaluations").grid(row=0, column=0, sticky="w", padx=2)
        self.ent_fit_max_nfev = ttk.Entry(
            eval_f, width=12, textvariable=self.fit_max_nfev_var,
        )
        self.ent_fit_max_nfev.grid(row=0, column=1, sticky="ew", padx=2)

        ttk.Label(eval_f, text="Restarts").grid(row=1, column=0, sticky="w", padx=2, pady=(4, 0))
        self.ent_fit_n_restarts = ttk.Entry(
            eval_f, width=12, textvariable=self.fit_n_restarts_var,
        )
        self.ent_fit_n_restarts.grid(row=1, column=1, sticky="ew", padx=2, pady=(4, 0))
        ttk.Label(
            self.fit_container,
            text="Multi-start: ranks N tries by HFR / R_CL accuracy.",
            style="Hint.TLabel",
            wraplength=300, justify="left",
        ).pack(anchor="w", fill=tk.X, pady=(2, 0))

        self.lbl_fit_status = ttk.Label(
            self.fit_container, text="", style="Status.TLabel",
            wraplength=300, justify="left",
        )
        self.lbl_fit_status.pack(anchor="w", fill=tk.X, pady=(4, 0))

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

        # Tab 3: Simulate — manual parameter sweeps over the forward model.
        self.tab_sim = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_sim, text="Simulate")

        self.tab_sim.columnconfigure(0, weight=1, uniform="sim_cols")
        self.tab_sim.columnconfigure(1, weight=2, uniform="sim_cols")
        self.tab_sim.rowconfigure(0, weight=1)

        sim_ctrl = ttk.Frame(self.tab_sim)
        sim_ctrl.grid(row=0, column=0, sticky="nsew", padx=(8, 4), pady=8)

        sim_plot = ttk.Frame(self.tab_sim)
        sim_plot.grid(row=0, column=1, sticky="nsew", padx=(4, 8), pady=8)

        ttk.Label(
            sim_ctrl, text="Simulation parameters",
            font=("Segoe UI", 11, "bold"),
        ).pack(anchor="w", pady=(0, 6))

        ttk.Label(sim_ctrl, text="Model", style="Hint.TLabel").pack(anchor="w")
        self.lbl_sim_model = ttk.Label(sim_ctrl, text=self.model_var.get())
        self.lbl_sim_model.pack(anchor="w", pady=(0, 8))
        self.model_var.trace_add("write", self._on_model_var_changed)

        sim_grid = ttk.Frame(sim_ctrl)
        sim_grid.pack(fill=tk.X)
        sim_grid.columnconfigure(1, weight=1)

        header_font = ("Segoe UI", 9, "italic")
        ttk.Label(sim_grid, text="Parameter", font=header_font, foreground=SUBTLE).grid(
            row=0, column=0, pady=(0, 4), sticky="w",
        )
        ttk.Label(sim_grid, text="Value", font=header_font, foreground=SUBTLE).grid(
            row=0, column=1, pady=(0, 4), sticky="w",
        )

        sim_labels = [
            ("Lwire", "Lwire (H·cm²)", "0"),
            ("HFR", "HFR (Ω·cm²)", "0.2"),
            ("Rcl", "Rcl (Ω·cm²)", "0.2"),
            ("Qdl", "Qdl (F)", "0.1"),
            ("Phi", "Phi (—)", "0.9"),
            ("Theta", "Theta (—)", "0"),
            ("Rk", "Rk (Ω·cm²)", "1.0"),
        ]
        for i, (key, lbl, default) in enumerate(sim_labels):
            label = ttk.Label(sim_grid, text=lbl)
            label.grid(row=i + 1, column=0, sticky="w", padx=2, pady=2)
            self.sim_param_labels[key] = label

            ent = ttk.Entry(sim_grid, width=12)
            ent.insert(0, default)
            ent.grid(row=i + 1, column=1, sticky="ew", padx=2, pady=2)
            self.sim_entries[key] = ent

            if key in ("Lwire", "Theta"):
                ent.configure(state="readonly")

        ttk.Label(
            sim_ctrl, text="Frequency range",
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor="w", pady=(12, 4))

        freq_grid = ttk.Frame(sim_ctrl)
        freq_grid.pack(fill=tk.X)
        freq_grid.columnconfigure(1, weight=1)
        freq_rows = [
            ("fmin", "f_min (Hz)", "0.01"),
            ("fmax", "f_max (Hz)", "1e6"),
            ("points", "Points", "100"),
        ]
        for i, (key, lbl, default) in enumerate(freq_rows):
            ttk.Label(freq_grid, text=lbl).grid(
                row=i, column=0, sticky="w", padx=2, pady=2,
            )
            ent = ttk.Entry(freq_grid, width=12)
            ent.insert(0, default)
            ent.grid(row=i, column=1, sticky="ew", padx=2, pady=2)
            self.sim_freq_entries[key] = ent

        btn_box = ttk.Frame(sim_ctrl)
        btn_box.pack(fill=tk.X, pady=(10, 0))
        ttk.Button(
            btn_box, text="Sync from fit", command=self.sync_sim_from_fit,
        ).pack(fill=tk.X)
        ttk.Button(
            btn_box, text="Simulate", command=self.run_simulation,
            style="Accent.TButton",
        ).pack(fill=tk.X, pady=(6, 0))

        self.lbl_sim_status = ttk.Label(sim_ctrl, text="", style="Status.TLabel")
        self.lbl_sim_status.pack(anchor="w", pady=(8, 0))

        self.fig_sim, self.ax_sim = plt.subplots(2, 2, figsize=(10, 6))
        self.fig_sim.patch.set_facecolor(BG)
        for ax in self.ax_sim.flat:
            style_axes(ax)
        self.canvas_sim = FigureCanvasTkAgg(self.fig_sim, master=sim_plot)
        self.canvas_sim.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        toolbar_frame_sim = ttk.Frame(sim_plot)
        toolbar_frame_sim.pack(fill=tk.X, side=tk.BOTTOM)
        self.toolbar_sim = NavigationToolbar2Tk(self.canvas_sim, toolbar_frame_sim)
        self.toolbar_sim.update()

        self.toggle_model_ui()

    def _is_faradaic_model(self):
        return self.logic.is_faradaic_model(self.model_var.get())

    def _active_fit_keys(self, fit_inductance, fit_faradaic):
        if fit_inductance and fit_faradaic:
            return ["Lwire", "HFR", "Rcl", "Qdl", "Phi", "Theta", "Rk"]
        if fit_inductance:
            return ["Lwire", "HFR", "Rcl", "Qdl", "Phi", "Theta"]
        if fit_faradaic:
            return ["HFR", "Rcl", "Qdl", "Phi", "Rk"]
        return ["HFR", "Rcl", "Qdl", "Phi"]

    def _export_parameter_rows(self, params, se, fit_faradaic):
        canonical_keys = list(self.logic.CANONICAL_KEYS)
        export_keys = canonical_keys if fit_faradaic else canonical_keys[:-1]
        export_params = [params[canonical_keys.index(k)] for k in export_keys]
        export_se = [se[canonical_keys.index(k)] for k in export_keys]

        if self.model_var.get() == "Transmission Line" and not fit_faradaic:
            export_keys.append("Rcl/3 low-frequency intercept")
            export_params.append(params[canonical_keys.index("Rcl")] / 3.0)
            export_se.append(se[canonical_keys.index("Rcl")] / 3.0)

        return export_keys, export_params, export_se

    def toggle_fit_ui(self):
        fit_on = self.do_fit_var.get()
        state = "normal" if fit_on else "disabled"
        state_cmb = "readonly" if fit_on else "disabled"

        for child in self.param_frame.winfo_children():
            if isinstance(child, ttk.Entry):
                if child in [self.entries["Lwire"], self.entries["Theta"], self.entries["Rk"]]:
                    # These entries are gated by their dedicated toggles below.
                    continue
            child.configure(state=state)

        self.cmb_model.configure(state=state_cmb)
        self.ent_fit_max_nfev.configure(state=state)
        self.ent_fit_n_restarts.configure(state=state)
        self.chk_fit_inductance.configure(state=state)
        self.btn_fit.configure(state=state)
        # Re-apply lock-states consistent with the parent toggle.
        self.toggle_inductance_ui()
        self.toggle_model_ui()

    def _apply_inductance_state(self, entries, inductance_on, se_labels=None):
        if inductance_on:
            for key, default in (("Lwire", "2e-5"), ("Theta", "0.95")):
                ent = entries[key]
                ent.configure(state="normal")
                current = ent.get().strip()
                if current in ("", "0", "0.0"):
                    ent.delete(0, tk.END)
                    ent.insert(0, default)
                if se_labels is not None:
                    se_labels[key].config(text="± —", foreground=SUBTLE)
        else:
            for key in ("Lwire", "Theta"):
                ent = entries[key]
                ent.configure(state="normal")
                ent.delete(0, tk.END)
                ent.insert(0, "0")
                ent.configure(state="readonly")
                if se_labels is not None:
                    se_labels[key].config(text="fixed at 0", foreground=SUBTLE)

    def toggle_inductance_ui(self):
        # The inductance entries follow both the fit-enable and the inductance
        # toggles: visible+editable only when both are on, otherwise locked.
        parent_on = self.do_fit_var.get()
        inductance_on = parent_on and self.fit_inductance_var.get()
        self._apply_inductance_state(self.entries, inductance_on, self.se_labels)
        # Sim grid follows the inductance toggle alone — fit-enable does not gate sim.
        self._apply_inductance_state(
            self.sim_entries, self.fit_inductance_var.get(),
        )

    def _apply_model_state(self, entries, param_labels, faradaic_visible,
                           faradaic_editable, se_labels=None):
        widgets = [param_labels["Rk"], entries["Rk"]]
        if se_labels is not None:
            widgets.append(se_labels["Rk"])
        if faradaic_visible:
            for widget in widgets:
                widget.grid()
        else:
            for widget in widgets:
                widget.grid_remove()

        ent = entries["Rk"]
        if faradaic_editable:
            ent.configure(state="normal")
            current = ent.get().strip().lower()
            if current in ("", "0", "0.0", "inf", "infinity"):
                ent.delete(0, tk.END)
                ent.insert(0, "1.0")
            if se_labels is not None:
                se_labels["Rk"].config(text="± —", foreground=SUBTLE)
        else:
            ent.configure(state="normal")
            if ent.get().strip() == "":
                ent.insert(0, "1.0")
            ent.configure(state="readonly")
            if se_labels is not None:
                se_labels["Rk"].config(text="unused", foreground=SUBTLE)

    def toggle_model_ui(self):
        parent_on = self.do_fit_var.get()
        faradaic = self._is_faradaic_model()
        faradaic_on = parent_on and faradaic
        self._apply_model_state(
            self.entries, self.param_labels,
            faradaic_visible=faradaic, faradaic_editable=faradaic_on,
            se_labels=self.se_labels,
        )
        # Sim grid: visible whenever the model is Faradaic, editable too — sim
        # does not require fitting to be enabled.
        self._apply_model_state(
            self.sim_entries, self.sim_param_labels,
            faradaic_visible=faradaic, faradaic_editable=faradaic,
        )

    def _on_model_var_changed(self, *_):
        if self.lbl_sim_model is not None:
            self.lbl_sim_model.config(text=self.model_var.get())
        if self.sim_entries:
            faradaic = self._is_faradaic_model()
            self._apply_model_state(
                self.sim_entries, self.sim_param_labels,
                faradaic_visible=faradaic, faradaic_editable=faradaic,
            )

    @staticmethod
    def _set_entry_value(entry, text):
        prior = entry.cget("state")
        entry.configure(state="normal")
        entry.delete(0, tk.END)
        entry.insert(0, text)
        if prior != "normal":
            entry.configure(state=prior)

    def sync_sim_from_fit(self, silent=False):
        if not self.last_fit_data:
            if silent:
                return
            messagebox.showinfo("Sync", "No fit results yet — run a fit first.")
            return

        for key, value in zip(self.last_fit_data["keys"], self.last_fit_data["params"]):
            if key not in self.sim_entries:
                continue
            text = f"{value:.3e}" if key == "Lwire" else f"{value:.6f}"
            self._set_entry_value(self.sim_entries[key], text)

        f_fit = self.last_fit_data.get("f_fit")
        if f_fit is not None and len(f_fit) > 0:
            self._set_entry_value(self.sim_freq_entries["fmin"], f"{float(np.min(f_fit)):.4g}")
            self._set_entry_value(self.sim_freq_entries["fmax"], f"{float(np.max(f_fit)):.4g}")
            self._set_entry_value(self.sim_freq_entries["points"], str(int(len(f_fit))))

        if not silent and self.lbl_sim_status is not None:
            self.lbl_sim_status.config(
                text="Synced from latest fit.", foreground=ACCENT,
            )

    def _raw_averaged_spectrum(self):
        if not self.raw_data_list:
            return None, None
        rows = []
        for data in self.raw_data_list:
            rows.append(pd.DataFrame({
                "f": np.asarray(data["frequency"], dtype=float),
                "zr": np.asarray(data["z_real"], dtype=float),
                "zi": np.asarray(data["z_imag"], dtype=float),
            }))
        df = pd.concat(rows, ignore_index=True)
        grouped = df.groupby("f", sort=True)
        f = np.asarray(list(grouped.groups.keys()), dtype=float)
        zr = grouped["zr"].mean().values
        zi = grouped["zi"].mean().values
        return f, zr + 1j * zi

    def run_simulation(self):
        try:
            params = [float(self.sim_entries[k].get())
                      for k in self.logic.CANONICAL_KEYS]
        except ValueError:
            messagebox.showerror(
                "Invalid input", "Simulation parameters must be numeric.",
            )
            return

        try:
            fmin = float(self.sim_freq_entries["fmin"].get())
            fmax = float(self.sim_freq_entries["fmax"].get())
            n_pts = int(float(self.sim_freq_entries["points"].get()))
        except ValueError:
            messagebox.showerror(
                "Invalid input",
                "Frequency range entries must be numeric (points must be an integer).",
            )
            return
        if not (np.isfinite(fmin) and np.isfinite(fmax)) or fmin <= 0 or fmax <= 0:
            messagebox.showerror(
                "Invalid input", "f_min and f_max must be positive finite numbers.",
            )
            return
        if fmin >= fmax:
            messagebox.showerror("Invalid input", "f_min must be less than f_max.")
            return
        if n_pts < 2:
            messagebox.showerror("Invalid input", "Points must be at least 2.")
            return
        f = np.logspace(np.log10(fmin), np.log10(fmax), n_pts)

        # Experimental overlay: aggregate the raw measurements (no KK scope, no
        # outlier filter) so widening the sim window can reveal points that the
        # preprocessing step might have dropped. Falls back to processed_f / the
        # fit cache only if no raw data is loaded.
        f_src, z_src = self._raw_averaged_spectrum()
        if f_src is None and self.processed_f is not None:
            f_src = np.asarray(self.processed_f, dtype=float)
            z_src = self.processed_zr + 1j * self.processed_zi
        elif f_src is None and self.last_fit_data and self.last_fit_data.get("f_fit") is not None:
            f_src = np.asarray(self.last_fit_data["f_fit"], dtype=float)
            z_src = np.asarray(self.last_fit_data["z_exp"], dtype=complex)
        if f_src is not None:
            mask = (f_src >= fmin) & (f_src <= fmax)
            if mask.any():
                f_exp = f_src[mask]
                z_exp = z_src[mask]
            else:
                f_exp = None
                z_exp = None
        else:
            f_exp = None
            z_exp = None

        try:
            Z_sim = self.logic.evaluate_model(self.model_var.get(), params, f)
        except Exception as exc:
            messagebox.showerror("Simulation error", str(exc))
            return

        for ax in self.ax_sim.flat:
            ax.clear()
            style_axes(ax)

        ax_ny, ax_bode, ax_re, ax_im = (
            self.ax_sim[0, 0], self.ax_sim[0, 1],
            self.ax_sim[1, 0], self.ax_sim[1, 1],
        )

        if z_exp is not None:
            ax_ny.plot(
                np.real(z_exp), -np.imag(z_exp),
                marker="o", linestyle="none", markersize=4,
                markerfacecolor="white", markeredgecolor=COLOR_DATA,
                markeredgewidth=1.2, label="Data",
            )
        ax_ny.plot(
            np.real(Z_sim), -np.imag(Z_sim),
            color=COLOR_FIT, linewidth=1.8, label="Simulation",
        )
        ax_ny.set_aspect("equal", adjustable="datalim")
        ax_ny.set_xlabel(r"Z′  (Ω·cm²)")
        ax_ny.set_ylabel(r"−Z″  (Ω·cm²)")
        ax_ny.set_title("Nyquist", loc="left")
        leg = ax_ny.legend(loc="upper left", frameon=False)
        for t in leg.get_texts():
            t.set_color(INK)

        if z_exp is not None:
            ax_bode.loglog(
                f_exp, np.abs(z_exp), marker="o", linestyle="none",
                markersize=4, markerfacecolor="white",
                markeredgecolor=COLOR_DATA, markeredgewidth=1.2, label="Data",
            )
        ax_bode.loglog(f, np.abs(Z_sim), color=COLOR_FIT, linewidth=1.6, label="Simulation")
        ax_bode.set_xlabel(r"Frequency  (Hz)")
        ax_bode.set_ylabel(r"|Z|  (Ω·cm²)")
        ax_bode.set_title("Bode  ·  |Z|", loc="left")

        if z_exp is not None:
            ax_re.semilogx(
                f_exp, np.real(z_exp), marker="o", linestyle="none",
                markersize=4, markerfacecolor="white",
                markeredgecolor=COLOR_DATA, markeredgewidth=1.2,
            )
        ax_re.semilogx(f, np.real(Z_sim), color=COLOR_FIT, linewidth=1.6)
        ax_re.set_xlabel(r"Frequency  (Hz)")
        ax_re.set_ylabel(r"Re(Z)  (Ω·cm²)")
        ax_re.set_title("Re(Z)", loc="left")

        if z_exp is not None:
            ax_im.semilogx(
                f_exp, -np.imag(z_exp), marker="o", linestyle="none",
                markersize=4, markerfacecolor="white",
                markeredgecolor=COLOR_DATA, markeredgewidth=1.2,
            )
        ax_im.semilogx(f, -np.imag(Z_sim), color=COLOR_FIT, linewidth=1.6)
        ax_im.set_xlabel(r"Frequency  (Hz)")
        ax_im.set_ylabel(r"−Im(Z)  (Ω·cm²)")
        ax_im.set_title("−Im(Z)", loc="left")

        self.fig_sim.tight_layout()
        self.canvas_sim.draw_idle()

        if self.lbl_sim_status is not None:
            self.lbl_sim_status.config(
                text=(f"Simulated {len(f)} points · "
                      f"{f.min():.3g}–{f.max():.3g} Hz"),
                foreground=ACCENT,
            )

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
        self.last_fit_data = None
        self.lbl_status.config(text="")

        for ax in self.ax_diag:
            ax.clear()
            style_axes(ax)
        for ax in self.ax_fit.flat:
            ax.clear()
            style_axes(ax)
        self.canvas_diag.draw_idle()
        self.canvas_fit.draw_idle()

        if hasattr(self, "ax_sim"):
            for ax in self.ax_sim.flat:
                ax.clear()
                style_axes(ax)
            self.canvas_sim.draw_idle()
            if self.lbl_sim_status is not None:
                self.lbl_sim_status.config(text="")

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------
    def _parse_kk_frequency_range(self):
        def parse_optional(entry, name):
            text = entry.get().strip()
            if not text:
                return None
            try:
                value = float(text)
            except ValueError:
                raise ValueError(f"{name} must be numeric or blank.") from None
            if not np.isfinite(value):
                raise ValueError(f"{name} must be a finite number.")
            return value

        fmax = parse_optional(self.ent_fmax, "f_max")
        fmin = parse_optional(self.ent_fmin, "f_min")

        if fmin is not None and fmax is not None and fmin > fmax:
            raise ValueError("f_min must be less than or equal to f_max.")

        if fmin is None and fmax is None:
            return None

        return fmin, fmax

    def _update_frequency_entries_after_preprocess(self, suggested_min, suggested_max):
        # Always narrow the fit window to the KK-valid sub-range, whether or not
        # the user provided a manual scope. The manual entries are interpreted as
        # the KK *scope*, not the fit window.
        self.ent_fmax.delete(0, tk.END)
        self.ent_fmax.insert(0, f"{suggested_max:.2f}")

        self.ent_fmin.delete(0, tk.END)
        self.ent_fmin.insert(0, f"{suggested_min:.2f}")

    def run_preprocessing(self):
        if not self.raw_data_list:
            messagebox.showwarning("Warning", "No files loaded. Please add files first.")
            return

        freq_range = None
        if self.limit_kk_range_var.get():
            try:
                freq_range = self._parse_kk_frequency_range()
            except ValueError as e:
                messagebox.showerror("Invalid Frequency Range", str(e))
                return

        try:
            df_all, df_avg, (fmin, fmax), rsd, warnings = self.logic.process_spectra(
                self.raw_data_list,
                freq_range=freq_range,
            )
        except ValueError as e:
            messagebox.showerror("Preprocessing Error", str(e))
            return

        self.df_all_diagnostics = df_all
        self.df_avg = df_avg

        self.processed_f = df_avg["Freq(Hz)"].values
        self.processed_zr = df_avg["Z'(Ohm.cm²)"].values
        self.processed_zi = df_avg["Z''(Ohm.cm²)"].values

        self._update_frequency_entries_after_preprocess(fmin, fmax)

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

        processed_count = df_all["scan"].nunique()
        if processed_count >= 3:
            status_prefix = f"{processed_count} files averaged."
            status_suffix = f" RSD: {rsd:.2%}"
        else:
            status_prefix = f"{processed_count} of {len(self.raw_data_list)} file(s) processed."
            status_suffix = ""
        if freq_range is not None:
            req_min, req_max = freq_range
            if req_min is not None and req_max is not None:
                scope_text = f"{req_min:.2f}–{req_max:.2f} Hz"
            elif req_min is not None:
                scope_text = f"≥{req_min:.2f} Hz"
            else:
                scope_text = f"≤{req_max:.2f} Hz"
            status_suffix += f" KK scope: {scope_text}."
        if warnings:
            status_suffix += f" {len(warnings)} warning(s)."
        self.lbl_status.config(text=f"{status_prefix}{status_suffix}")

        if warnings:
            warning_text = "\n".join(warnings[:8])
            if len(warnings) > 8:
                warning_text += f"\n...and {len(warnings) - 8} more warning(s)."
            messagebox.showwarning("KK Preprocessing Warnings", warning_text)

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def _parse_fit_max_nfev(self):
        text = self.fit_max_nfev_var.get().strip()
        if not text:
            return self.logic.fit_max_nfev
        try:
            value = int(text)
        except ValueError:
            raise ValueError("Max evaluations must be a positive integer.") from None
        if value <= 0:
            raise ValueError("Max evaluations must be a positive integer.")
        return value

    def _parse_fit_n_restarts(self):
        text = self.fit_n_restarts_var.get().strip()
        if not text:
            return self.logic.fit_n_restarts
        try:
            value = int(text)
        except ValueError:
            raise ValueError("Restarts must be a positive integer.") from None
        if value <= 0:
            raise ValueError("Restarts must be a positive integer.")
        return value

    def run_fitting(self):
        if self.processed_f is None:
            messagebox.showerror("Error", "Run preprocessing first.")
            return

        try:
            fmax, fmin = float(self.ent_fmax.get()), float(self.ent_fmin.get())
            max_nfev = self._parse_fit_max_nfev()
            n_restarts = self._parse_fit_n_restarts()
        except ValueError as e:
            messagebox.showerror("Invalid Fit Input", str(e))
            return

        mask = (self.processed_f >= fmin) & (self.processed_f <= fmax)
        f_fit = self.processed_f[mask]
        z_exp = self.processed_zr[mask] + 1j * self.processed_zi[mask]

        fit_inductance = self.fit_inductance_var.get()
        fit_faradaic = self._is_faradaic_model()

        # Init vector matches the optimizer's working layout.
        init_keys = self._active_fit_keys(fit_inductance, fit_faradaic)
        try:
            init_p = [float(self.entries[k].get()) for k in init_keys]
        except ValueError:
            messagebox.showerror("Invalid Fit Input", "Fit parameters must be numeric.")
            return

        results, error_msg = self.logic.fit_impedance(
            self.model_var.get(), init_p, f_fit, z_exp,
            max_nfev=max_nfev, n_restarts=n_restarts,
            fit_inductance=fit_inductance,
        )

        if error_msg:
            messagebox.showerror("Fitting Error", error_msg)
            return

        # Logic always returns canonical 7-vectors:
        # [L_wire, HFR, R_CL, Q_dl, Phi, Theta, R_k].
        params, se, residuals, info = results
        canonical_keys = list(self.logic.CANONICAL_KEYS)
        # Only the params that were actually fit get value/SE labels updated.
        editable_keys = init_keys

        high_se_threshold = self.logic.fit_high_uncertainty_pct
        flagged = []
        for i, k in enumerate(canonical_keys):
            if k not in editable_keys:
                # Stays at its fixed-at-zero presentation; skip.
                continue
            self.entries[k].delete(0, tk.END)
            value_fmt = f"{params[i]:.3e}" if k == "Lwire" else f"{params[i]:.6f}"
            self.entries[k].insert(0, value_fmt)
            pct_error = (abs(se[i]) / abs(params[i]) * 100) if params[i] != 0 else float("inf")
            label_text = f"± {se[i]:.3g} ({pct_error:.1f}%)"
            if not np.isfinite(pct_error) or pct_error > high_se_threshold:
                self.se_labels[k].config(
                    text=label_text + "  high",
                    foreground=COLOR_HIGH_SE,
                )
                flagged.append(f"{k} ±{pct_error:.0f}%" if np.isfinite(pct_error) else f"{k} ±inf")
            else:
                self.se_labels[k].config(text=label_text, foreground=SUBTLE)

        if info["ranking_basis"] == "hfr_rcl_se":
            basis_label = "HFR/R_CL accuracy"
        else:
            basis_label = "lowest SSR (fallback)"
        score_label = "HFR/R_CL"
        fit_msg = (
            f"Best of {info['n_restarts']} restarts (#{info['best_attempt']}, by {basis_label})  ·  "
            f"max rel. SE on {score_label} = {info['score_hfr_rcl_pct']:.1f}%  ·  "
            f"SSR = {info['ssr']:.3g}"
        )
        if info["n_failed"]:
            fit_msg += f"  ·  {info['n_failed']} restart(s) failed"
        if flagged:
            fit_msg += f"  ·  high uncertainty: {', '.join(flagged)}"
        self.lbl_fit_status.config(
            text=fit_msg,
            foreground=COLOR_HIGH_SE if flagged else ACCENT,
        )

        Z_model = self.logic.evaluate_model(self.model_var.get(), list(params), f_fit)
        export_keys, export_params, export_se = self._export_parameter_rows(params, se, fit_faradaic)

        self.last_fit_data = {
            "f_fit": f_fit,
            "z_exp": z_exp,
            "Z_model": Z_model,
            "keys": export_keys,
            "params": list(np.asarray(export_params, dtype=float)),
            "se": list(np.asarray(export_se, dtype=float)),
        }
        self.sync_sim_from_fit(silent=True)

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

        # Inline parameter readout in the Nyquist axes corner.
        # params is canonical 7-vector: [L_wire, HFR, R_CL, Q_dl, Phi, Theta, R_k].
        param_lines = [
            f"HFR  = {params[1]:.4f} Ω·cm²",
            f"R_CL = {params[2]:.4f} Ω·cm²",
            f"Q_dl = {params[3]:.3g} F",
            f"φ    = {params[4]:.3f}",
        ]
        if self.model_var.get() == "Transmission Line" and not fit_faradaic:
            param_lines.append(f"R_CL/3 = {params[2] / 3.0:.4f} Ω·cm²")
        if fit_faradaic:
            param_lines.append(f"R_k  = {params[6]:.4f} Ω·cm²")
        if fit_inductance:
            param_lines.append(f"L_wire = {params[0]:.3e} H·cm²")
            param_lines.append(f"Θ    = {params[5]:.3f}")
        param_text = "\n".join(param_lines)
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
