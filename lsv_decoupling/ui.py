import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
import pandas as pd

from core.data_io import read_lsv_data
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

        self.reference_options = {
            "RHE": 0.0,
            "Ag/AgCl (sat)": 0.197,
            "SCE (sat)": 0.242,
            "HgO (sat)": 0.098
        }

        self._init_style()
        self._build_layout()

    def _init_style(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(".", font=("Arial", 10))
        style.configure("TLabelframe.Label", font=("Arial", 10, "bold"))
        style.configure("TButton", font=("Arial", 10))

    def _build_layout(self):
        # --- Left Panel: Controls ---
        left_panel = ttk.Frame(self, width=350)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        # File Loading
        btn_frame = ttk.Frame(left_panel)
        btn_frame.pack(fill=tk.X, pady=5)
        ttk.Button(btn_frame, text="Load LSV File", command=self.load_file).pack(fill=tk.X)
        self.lbl_status = ttk.Label(left_panel, text="No file selected.")
        self.lbl_status.pack(anchor="w")

        # Ref & Electrolyte
        ref_frame = ttk.LabelFrame(left_panel, text="Electrolyte & Reference")
        ref_frame.pack(fill=tk.X, pady=5)

        ttk.Label(ref_frame, text="Ref:").grid(row=0, column=0, sticky="e", padx=2, pady=2)
        self.ref_var = tk.StringVar(value="RHE")
        ttk.Combobox(ref_frame, textvariable=self.ref_var, values=list(self.reference_options.keys()), state="readonly",
                     width=12).grid(row=0, column=1, padx=2, pady=2)

        ttk.Label(ref_frame, text="pH:").grid(row=1, column=0, sticky="e", padx=2, pady=2)
        self.pH_entry = ttk.Entry(ref_frame, width=14)
        self.pH_entry.insert(0, "0")
        self.pH_entry.grid(row=1, column=1, padx=2, pady=2)

        ttk.Label(ref_frame, text="Temp(K):").grid(row=2, column=0, sticky="e", padx=2, pady=2)
        self.temp_entry = ttk.Entry(ref_frame, width=14)
        self.temp_entry.insert(0, "378")
        self.temp_entry.grid(row=2, column=1, padx=2, pady=2)

        # Resistances
        res_frame = ttk.LabelFrame(left_panel, text="Resistances")
        res_frame.pack(fill=tk.X, pady=5)

        ttk.Label(res_frame, text="R_CL (Ω·cm²):").grid(row=0, column=0, sticky="e", padx=2, pady=2)
        self.rcl_entry = ttk.Entry(res_frame, width=12)
        self.rcl_entry.insert(0, "0")
        self.rcl_entry.grid(row=0, column=1, padx=2, pady=2)

        ttk.Label(res_frame, text="HFR (Ω·cm²):").grid(row=1, column=0, sticky="e", padx=2, pady=2)
        self.hfr_entry = ttk.Entry(res_frame, width=12)
        self.hfr_entry.insert(0, "0.01")
        self.hfr_entry.grid(row=1, column=1, padx=2, pady=2)

        # Tafel Window
        tafel_frame = ttk.LabelFrame(left_panel, text="Tafel Fit Range (A/cm²)")
        tafel_frame.pack(fill=tk.X, pady=5)

        ttk.Label(tafel_frame, text="Lower:").grid(row=0, column=0, sticky="e", padx=2, pady=2)
        self.tafel_lower_entry = ttk.Entry(tafel_frame, width=10)
        self.tafel_lower_entry.insert(0, "0.02")
        self.tafel_lower_entry.grid(row=0, column=1, padx=2, pady=2)

        ttk.Label(tafel_frame, text="Upper:").grid(row=0, column=2, sticky="e", padx=2, pady=2)
        self.tafel_upper_entry = ttk.Entry(tafel_frame, width=10)
        self.tafel_upper_entry.insert(0, "0.8")
        self.tafel_upper_entry.grid(row=0, column=3, padx=2, pady=2)

        # NEW: Target Breakdown Setting
        bar_frame = ttk.LabelFrame(left_panel, text="Column Plot Target")
        bar_frame.pack(fill=tk.X, pady=5)

        ttk.Label(bar_frame, text="Current (A/cm²):").grid(row=0, column=0, sticky="e", padx=2, pady=2)
        self.target_i_entry = ttk.Entry(bar_frame, width=10)
        self.target_i_entry.insert(0, "1.0")
        self.target_i_entry.grid(row=0, column=1, padx=2, pady=2)
        ttk.Button(bar_frame, text="Update Plot", command=self.update_bar_plot).grid(row=0, column=2, padx=5, pady=2)

        # Actions
        action_frame = ttk.Frame(left_panel)
        action_frame.pack(fill=tk.X, pady=10)
        ttk.Button(action_frame, text="Fit & Plot", command=self.perform_fit).pack(fill=tk.X, pady=2)
        ttk.Button(action_frame, text="Toggle X-Axis Scale", command=self.toggle_x_scale).pack(fill=tk.X, pady=2)
        ttk.Button(action_frame, text="Export Decoupled Data", command=self.export_data).pack(fill=tk.X, pady=10)

        # --- Right Panel: Notebook ---
        right_panel = ttk.Frame(self)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.notebook = ttk.Notebook(right_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 1: Plot
        self.tab_plot = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_plot, text="Decoupled Plot")

        self.fig = plt.Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.grid(True, alpha=0.3)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.tab_plot)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar_frame = ttk.Frame(self.tab_plot)
        toolbar_frame.pack(fill=tk.X, side=tk.BOTTOM)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()

        # Tab 2: NEW Column Plot
        self.tab_bar = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_bar, text="Overpotential Column Plot")

        self.fig_bar = plt.Figure(figsize=(8, 6), dpi=100)
        self.ax_bar = self.fig_bar.add_subplot(111)
        self.canvas_bar = FigureCanvasTkAgg(self.fig_bar, master=self.tab_bar)
        self.canvas_bar.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar_frame_bar = ttk.Frame(self.tab_bar)
        toolbar_frame_bar.pack(fill=tk.X, side=tk.BOTTOM)
        self.toolbar_bar = NavigationToolbar2Tk(self.canvas_bar, toolbar_frame_bar)
        self.toolbar_bar.update()

        # Tab 3: Diagnostics
        self.tab_diag = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_diag, text="Fit Diagnostics")
        self.diag_text = tk.Text(self.tab_diag, font=("Consolas", 10), wrap=tk.WORD)
        self.diag_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

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

        result, warn_err = iterative_tafel_fit(self.V_data, self.i_data, self.E_rev, HFR, R_CL, i_lower, i_upper)

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
        self.Eta_res = np.maximum((self.V_data - self.E_rev) - (self.eta_kin + self.Eta_ohm + self.Eta_RCL), 0)

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
        self.update_bar_plot(auto_focus=False)  # Generate bar chart silently in background
        self.notebook.select(self.tab_plot)

    def _plot_curves(self):
        colors = ["#111827", "#2563EB", "#059669", "#F59E0B", "#DC2626", "#6B7280"]
        self.ax.clear()
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xscale("log" if self.x_log_scale else "linear")

        # Added alpha=0.8 for transparency to see overlapping limits
        self.ax.plot(self.i_data, self.y1, color=colors[0], linestyle="--", linewidth=2, label="E_rev", alpha=0.8)
        self.ax.plot(self.i_data, self.y2, color=colors[1], linewidth=2, label="E_rev + η_kin", alpha=0.8)
        self.ax.plot(self.i_data, self.y3, color=colors[2], linewidth=2, label="+ η_ohm", alpha=0.8)
        self.ax.plot(self.i_data, self.y4, color=colors[3], linewidth=2, label="+ η_RCL", alpha=0.8)
        self.ax.plot(self.i_data, self.y5, color=colors[4], linewidth=2, label="+ η_res", alpha=0.8)
        self.ax.plot(self.i_data, self.y6, color=colors[5], marker="o", linestyle="none", markersize=4,
                     label="Original LSV (RHE)", alpha=0.8)

        self.ax.set_xlabel("Current Density (A/cm²)")
        self.ax.set_ylabel("Potential (V)")
        self.ax.set_title("LSV Overpotential Analysis")
        self.ax.legend(fontsize=9, ncol=2)

        self.fig.tight_layout()
        self.canvas.draw()

    def update_bar_plot(self, auto_focus=True):
        if self.fit_results is None:
            if auto_focus: messagebox.showerror("Error", "Perform a fit first.")
            return

        try:
            target_i = float(self.target_i_entry.get())
            HFR = float(self.hfr_entry.get())
            R_CL = float(self.rcl_entry.get())
        except ValueError:
            messagebox.showerror("Input Error", "Please ensure Target Current and Resistances are valid numbers.")
            return

        comps = get_overpotential_at_current(target_i, self.i_data, self.V_data, self.E_rev, HFR, R_CL,
                                             self.fit_results)

        self.ax_bar.clear()
        labels = ["E_rev", "η_kin", "η_ohm", "η_RCL", "η_res"]
        values = [comps["E_rev"], comps["eta_kin"], comps["eta_ohm"], comps["eta_rcl"], comps["eta_res"]]
        colors = ["#111827", "#2563EB", "#059669", "#F59E0B", "#DC2626"]

        bottom = 0
        for label, val, col in zip(labels, values, colors):
            # width is thin since there's only one column
            self.ax_bar.bar([f"{target_i} A/cm²"], [val], bottom=bottom, color=col, label=f"{label}: {val:.4f} V",
                            alpha=0.85, width=0.3)
            bottom += val

        self.ax_bar.set_ylabel("Potential (V)")
        self.ax_bar.set_title(f"Overpotential Breakdown at {target_i} A/cm²")

        # Place legend nicely outside the column
        self.ax_bar.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        self.fig_bar.tight_layout()
        self.canvas_bar.draw()

        if auto_focus:
            self.notebook.select(self.tab_bar)

    def toggle_x_scale(self):
        if self.i_data is None: return
        self.x_log_scale = not self.x_log_scale
        self._plot_curves()

    def export_data(self):
        if self.i_data is None: return

        df_curves = pd.DataFrame({
            "Current Density (A/cm²)": self.i_data,
            "E_rev": self.y1,
            "E_rev + η_kin": self.y2,
            "+ η_ohm": self.y3,
            "+ η_RCL": self.y4,
            "+ η_res": self.y5,
            "Original LSV (RHE)": self.y6
        })

        df_comp = pd.DataFrame({
            "Current Density (A/cm²)": self.i_data,
            "η_kin (V)": self.eta_kin,
            "η_ohm (V)": self.Eta_ohm,
            "η_RCL (V)": self.Eta_RCL,
            "η_res (V)": self.Eta_res
        })

        df_info = pd.DataFrame({
            "Parameter": ["Tafel slope (V/dec)", "Exchange current density (A/cm²)", "Intercept (V)", "R²", "N points"],
            "Value": [self.fit_results["b_kin"], self.fit_results["i0"], self.fit_results["intercept"],
                      self.fit_results["r_squared"], self.fit_results["n_points"]]
        }) if self.fit_results else pd.DataFrame()

        file_path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel", "*.xlsx")])
        if not file_path: return

        try:
            with pd.ExcelWriter(file_path) as writer:
                df_curves.to_excel(writer, sheet_name="Curves (stacked)", index=False)
                df_comp.to_excel(writer, sheet_name="Overpotential Components", index=False)
                df_info.to_excel(writer, sheet_name="Fitting Info", index=False)
            messagebox.showinfo("Export", "Data successfully exported.")
        except Exception as e:
            messagebox.showerror("Export Error", str(e))