import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
import pandas as pd

from core.data_io import read_eis_data
from osif.logic import EisLogic


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
        left_panel = ttk.Frame(self, width=380)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        # File Management Frame
        file_frame = ttk.LabelFrame(left_panel, text="Loaded EIS Files")
        file_frame.pack(fill=tk.X, pady=5)

        self.listbox_files = tk.Listbox(file_frame, height=5, selectmode=tk.EXTENDED)
        self.listbox_files.pack(fill=tk.X, padx=5, pady=5)

        btn_box = ttk.Frame(file_frame)
        btn_box.pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(btn_box, text="Add", command=self.add_files).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        ttk.Button(btn_box, text="Remove", command=self.remove_files).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        ttk.Button(btn_box, text="Clear All", command=self.clear_files).pack(side=tk.LEFT, expand=True, fill=tk.X,
                                                                             padx=1)

        # 1. Validation & HFR Frame
        val_frame = ttk.LabelFrame(left_panel, text="1. Validation & Averaging")
        val_frame.pack(fill=tk.X, pady=10)

        ttk.Button(val_frame, text="Run Preprocess & KK Test", command=self.run_preprocessing).pack(fill=tk.X, padx=5,
                                                                                                    pady=5)

        freq_f = ttk.Frame(val_frame)
        freq_f.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(freq_f, text="Max (Hz):").grid(row=0, column=0, sticky="e")
        self.ent_fmax = ttk.Entry(freq_f, width=10)
        self.ent_fmax.grid(row=0, column=1)
        ttk.Label(freq_f, text="Min (Hz):").grid(row=0, column=2, sticky="e")
        self.ent_fmin = ttk.Entry(freq_f, width=10)
        self.ent_fmin.grid(row=0, column=3)

        hfr_f = ttk.Frame(val_frame)
        hfr_f.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(hfr_f, text="HFR [Ω·cm²]:").grid(row=0, column=0, sticky="e", padx=2)
        self.entries["HFR"] = ttk.Entry(hfr_f, width=12)
        self.entries["HFR"].insert(0, "0.2")
        self.entries["HFR"].grid(row=0, column=1, padx=2)
        self.se_labels["HFR"] = ttk.Label(hfr_f, text="± --")
        self.se_labels["HFR"].grid(row=0, column=2, sticky="w", padx=5)

        # Status output for RSD
        self.lbl_status = ttk.Label(val_frame, text="", foreground="blue")
        self.lbl_status.pack(anchor="w", padx=5, pady=2)

        # 2. Impedance Fitting Frame
        self.fit_container = ttk.LabelFrame(left_panel, text="2. Impedance Fitting")
        self.fit_container.pack(fill=tk.X, pady=5)

        self.do_fit_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.fit_container, text="Enable Impedance Fitting (Non-Faradaic)",
                        variable=self.do_fit_var, command=self.toggle_fit_ui).pack(anchor="w", padx=5, pady=5)

        self.param_frame = ttk.Frame(self.fit_container)
        self.param_frame.pack(fill=tk.X, padx=5)

        ttk.Label(self.param_frame, text="Parameter", font=("Arial", 9, "italic")).grid(row=0, column=0, pady=2)
        ttk.Label(self.param_frame, text="Value", font=("Arial", 9, "italic")).grid(row=0, column=1, pady=2)
        ttk.Label(self.param_frame, text="Std. Error (%)", font=("Arial", 9, "italic")).grid(row=0, column=2, pady=2)

        labels = ["Lwire [H·cm²]", "Rcl [Ω·cm²]", "Qdl [F]", "Phi [-]", "Theta [-]"]
        defaults = ["0", "0.2", "0.1", "0.9", "0"]

        for i, (lbl, df) in enumerate(zip(labels, defaults)):
            key = lbl.split()[0]
            ttk.Label(self.param_frame, text=lbl).grid(row=i + 1, column=0, sticky="e", padx=5, pady=2)

            ent = ttk.Entry(self.param_frame, width=12)
            ent.insert(0, df)
            ent.grid(row=i + 1, column=1, padx=5, pady=2)
            self.entries[key] = ent

            se_lbl = ttk.Label(self.param_frame, text="± --")
            se_lbl.grid(row=i + 1, column=2, sticky="w", padx=5)
            self.se_labels[key] = se_lbl

            if key in ["Lwire", "Theta"]:
                ent.config(state="readonly")
                se_lbl.config(text="± 0 (0%)")

        self.model_var = tk.StringVar(value="Transmission Line")
        models = ["Transmission Line", "1-D Linear Diffusion", "1-D Spherical Diffusion"]
        self.cmb_model = ttk.Combobox(self.fit_container, textvariable=self.model_var, values=models, state="readonly")
        self.cmb_model.pack(fill=tk.X, padx=5, pady=5)

        self.btn_fit = ttk.Button(self.fit_container, text="Fit Model", command=self.run_fitting)
        self.btn_fit.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(left_panel, text="Export Results", command=self.export_results).pack(fill=tk.X, pady=10)

        # --- Right Panel: Plotting Tabs ---
        right_panel = ttk.Frame(self)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.notebook = ttk.Notebook(right_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 1
        self.tab_diag = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_diag, text="KK Diagnostics & Previews")
        self.fig_diag, self.ax_diag = plt.subplots(1, 2, figsize=(10, 4))
        self.canvas_diag = FigureCanvasTkAgg(self.fig_diag, master=self.tab_diag)
        self.canvas_diag.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        toolbar_frame_diag = ttk.Frame(self.tab_diag)
        toolbar_frame_diag.pack(fill=tk.X, side=tk.BOTTOM)
        self.toolbar_diag = NavigationToolbar2Tk(self.canvas_diag, toolbar_frame_diag)
        self.toolbar_diag.update()

        # Tab 2
        self.tab_fit = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_fit, text="Fitting Results")
        self.fig_fit, self.ax_fit = plt.subplots(2, 2, figsize=(10, 6))
        self.canvas_fit = FigureCanvasTkAgg(self.fig_fit, master=self.tab_fit)
        self.canvas_fit.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

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

    # --- File Management Functions ---
    def add_files(self):
        filepaths = filedialog.askopenfilenames(filetypes=[("EIS Files", "*.txt *.csv *.xlsx")])
        if not filepaths: return

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

        for ax in self.ax_diag: ax.clear()
        for ax in self.ax_fit.flat: ax.clear()
        self.canvas_diag.draw()
        self.canvas_fit.draw()

    # --- Math & Fitting ---
    def run_preprocessing(self):
        if not self.raw_data_list:
            messagebox.showwarning("Warning", "No files loaded. Please add files first.")
            return

        df_all, df_avg, (fmin, fmax), rsd = self.logic.process_spectra(self.raw_data_list)
        self.df_all_diagnostics = df_all
        self.df_avg = df_avg

        self.processed_f = df_avg['Freq(Hz)'].values
        self.processed_zr = df_avg["Z'(Ohm.cm²)"].values
        self.processed_zi = df_avg["Z''(Ohm.cm²)"].values

        self.ent_fmax.delete(0, tk.END)
        self.ent_fmax.insert(0, f"{fmax:.2f}")
        self.ent_fmin.delete(0, tk.END)
        self.ent_fmin.insert(0, f"{fmin:.2f}")

        est_hfr = np.min(self.processed_zr)
        self.entries["HFR"].delete(0, tk.END)
        self.entries["HFR"].insert(0, f"{est_hfr:.4f}")

        for k in self.se_labels:
            if k not in ["Lwire", "Theta"]:
                self.se_labels[k].config(text="± --")

        self.ax_diag[0].clear()
        self.ax_diag[1].clear()

        # Plot Raw Valid/Invalid Scans
        mask_valid = df_all['kk_err'] <= self.logic.kk_threshold
        self.ax_diag[0].scatter(df_all.loc[mask_valid, 'zr'], -df_all.loc[mask_valid, 'zi'], c='royalblue', alpha=0.5,
                                label='Valid')
        self.ax_diag[0].scatter(df_all.loc[~mask_valid, 'zr'], -df_all.loc[~mask_valid, 'zi'], c='crimson', marker='x',
                                label='Invalid')

        # Plot Final Averaged Curve
        self.ax_diag[0].plot(self.processed_zr, -self.processed_zi, color='black', linewidth=2.5, label='Final Avg')

        self.ax_diag[0].set_aspect('equal', adjustable='datalim')
        self.ax_diag[0].set_title("Nyquist KK Validation & Averaging")
        self.ax_diag[0].legend()

        self.ax_diag[1].scatter(df_all['f'], df_all['kk_err'] * 100, c='grey', alpha=0.4, s=10)
        self.ax_diag[1].axhline(self.logic.kk_threshold * 100, color='crimson', linestyle='--')
        self.ax_diag[1].set_xscale('log')
        self.ax_diag[1].set_yscale('log')
        self.ax_diag[1].set_title("Bode Error")

        self.fig_diag.tight_layout()
        self.canvas_diag.draw()
        self.notebook.select(self.tab_diag)

        # Notify the user of successful averaging
        if len(self.raw_data_list) >= 3:
            self.lbl_status.config(text=f"{len(self.raw_data_list)} files averaged. RSD: {rsd:.2%}")
        else:
            self.lbl_status.config(text=f"{len(self.raw_data_list)} file(s) processed.")

    def run_fitting(self):
        if self.processed_f is None:
            messagebox.showerror("Error", "Run Preprocessing first.")
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
            self.se_labels[k].config(text=f"± {se[i]:.4g} ({pct_error:.1f}%)")

        Z_model = self.logic.evaluate_model(self.model_var.get(), params, f_fit)

        self.last_fit_data = {
            'f_fit': f_fit,
            'z_exp': z_exp,
            'Z_model': Z_model,
            'keys': ["Lwire", "HFR", "Rcl", "Qdl", "Phi", "Theta"],
            'params': [0, params[0], params[1], params[2], params[3], 0],
            'se': [0, se[0], se[1], se[2], se[3], 0]
        }

        for ax in self.ax_fit.flat: ax.clear()

        self.ax_fit[0, 0].plot(np.real(z_exp), -np.imag(z_exp), 'bo', label='Data')

        fit_label = (f"Fit\n"
                     f"HFR: {params[0]:.4f}\n"
                     f"Rcl: {params[1]:.4f}\n"
                     f"Qdl: {params[2]:.4g}\n"
                     f"Phi: {params[3]:.4f}")
        self.ax_fit[0, 0].plot(np.real(Z_model), -np.imag(Z_model), 'r-', label=fit_label)

        self.ax_fit[0, 0].set_title("Nyquist")
        self.ax_fit[0, 0].legend(fontsize=8)
        self.ax_fit[0, 0].set_aspect('equal', adjustable='datalim')

        self.ax_fit[0, 1].loglog(f_fit, np.abs(z_exp), 'bo')
        self.ax_fit[0, 1].loglog(f_fit, np.abs(Z_model), 'r-')
        self.ax_fit[0, 1].set_title("Bode |Z|")

        self.ax_fit[1, 0].semilogx(f_fit, np.real(z_exp), 'bo')
        self.ax_fit[1, 0].semilogx(f_fit, np.real(Z_model), 'r-')
        self.ax_fit[1, 0].set_title("Real(Z)")

        self.ax_fit[1, 1].semilogx(f_fit, -np.imag(z_exp), 'bo')
        self.ax_fit[1, 1].semilogx(f_fit, -np.imag(Z_model), 'r-')
        self.ax_fit[1, 1].set_title("-Imag(Z)")

        self.fig_fit.tight_layout()
        self.canvas_fit.draw()
        self.notebook.select(self.tab_fit)

    def export_results(self):
        if self.df_all_diagnostics is None:
            messagebox.showerror("Export Error", "No data available to export. Run 'Preprocess & KK Test' first.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx")],
            title="Save OSIF Results As"
        )
        if not file_path:
            return

        try:
            with pd.ExcelWriter(file_path) as writer:
                self.df_all_diagnostics.to_excel(writer, sheet_name="KK Diagnostics & Raw", index=False)

                # New Export Sheet: Averaged Data (Matches EIS ave.py)
                if self.df_avg is not None:
                    self.df_avg.to_excel(writer, sheet_name="Averaged Data", index=False)

                if self.last_fit_data is not None and self.do_fit_var.get():
                    df_curve = pd.DataFrame({
                        "Frequency (Hz)": self.last_fit_data['f_fit'],
                        "Data Re(Z)": np.real(self.last_fit_data['z_exp']),
                        "Data Im(Z)": np.imag(self.last_fit_data['z_exp']),
                        "Fit Re(Z)": np.real(self.last_fit_data['Z_model']),
                        "Fit Im(Z)": np.imag(self.last_fit_data['Z_model']),
                        "Fit |Z|": np.abs(self.last_fit_data['Z_model'])
                    })
                    df_curve.to_excel(writer, sheet_name="Fitted Curve", index=False)

                    df_params = pd.DataFrame({
                        "Parameter": self.last_fit_data['keys'],
                        "Fitted Value": self.last_fit_data['params'],
                        "Standard Error": self.last_fit_data['se']
                    })
                    df_params.to_excel(writer, sheet_name="Fit Parameters", index=False)

            messagebox.showinfo("Export Successful", f"Results successfully exported to:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Export Error", f"An error occurred while exporting:\n{e}")