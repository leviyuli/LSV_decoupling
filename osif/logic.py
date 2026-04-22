import numpy as np
import pandas as pd
import scipy.optimize
import builtins
import impedance.validation
from impedance.validation import linKK
import sys
import io
from contextlib import redirect_stdout

# --- BUG FIX FOR impedance.py ---
_orig_eval = builtins.eval


def patched_eval(expr, globals_dict=None, locals_dict=None):
    if globals_dict is not None and isinstance(globals_dict, dict):
        globals_dict['np'] = np
    return _orig_eval(expr, globals_dict, locals_dict)


impedance.validation.eval = patched_eval


# --------------------------------

class EisLogic:
    def __init__(self):
        # Matched to EIS ave.py settings
        self.kk_threshold = 0.1
        self.outlier_z_threshold = 1.5

    # --- Preprocessing & KK ---
    def process_spectra(self, data_list):
        all_scans = []
        for i, data in enumerate(data_list):
            freq = data['frequency']
            z_complex = data['z_complex']

            try:
                # Mute the automatic print statements from linKK
                with redirect_stdout(io.StringIO()):
                    M, mu, Z_fit, res_real, res_imag = linKK(freq, z_complex, c=0.85)

                point_errors = np.sqrt(res_real ** 2 + res_imag ** 2)

                df_scan = pd.DataFrame({
                    'f': freq, 'zr': data['z_real'], 'zi': data['z_imag'],
                    'scan': i + 1, 'kk_err': point_errors
                })
                all_scans.append(df_scan)
            except Exception as e:
                print(f"KK Error on scan {i + 1}: {e}")

        df_all = pd.concat(all_scans, ignore_index=True)

        valid_mask = df_all['kk_err'] <= self.kk_threshold
        valid_freqs = df_all.loc[valid_mask, 'f']
        freq_max_suggested = valid_freqs.max() if not valid_freqs.empty else df_all['f'].max()
        freq_min_suggested = valid_freqs.min() if not valid_freqs.empty else df_all['f'].min()

        # Outlier rejection and Averaging (if >= 3 scans)
        if df_all['scan'].nunique() >= 3:
            def calc_z(x):
                std = x.std(ddof=1)
                if pd.isna(std) or std == 0:
                    return pd.Series(np.zeros(len(x)), index=x.index)
                return np.abs((x - x.mean()) / std)

            z_zr = df_all.groupby('f')['zr'].transform(calc_z)
            z_zi = df_all.groupby('f')['zi'].transform(calc_z)
            df_clean = df_all[(z_zr <= self.outlier_z_threshold) & (z_zi <= self.outlier_z_threshold)]
        else:
            df_clean = df_all.copy()

        grouped = df_clean.groupby('f', sort=False)
        f_avg = np.array(list(grouped.groups.keys()))
        zr_avg = grouped['zr'].mean().values
        zi_avg = grouped['zi'].mean().values

        # Calculate standard deviations for the exported dataset
        zr_std = grouped['zr'].std().fillna(0).values
        zi_std = grouped['zi'].std().fillna(0).values

        # Drift / RSD Calculation matched to EIS ave.py
        mag = np.sqrt(zr_avg ** 2 + zi_avg ** 2)
        std_total = np.sqrt(zr_std ** 2 + zi_std ** 2)
        rsd = np.nanmean(std_total / mag) if len(mag) > 0 else 0

        # Construct the final averaged dataframe
        df_avg = pd.DataFrame({
            'Freq(Hz)': f_avg,
            "Z'(Ohm.cm²)": zr_avg,
            "Z''(Ohm.cm²)": zi_avg,
            "Z'_std": zr_std,
            "Z''_std": zi_std
        })

        return df_all, df_avg, (freq_min_suggested, freq_max_suggested), rsd

    # --- Fitting Models ---
    def JPcoth(self, x):
        return (np.exp(x) + np.exp(-x)) / (np.exp(x) - np.exp(-x))

    def evaluate_model(self, model_name, params, freq):
        HFR, Rcl, Qdl, Phi = params
        omega = 1j * 2 * np.pi * freq

        if model_name == "Transmission Line":
            term = np.sqrt(Rcl / (Qdl * (omega ** Phi)))
            Z = HFR + term * self.JPcoth(np.sqrt(Rcl * Qdl * (omega ** Phi)))
        elif model_name == "1-D Linear Diffusion":
            term = Rcl * (Rcl * (Qdl * (omega ** Phi))) ** (-0.5)
            Z = HFR + term * self.JPcoth(np.sqrt(Rcl * Qdl * (omega ** Phi)))
        elif model_name == "1-D Spherical Diffusion":
            term = np.sqrt(Rcl * Qdl * (omega ** Phi))
            Z = HFR + Rcl / (term * self.JPcoth(term) - 1)
        return Z

    def fit_impedance(self, model_name, init_params, freq, z_exp):
        def cost_func(params):
            Z_model = self.evaluate_model(model_name, params, freq)
            diff = (np.real(Z_model) - np.real(z_exp)) ** 2 + (np.imag(Z_model) - np.imag(z_exp)) ** 2
            return np.sqrt(diff)

        hfr_init = init_params[0]
        lower_bounds = [0.9 * hfr_init, 0, 0, 0]
        upper_bounds = [1.1 * hfr_init, np.inf, np.inf, 1]

        try:
            res = scipy.optimize.least_squares(
                cost_func, init_params, bounds=(lower_bounds, upper_bounds),
                method='trf', xtol=1e-11, ftol=1e-11, gtol=1e-11
            )

            if not res.success:
                return None, f"Fitting algorithm failed: {res.message}"

            try:
                cov_matrix = np.linalg.inv(res.jac.T @ res.jac)
            except np.linalg.LinAlgError:
                cov_matrix = np.linalg.pinv(res.jac.T @ res.jac)

            dof = len(res.fun) - len(res.x)
            s2 = (res.fun.T @ res.fun) / dof if dof > 0 else 0
            se = np.sqrt(np.diag(cov_matrix * s2))

            return (res.x, se, cost_func(res.x)), None

        except Exception as e:
            return None, f"An error occurred during fitting: {str(e)}"