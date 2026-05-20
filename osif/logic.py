import builtins
import io
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
import scipy.optimize
import impedance.validation
from impedance.validation import linKK

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
        self.kk_threshold = 0.1
        self.outlier_z_threshold = 1.5
        self.min_kk_points = 3
        self.fit_max_nfev = 20000
        # Multi-start fitting: total optimizer launches (1 from user init + N-1 perturbed).
        # Restarts are ranked by parameter accuracy on HFR and R_CL, not by SSR
        # alone (low SSR with sloppy R_CL is useless).
        self.fit_n_restarts = 8
        # SSR-sanity gate for multi-start selection: only restarts whose SSR is
        # within this factor of the best SSR are eligible for the accuracy
        # ranking — others sit in clearly-worse local minima.
        self.fit_ssr_tolerance = 2.0
        # Threshold above which a parameter's relative standard error is flagged
        # as "high uncertainty" to the user (does not block the result).
        self.fit_high_uncertainty_pct = 50.0

    # --- Preprocessing & KK ---
    def process_spectra(self, data_list, freq_range=None):
        f_min, f_max = self._normalize_frequency_range(freq_range)
        use_range = f_min is not None or f_max is not None

        all_scans = []
        warnings = []
        for i, data in enumerate(data_list):
            freq = np.asarray(data['frequency'])
            z_real = np.asarray(data['z_real'])
            z_imag = np.asarray(data['z_imag'])
            z_complex = np.asarray(data['z_complex'])

            if use_range:
                in_scope = np.ones(len(freq), dtype=bool)
                if f_min is not None:
                    in_scope &= freq >= f_min
                if f_max is not None:
                    in_scope &= freq <= f_max

                freq = freq[in_scope]
                z_real = z_real[in_scope]
                z_imag = z_imag[in_scope]
                z_complex = z_complex[in_scope]

            if len(freq) < self.min_kk_points:
                warnings.append(
                    f"Scan {i + 1}: skipped; {len(freq)} in-scope point(s), "
                    f"need at least {self.min_kk_points} for KK validation."
                )
                continue

            try:
                # Mute the automatic print statements from linKK
                with redirect_stdout(io.StringIO()):
                    M, mu, Z_fit, res_real, res_imag = linKK(freq, z_complex, c=0.85)

                point_errors = np.sqrt(res_real ** 2 + res_imag ** 2)

                df_scan = pd.DataFrame({
                    'f': freq, 'zr': z_real, 'zi': z_imag,
                    'scan': i + 1, 'kk_err': point_errors,
                    'in_scope': True,
                    'range_f_min': f_min if f_min is not None else np.nan,
                    'range_f_max': f_max if f_max is not None else np.nan,
                })
                all_scans.append(df_scan)
            except Exception as e:
                msg = f"Scan {i + 1}: KK validation failed ({e})."
                warnings.append(msg)
                print(f"KK Error on scan {i + 1}: {e}")

        if not all_scans:
            detail = " ".join(warnings)
            if use_range:
                raise ValueError(
                    "No spectra could be processed for KK validation inside the selected "
                    f"frequency range. {detail}".strip()
                )
            raise ValueError(f"No spectra could be processed for KK validation. {detail}".strip())

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

        # Drift / RSD across the spectrum.
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

        return df_all, df_avg, (freq_min_suggested, freq_max_suggested), rsd, warnings

    def _normalize_frequency_range(self, freq_range):
        if freq_range is None:
            return None, None

        f_min, f_max = freq_range

        if f_min is not None:
            f_min = float(f_min)
            if not np.isfinite(f_min):
                raise ValueError("f_min must be a finite number.")

        if f_max is not None:
            f_max = float(f_max)
            if not np.isfinite(f_max):
                raise ValueError("f_max must be a finite number.")

        if f_min is not None and f_max is not None and f_min > f_max:
            raise ValueError("f_min must be less than or equal to f_max.")

        return f_min, f_max

    # --- Fitting Models ---
    def JPcoth(self, x):
        return (np.exp(x) + np.exp(-x)) / (np.exp(x) - np.exp(-x))

    def JPtanh(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def evaluate_model(self, model_name, params, freq):
        # Canonical 6-vector: [L_wire, HFR, R_CL, Q_dl, Phi, Theta].
        # L_wire == 0 ⇒ inductive term collapses to 0 regardless of Theta.
        Lwire, HFR, Rcl, Qdl, Phi, Theta = params
        omega = 1j * 2 * np.pi * freq

        Z_L = Lwire * (omega ** Theta) if Lwire != 0 else 0

        # x = √(R_cl · Q_dl · (jω)^φ); prefactor = √(R_cl / (Q_dl·(jω)^φ)) = R_cl / x.
        x = np.sqrt(Rcl * Qdl * (omega ** Phi))
        prefactor = np.sqrt(Rcl / (Qdl * (omega ** Phi)))

        if model_name == "Transmission Line":
            # Porous-electrode TLM / finite-space (restricted) diffusion: reflecting boundary.
            Z = Z_L + HFR + prefactor * self.JPcoth(x)
        elif model_name == "1-D Linear Diffusion":
            # Finite-length (Warburg-short) diffusion: transmissive boundary.
            # Fix vs. upstream OSIF 2.0, which used JPcoth here and so collapsed
            # to the Transmission Line form — see Diard/Le Gorrec/Montella,
            # Handbook of EIS: Diffusion Impedances.
            Z = Z_L + HFR + prefactor * self.JPtanh(x)
        elif model_name == "1-D Spherical Diffusion":
            # Restricted spherical diffusion (reflecting boundary).
            Z = Z_L + HFR + Rcl / (x * self.JPcoth(x) - 1)
        return Z

    def _perturb_init_params(self, init_params, lower_bounds, upper_bounds, rng,
                             fit_inductance=False):
        """Generate a randomized initial guess for a multi-start restart.

        HFR is jittered uniformly within its bounds. R_CL and Q_dl span orders
        of magnitude in practice, so they are drawn log-uniformly. Phi is
        drawn uniformly within physically reasonable bounds. When inductance
        fitting is active, L_wire is log-jittered ±1 decade around its init
        and Theta is drawn uniformly in [0.6, 1.0]. Any draw is clipped to
        the optimizer bounds.
        """
        def log_jitter(value, decades=1.0):
            base = max(abs(value), 1e-9)
            factor = 10.0 ** rng.uniform(-decades, decades)
            return base * factor

        if fit_inductance:
            lwire_init, hfr_init, rcl_init, qdl_init, phi_init, theta_init = init_params
            lwire_lo, hfr_lo = lower_bounds[0], lower_bounds[1]
            lwire_hi, hfr_hi = upper_bounds[0], upper_bounds[1]
            lwire = float(np.clip(log_jitter(lwire_init, decades=1.0), 1e-9, lwire_hi))
            hfr = float(rng.uniform(hfr_lo, hfr_hi))
            rcl = float(np.clip(log_jitter(rcl_init, decades=1.0), 1e-6, 1e6))
            qdl = float(np.clip(log_jitter(qdl_init, decades=1.0), 1e-9, 1e3))
            phi = float(rng.uniform(0.5, 0.95))
            theta = float(rng.uniform(0.6, 1.0))
            return [lwire, hfr, rcl, qdl, phi, theta]

        hfr_init, rcl_init, qdl_init, phi_init = init_params
        hfr_lo, hfr_hi = lower_bounds[0], upper_bounds[0]
        hfr = float(rng.uniform(hfr_lo, hfr_hi))
        rcl = float(np.clip(log_jitter(rcl_init, decades=1.0), 1e-6, 1e6))
        qdl = float(np.clip(log_jitter(qdl_init, decades=1.0), 1e-9, 1e3))
        phi = float(rng.uniform(0.5, 0.95))
        return [hfr, rcl, qdl, phi]

    def fit_impedance(self, model_name, init_params, freq, z_exp, max_nfev=None,
                      n_restarts=None, random_seed=0, fit_inductance=False):
        if max_nfev is None:
            max_nfev = self.fit_max_nfev
        max_nfev = int(max_nfev)
        if max_nfev <= 0:
            raise ValueError("Maximum function evaluations must be a positive integer.")

        if n_restarts is None:
            n_restarts = self.fit_n_restarts
        n_restarts = max(1, int(n_restarts))

        # The optimizer trial vector is 4-element when inductance is off, 6 when on.
        # `evaluate_model` always wants a 6-vector — pad trial vectors accordingly.
        # `params_to_canonical` returns the canonical 6-vector [L_wire, HFR, R_CL, Q_dl, Phi, Theta].
        if fit_inductance:
            if len(init_params) != 6:
                raise ValueError(
                    "init_params must have 6 elements when fit_inductance=True: "
                    "[L_wire, HFR, R_CL, Q_dl, Phi, Theta]."
                )
            hfr_init = init_params[1]
            # Widen HFR window because L_wire absorbs a slice of HFR at high f.
            lower_bounds = [0.0, 0.8 * hfr_init, 0.0, 0.0, 0.0, 0.0]
            upper_bounds = [1.0, 1.2 * hfr_init, np.inf, np.inf, 1.0, 1.0]

            def params_to_canonical(p):
                return p

        else:
            if len(init_params) != 4:
                raise ValueError(
                    "init_params must have 4 elements when fit_inductance=False: "
                    "[HFR, R_CL, Q_dl, Phi]."
                )
            hfr_init = init_params[0]
            lower_bounds = [0.9 * hfr_init, 0.0, 0.0, 0.0]
            upper_bounds = [1.1 * hfr_init, np.inf, np.inf, 1.0]

            def params_to_canonical(p):
                # Pad with L_wire=0 (front) and Theta=0 (back).
                return [0.0, p[0], p[1], p[2], p[3], 0.0]

        def cost_func(params):
            Z_model = self.evaluate_model(model_name, params_to_canonical(params), freq)
            diff = (np.real(Z_model) - np.real(z_exp)) ** 2 + (np.imag(Z_model) - np.imag(z_exp)) ** 2
            return np.sqrt(diff)

        rng = np.random.default_rng(random_seed)

        attempts = []  # each item: dict with res, ssr, se, params, attempt_idx
        attempt_messages = []
        failure_count = 0

        for attempt in range(n_restarts):
            if attempt == 0:
                trial_init = list(init_params)
            else:
                trial_init = self._perturb_init_params(
                    init_params, lower_bounds, upper_bounds, rng,
                    fit_inductance=fit_inductance,
                )

            try:
                res = scipy.optimize.least_squares(
                    cost_func, trial_init, bounds=(lower_bounds, upper_bounds),
                    method='trf', xtol=1e-11, ftol=1e-11, gtol=1e-11,
                    max_nfev=max_nfev,
                )
            except Exception as e:
                failure_count += 1
                attempt_messages.append(f"attempt {attempt + 1}: exception ({e})")
                continue

            ssr = float(res.fun @ res.fun)
            if not np.isfinite(ssr):
                failure_count += 1
                attempt_messages.append(f"attempt {attempt + 1}: non-finite residual")
                continue

            try:
                cov_matrix = np.linalg.inv(res.jac.T @ res.jac)
            except np.linalg.LinAlgError:
                cov_matrix = np.linalg.pinv(res.jac.T @ res.jac)
            dof = len(res.fun) - len(res.x)
            s2 = ssr / dof if dof > 0 else 0.0
            se = np.sqrt(np.maximum(np.diag(cov_matrix * s2), 0.0))

            attempts.append({
                "res": res, "ssr": ssr, "se": se,
                "params": res.x, "attempt_idx": attempt,
            })

        if not attempts:
            return None, (
                "All fitting attempts failed. "
                + "; ".join(attempt_messages[:4])
            )

        # Indices of HFR and R_CL within the optimizer's working vector — depend on
        # which path we're on. Canonical 6-vector indices are HFR=1, R_CL=2.
        hfr_idx = 1 if fit_inductance else 0
        rcl_idx = 2 if fit_inductance else 1

        # --- Ranking: parameter accuracy on HFR and R_CL ---
        # The score is the worst (largest) of the relative standard errors on
        # HFR and R_CL. SSR-sanity gate excludes restarts that clearly sit in a
        # worse local minimum, since SE is only meaningful at a good fit.
        def hfr_rcl_score(a):
            p, s = a["params"], a["se"]
            hfr_rel = (s[hfr_idx] / abs(p[hfr_idx])) if p[hfr_idx] != 0 else np.inf
            rcl_rel = (s[rcl_idx] / abs(p[rcl_idx])) if p[rcl_idx] != 0 else np.inf
            score = max(hfr_rel, rcl_rel)
            return score if np.isfinite(score) else np.inf

        best_ssr = min(a["ssr"] for a in attempts)
        ssr_gate = best_ssr * float(self.fit_ssr_tolerance)
        eligible = [a for a in attempts if a["ssr"] <= ssr_gate]
        ranking_basis = "hfr_rcl_se"
        if not eligible:
            # Every restart's SE was non-finite or the SSR gate filtered them
            # all out (shouldn't happen since the best-SSR result is always
            # inside the gate); fall back to lowest SSR.
            eligible = attempts
            ranking_basis = "ssr_fallback"

        best = min(eligible, key=hfr_rcl_score)
        res = best["res"]
        se = best["se"]
        ssr = best["ssr"]

        # Canonical 6-vector return so the UI/export indexing is stable across paths.
        if fit_inductance:
            params_canonical = np.asarray(res.x, dtype=float)
            se_canonical = np.asarray(se, dtype=float)
        else:
            params_canonical = np.array(
                [0.0, res.x[0], res.x[1], res.x[2], res.x[3], 0.0], dtype=float,
            )
            se_canonical = np.array(
                [0.0, se[0], se[1], se[2], se[3], 0.0], dtype=float,
            )

        info = {
            "ssr": ssr,
            "ssr_best": best_ssr,
            "best_attempt": best["attempt_idx"] + 1,
            "n_restarts": n_restarts,
            "n_eligible": len(eligible),
            "n_failed": failure_count,
            "converged": bool(res.success),
            "message": res.message,
            "nfev": int(res.nfev),
            "ranking_basis": ranking_basis,
            "score_hfr_rcl_pct": hfr_rcl_score(best) * 100.0,
            "fit_inductance": bool(fit_inductance),
        }

        return (params_canonical, se_canonical, cost_func(res.x), info), None
