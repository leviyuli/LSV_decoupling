import os
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# LSV column / unit detection
# ---------------------------------------------------------------------------

# Order matters: longer / more-specific patterns are tried first so that
# "e(mv)" matches the mV branch instead of falling through to "e(v".
_V_PATTERNS = [
    ('e(mv',          1e-3),
    ('e/mv',          1e-3),
    ('potential(mv)', 1e-3),
    ('voltage(mv)',   1e-3),
    ('ewe(mv)',       1e-3),
    ('e(v',           1.0),
    ('e/v',           1.0),
    ('potential',     1.0),
    ('voltage',       1.0),
    ('we.potential',  1.0),
    ('wepot',         1.0),
    ('ewe',           1.0),
]

_I_PATTERNS = [
    ('i(µa',  1e-6),  # i(µA
    ('j(µa',  1e-6),  # j(µA
    ('i(ua',       1e-6),
    ('j(ua',       1e-6),
    ('µa/cm', 1e-6),
    ('ua/cm',      1e-6),
    ('i(ma',       1e-3),
    ('j(ma',       1e-3),
    ('i/ma',       1e-3),
    ('j/ma',       1e-3),
    ('ma/cm',      1e-3),
    ('i(a',        1.0),
    ('j(a',        1.0),
    ('i/a',        1.0),
    ('j/a',        1.0),
    ('a/cm',       1.0),
    ('we.current', 1.0),
    ('current',    1.0),
]

_BARE_V = {'e', 'v', 'u'}
_BARE_I = {'i', 'j'}


def _classify_column(col_name):
    """
    Classify a column-header string.

    Returns (kind, scale, is_density) where:
        kind: 'V', 'i', or None
        scale: multiplier to bring values to V or A
        is_density: True if header contains a /cm² hint, False if explicitly
            absolute current, None if unknown.
    """
    s = str(col_name).strip().lower().replace(' ', '')

    for pat, scale in _V_PATTERNS:
        if pat in s:
            return ('V', scale, None)

    is_density = ('/cm' in s) or ('cm²' in s) or ('cm^2' in s) \
        or ('cm-2' in s) or ('cm⁻²' in s)

    for pat, scale in _I_PATTERNS:
        if pat in s:
            # A bare "(a)" or "(ma)" with no /cm hint -> probably absolute current
            if pat in ('i(a', 'j(a', 'i/a', 'j/a', 'i(ma', 'j(ma', 'i/ma', 'j/ma') \
                    and not is_density:
                return ('i', scale, False)
            return ('i', scale, is_density if is_density else True)

    if s in _BARE_V:
        return ('V', 1.0, None)
    if s in _BARE_I:
        return ('i', 1.0, None)

    return (None, 1.0, None)


def _find_cols(columns):
    """
    Locate the V and i column indices from a sequence of header strings.
    Strong (parenthesized) matches outrank bare single-letter matches.
    """
    strong_v = strong_i = bare_v = bare_i = None
    for j, c in enumerate(columns):
        s = str(c).strip().lower()
        is_bare = s in _BARE_V or s in _BARE_I
        kind, _, _ = _classify_column(c)
        if kind == 'V':
            if is_bare and bare_v is None:
                bare_v = j
            elif not is_bare and strong_v is None:
                strong_v = j
        elif kind == 'i':
            if is_bare and bare_i is None:
                bare_i = j
            elif not is_bare and strong_i is None:
                strong_i = j
    v_idx = strong_v if strong_v is not None else bare_v
    i_idx = strong_i if strong_i is not None else bare_i
    return v_idx, i_idx


def _find_header_row(lines, max_scan=80):
    """
    Scan up to the first ``max_scan`` lines for a row that contains both a
    voltage- and a current-style column name. Returns the line index or -1.
    """
    for idx, line in enumerate(lines[:max_scan]):
        # Split on tab/comma/whitespace to avoid matching content inside a
        # base64 metadata blob that happens to contain the letter 'e'.
        for sep in ('\t', ',', None):
            tokens = line.split(sep) if sep else line.split()
            if len(tokens) < 2:
                continue
            v_idx, i_idx = _find_cols(tokens)
            if v_idx is not None and i_idx is not None and v_idx != i_idx:
                return idx
    return -1


def _trim_to_forward_sweep(V, i):
    """
    If V is cyclic (rises then falls or vice-versa), return only the segment
    up to the apex. Returns (V_out, i_out, trimmed_flag).
    """
    if len(V) < 3:
        return V, i, False
    dV = np.diff(V)
    pos = int(np.sum(dV > 0))
    neg = int(np.sum(dV < 0))
    if pos > 0 and neg > 0:
        if pos >= neg:
            apex = int(np.argmax(V))
        else:
            apex = int(np.argmin(V))
        if 0 < apex < len(V) - 1:
            return V[:apex + 1], i[:apex + 1], True
    return V, i, False


def _read_single_lsv(filepath):
    """
    Read one LSV file, auto-detect E and i/j columns, normalize units to V
    and A (per cm² when discoverable), and trim cyclic sweeps to the forward
    segment.

    Returns (V, i, info, err). On failure V/i are None and err is a string.
    info is always populated with at least 'file'.
    """
    info = {'file': os.path.basename(filepath)}
    try:
        ext = os.path.splitext(filepath)[1].lower()

        if ext in ('.xlsx', '.xls'):
            df_raw = pd.read_excel(filepath, header=None)
            header_idx = -1
            for r in range(min(len(df_raw), 20)):
                row = [str(x) for x in df_raw.iloc[r].tolist()]
                v_idx, i_idx = _find_cols(row)
                if v_idx is not None and i_idx is not None and v_idx != i_idx:
                    header_idx = r
                    break
            if header_idx >= 0:
                df = df_raw.iloc[header_idx + 1:].reset_index(drop=True)
                df.columns = [str(x) for x in df_raw.iloc[header_idx].tolist()]
            else:
                df = pd.read_excel(filepath)

        elif ext in ('.csv', '.txt'):
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            header_idx = _find_header_row(lines)
            default_delim = ',' if ext == '.csv' else r'\s+'
            if header_idx >= 0:
                hdr_line = lines[header_idx]
                if '\t' in hdr_line:
                    delim = '\t'
                elif ',' in hdr_line and ext == '.csv':
                    delim = ','
                else:
                    delim = default_delim
                df = pd.read_csv(
                    filepath, sep=delim, skiprows=header_idx,
                    engine='python', on_bad_lines='skip',
                )
            else:
                df = pd.read_csv(
                    filepath, sep=default_delim, engine='python',
                    header=None, on_bad_lines='skip',
                )
        else:
            return None, None, info, f"Unsupported file extension: {ext}. Allowed: .txt, .csv, .xlsx"

        if df.shape[1] < 2:
            return None, None, info, "File must have at least two columns (Voltage, Current)."

        v_idx, i_idx = _find_cols(df.columns)
        if v_idx is None or i_idx is None:
            # Fallback: assume column 0 = V, column 1 = i (preserves legacy
            # behavior for plain header-less 2-column files).
            v_idx, i_idx = 0, 1
            v_name, i_name = 'col0', 'col1'
            v_scale, i_scale, is_density = 1.0, 1.0, True
        else:
            v_name = str(df.columns[v_idx])
            i_name = str(df.columns[i_idx])
            _, v_scale, _ = _classify_column(v_name)
            _, i_scale, is_density = _classify_column(i_name)
            if is_density is None:
                is_density = True  # benefit of the doubt for ambiguous labels

        V = pd.to_numeric(df.iloc[:, v_idx], errors='coerce') * v_scale
        i = pd.to_numeric(df.iloc[:, i_idx], errors='coerce') * i_scale

        mask = V.notna() & i.notna()
        V = V[mask].to_numpy(dtype=float)
        i = i[mask].to_numpy(dtype=float)

        if len(V) == 0:
            return None, None, info, "No numeric rows found in the data table."

        V, i, trimmed = _trim_to_forward_sweep(V, i)

        info.update({
            'v_col': v_name,
            'i_col': i_name,
            'v_scale': v_scale,
            'i_scale': i_scale,
            'is_density': bool(is_density),
            'sweep_trimmed': trimmed,
            'n_rows': len(V),
        })
        if is_density is False:
            info['warning'] = (
                f"Column '{i_name}' looks like absolute current (no /cm² in header); "
                "results will be off unless the file is already normalized."
            )
        return V, i, info, None

    except Exception as e:
        return None, None, info, f"Error reading {info['file']}: {e}"


def read_lsv_data(filepaths):
    """
    Load LSV data from one or more files.

    Accepts a single path or an iterable of paths. Auto-detects the E and i/j
    columns by header keyword, converts mV/mA/µA to V/A, trims cyclic sweeps
    to the forward segment, and (for multiple files) interpolates onto a
    common voltage grid before averaging.

    Returns (data, err):
        data: (V_array, i_array, info_dict) on success; None on failure
        err:  error message string on failure; None on success

    info_dict keys: files (list), v_col, i_col, is_density, sweep_trimmed,
    averaged (bool), n_rows, and an optional 'warning'.
    """
    if filepaths is None:
        return None, "No file selected."
    if isinstance(filepaths, str):
        paths = [filepaths]
    else:
        paths = [p for p in filepaths if p]
    if not paths:
        return None, "No file selected."

    series, file_infos, warns = [], [], []
    for fp in paths:
        V, i, info, err = _read_single_lsv(fp)
        file_infos.append(info)
        if err is not None:
            warns.append(err)
            continue
        if 'warning' in info:
            warns.append(f"{info['file']}: {info['warning']}")
        order = np.argsort(V)
        series.append((V[order], i[order], info))

    if not series:
        return None, "Could not read any file:\n" + "\n".join(warns)

    if len(series) == 1:
        V_out, i_out, base_info = series[0]
        averaged = False
    else:
        v_lo = max(s[0].min() for s in series)
        v_hi = min(s[0].max() for s in series)
        if v_hi <= v_lo:
            return None, "Selected files have no overlapping voltage range."
        n_med = int(np.median([len(s[0]) for s in series]))
        n_med = max(n_med, 50)
        V_out = np.linspace(v_lo, v_hi, n_med)
        stack = np.stack([np.interp(V_out, s[0], s[1]) for s in series])
        i_out = stack.mean(axis=0)
        base_info = series[0][2]
        averaged = True

    # Final mask: Tafel fit needs i > 0.
    mask = np.isfinite(V_out) & np.isfinite(i_out) & (i_out > 0)
    V_clean = V_out[mask]
    i_clean = i_out[mask]
    if len(V_clean) == 0:
        return None, "No data points with positive current density after filtering."

    info = {
        'files': file_infos,
        'v_col': base_info.get('v_col', 'E'),
        'i_col': base_info.get('i_col', 'i'),
        'is_density': base_info.get('is_density', True),
        'sweep_trimmed': any(s[2].get('sweep_trimmed', False) for s in series),
        'averaged': averaged,
        'n_rows': len(V_clean),
    }
    if warns:
        info['warning'] = ' | '.join(warns)
    return (V_clean, i_clean, info), None


def read_eis_data(filepath):
    """
    Reads EIS data robustly. Scans for headers to automatically skip metadata strings.
    Extracts only Frequency, Z', and Z'', ignoring all other columns.
    """
    try:
        ext = os.path.splitext(filepath)[1].lower()

        if ext in [".xlsx", ".xls"]:
            df = pd.read_excel(filepath, header=None)
        elif ext in [".csv", ".txt"]:
            # 1. Manually hunt for the header row to prevent Pandas from choking on
            #    inconsistent column counts (like single-column metadata headers)
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()

            header_idx = -1
            delimiter = ',' if ext == '.csv' else r'\s+'

            for i, line in enumerate(lines):
                line_lower = line.lower()

                has_f = 'freq' in line_lower or 'hz' in line_lower
                has_zi = "z''" in line_lower or 'z"' in line_lower or "im(z)" in line_lower or "zimag" in line_lower
                has_zr = "z'" in line_lower or "re(z)" in line_lower or "zreal" in line_lower

                if has_f and has_zi and has_zr:
                    header_idx = i
                    if '\t' in line and ext == '.txt':
                        delimiter = '\t'
                    break

            # Read the file starting directly from the header
            if header_idx != -1:
                df = pd.read_csv(filepath, sep=delimiter, skiprows=header_idx, engine='python')
            else:
                df = pd.read_csv(filepath, sep=delimiter, engine='python', header=None, on_bad_lines='skip')
        else:
            return None, f"Unsupported file extension: {ext}. Allowed: .txt, .csv, .xlsx"

        freq_col, zr_col, zi_col = -1, -1, -1

        # 2. Extract Columns
        # If the dataframe has string columns (header was passed directly to read_csv)
        if ext in [".csv", ".txt"] and header_idx != -1:
            cols = [str(c).lower() for c in df.columns]

            # Find Z'' (Imaginary) FIRST
            for j, val in enumerate(cols):
                if "z''" in val or 'z"' in val or "z imag" in val or "zi" in val or "im(z)" in val or "zimag" in val:
                    zi_col = j
                    break

            # Find Z' (Real) - Make sure we don't accidentally match the Z'' column
            for j, val in enumerate(cols):
                if j == zi_col:
                    continue
                if "z'" in val or "z real" in val or "zr" in val or "re(z)" in val or "zreal" in val:
                    zr_col = j
                    break

            # Find Freq
            for j, val in enumerate(cols):
                if 'freq' in val or 'hz' in val:
                    freq_col = j
                    break
        else:
            # Row-by-row hunt (for Excel or un-headered fallback)
            for i, row in df.iterrows():
                row_strs = [str(x).lower() for x in row.values]

                f_match = [j for j, val in enumerate(row_strs) if 'freq' in val or 'hz' in val]
                zi_match = [j for j, val in enumerate(row_strs) if
                            "z''" in val or 'z"' in val or "z imag" in val or "zi" in val or "im(z)" in val or "zimag" in val]
                zr_match = [j for j, val in enumerate(row_strs) if (
                            "z'" in val or "z real" in val or "zr" in val or "re(z)" in val or "zreal" in val) and j not in zi_match]

                if f_match and zi_match and zr_match:
                    freq_col = f_match[0]
                    zi_col = zi_match[0]
                    zr_col = zr_match[0]
                    # Slice df to start AFTER the header row
                    df = df.iloc[i + 1:].copy()
                    break

        if freq_col != -1 and zr_col != -1 and zi_col != -1:
            freq = pd.to_numeric(df.iloc[:, freq_col], errors='coerce')
            z_real = pd.to_numeric(df.iloc[:, zr_col], errors='coerce')
            z_imag = pd.to_numeric(df.iloc[:, zi_col], errors='coerce')
        else:
            # Pure numeric fallback if no headers exist at all
            df_numeric = df.apply(pd.to_numeric, errors='coerce').dropna(thresh=3)
            if df_numeric.empty:
                return None, "No valid numeric data or recognizable headers found."
            freq = df_numeric.iloc[:, 0]
            z_real = df_numeric.iloc[:, 1]
            z_imag = df_numeric.iloc[:, 2]

        # 3. Final Clean and Calculate
        mask = freq.notna() & z_real.notna() & z_imag.notna()
        freq_clean = freq[mask].to_numpy()
        z_real_clean = z_real[mask].to_numpy()
        z_imag_clean = z_imag[mask].to_numpy()

        if len(freq_clean) == 0:
            return None, "No valid EIS data points found after parsing."

        z_complex = z_real_clean + 1j * z_imag_clean
        z_mod = np.abs(z_complex)
        phase = (180 / np.pi) * np.arctan2(z_imag_clean, z_real_clean)

        data_dict = {
            'frequency': freq_clean,
            'z_real': z_real_clean,
            'z_imag': z_imag_clean,
            'z_mod': z_mod,
            'phase': phase,
            'z_complex': z_complex
        }

        return data_dict, None

    except Exception as e:
        return None, f"Error reading the EIS file:\n{e}"