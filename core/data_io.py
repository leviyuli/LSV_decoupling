import os
import numpy as np
import pandas as pd


def read_lsv_data(filepath):
    """
    Reads LSV data from .txt, .csv, or .xlsx files.
    Assumes Column 0 is Voltage (V) and Column 1 is Current Density (A/cm²).
    Filters out NaNs and negative/zero currents for logarithmic Tafel fitting.
    """
    try:
        ext = os.path.splitext(filepath)[1].lower()

        if ext in [".xlsx", ".xls"]:
            df = pd.read_excel(filepath)
        elif ext == ".csv":
            df = pd.read_csv(filepath)
        elif ext == ".txt":
            df = pd.read_csv(filepath, sep=r'\s+', engine='python', header=None)
        else:
            return None, f"Unsupported file extension: {ext}. Allowed: .txt, .csv, .xlsx"

        if df.shape[1] < 2:
            return None, "The file must have at least two columns (Voltage, Current)."

        V = pd.to_numeric(df.iloc[:, 0], errors='coerce')
        i = pd.to_numeric(df.iloc[:, 1], errors='coerce')

        mask = (i > 0) & V.notna() & i.notna()
        V_clean = V[mask].to_numpy()
        i_clean = i[mask].to_numpy()

        if len(V_clean) == 0:
            return None, "No valid data found (needs positive currents and numeric values)."

        return (V_clean, i_clean), None

    except Exception as e:
        return None, f"Error reading the LSV file:\n{e}"


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