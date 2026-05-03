import tkinter as tk
from tkinter import ttk
import ctypes
import sys

from core.style import apply_matplotlib_style, apply_ttk_style, BG
from osif.ui import OsifUI
from lsv_decoupling.ui import LsvUI


def main():
    if sys.platform == "win32":
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass

    apply_matplotlib_style()

    root = tk.Tk()
    root.title("Electrochemical Impedance & Voltammetry Analyzer")
    root.configure(background=BG)

    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    root.geometry(f"{int(sw * 0.85)}x{int(sh * 0.85)}")
    root.minsize(1200, 700)

    apply_ttk_style()

    notebook = ttk.Notebook(root)
    notebook.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

    tab_osif = OsifUI(notebook)
    tab_lsv = LsvUI(notebook)

    notebook.add(tab_osif, text="EIS Fitting (OSIF)")
    notebook.add(tab_lsv, text="LSV Decoupling")

    root.mainloop()


if __name__ == "__main__":
    main()
