import tkinter as tk
from tkinter import ttk
import ctypes
import sys

# Import our refactored UI modules
from osif.ui import OsifUI
from lsv_decoupling.ui import LsvUI


def main():
    # Make the UI crisp on high-DPI Windows displays
    if sys.platform == "win32":
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass

    root = tk.Tk()
    root.title("Electrochemical Impedance & Voltammetry Analyzer")

    # Responsive window sizing based on the user's screen
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    root.geometry(f"{int(sw * 0.85)}x{int(sh * 0.85)}")
    root.minsize(1200, 700)

    # Apply a clean, modern theme globally
    style = ttk.Style()
    if "clam" in style.theme_names():
        style.theme_use("clam")

    # Enforce the Arial font globally to keep the UI consistent
    style.configure(".", font=("Arial", 10))
    style.configure("TNotebook.Tab", font=("Arial", 11, "bold"), padding=[10, 5])

    # Create the main tabbed interface (Notebook)
    notebook = ttk.Notebook(root)
    notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    # Initialize the modules
    tab_osif = OsifUI(notebook)
    tab_lsv = LsvUI(notebook)

    # Add modules to the notebook
    notebook.add(tab_osif, text="Module 1: EIS Fitting (OSIF)")
    notebook.add(tab_lsv, text="Module 2: LSV Decoupling")

    # Start the application
    root.mainloop()


if __name__ == "__main__":
    main()