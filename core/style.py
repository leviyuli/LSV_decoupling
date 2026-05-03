"""Shared minimalist scientific styling for the application.

Centralises matplotlib rcParams and the ttk theme so every plot and widget
shares the same Nature/Science-journal-like restraint.
"""
from tkinter import ttk

import matplotlib as mpl

# --- Color tokens -----------------------------------------------------------
INK = "#1F2937"        # near-black for axes/text
SUBTLE = "#9CA3AF"     # grey for grids
BORDER = "#E5E7EB"     # light divider grey
PANEL = "#F9FAFB"      # off-white panel surfaces
BG = "#FFFFFF"
ACCENT = "#2563EB"     # indigo accent
ACCENT_HOVER = "#1D4ED8"

# Sequential palette for the overpotential stack: thermo -> kinetic -> ohmic
# -> catalyst-layer -> residual.  Chosen for legibility and print-friendliness.
OVERPOT_COLORS = {
    "E_rev":   "#374151",  # graphite
    "eta_kin": "#1D4ED8",  # indigo
    "eta_ohm": "#0F766E",  # teal
    "eta_rcl": "#B45309",  # amber
    "eta_res": "#B91C1C",  # crimson
}
OVERPOT_LABELS = {
    "E_rev":   r"$E_\mathrm{rev}$",
    "eta_kin": r"$\eta_\mathrm{kin}$",
    "eta_ohm": r"$\eta_\mathrm{ohm}$",
    "eta_rcl": r"$\eta_\mathrm{RCL}$",
    "eta_res": r"$\eta_\mathrm{res}$",
}

# LSV stacked-curve colors (one per cumulative trace) and the experimental dot.
LSV_CURVE_COLORS = [
    OVERPOT_COLORS["E_rev"],
    OVERPOT_COLORS["eta_kin"],
    OVERPOT_COLORS["eta_ohm"],
    OVERPOT_COLORS["eta_rcl"],
    OVERPOT_COLORS["eta_res"],
]
LSV_DATA_COLOR = "#111827"


def apply_matplotlib_style():
    """Apply a clean, minimalist scientific matplotlib style."""
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "regular",
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,

        "axes.facecolor": "white",
        "axes.edgecolor": INK,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlepad": 12,
        "axes.labelpad": 6,

        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": SUBTLE,
        "grid.linewidth": 0.4,
        "grid.linestyle": "-",
        "grid.alpha": 0.25,

        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.direction": "out",
        "ytick.direction": "out",

        "lines.linewidth": 1.8,
        "lines.markersize": 5,

        "legend.frameon": False,
        "legend.borderpad": 0.4,
        "legend.handlelength": 1.6,

        "figure.facecolor": "white",
        "figure.dpi": 100,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })


def apply_ttk_style():
    """Apply a clean, minimalist ttk theme on top of clam."""
    style = ttk.Style()
    if "clam" in style.theme_names():
        style.theme_use("clam")

    # Family: Segoe UI (Windows 11 default) with sensible fallbacks.
    base_font = ("Segoe UI", 10)
    bold_font = ("Segoe UI", 10, "bold")
    italic_font = ("Segoe UI", 9, "italic")

    style.configure(".", font=base_font, background=BG, foreground=INK)
    style.configure("TFrame", background=BG)
    style.configure("TLabel", background=BG, foreground=INK)
    style.configure("Status.TLabel", background=BG, foreground=ACCENT, font=italic_font)
    style.configure("Hint.TLabel", background=BG, foreground=SUBTLE, font=italic_font)

    style.configure(
        "TLabelframe",
        background=BG,
        bordercolor=BORDER,
        lightcolor=BORDER,
        darkcolor=BORDER,
        relief="solid",
        borderwidth=1,
        padding=8,
    )
    style.configure(
        "TLabelframe.Label",
        background=BG,
        foreground=INK,
        font=bold_font,
        padding=(4, 0),
    )

    style.configure(
        "TButton",
        background=PANEL,
        foreground=INK,
        bordercolor=BORDER,
        lightcolor=BORDER,
        darkcolor=BORDER,
        padding=(10, 6),
        relief="flat",
        focusthickness=0,
    )
    style.map(
        "TButton",
        background=[("active", "#EEF2FF"), ("pressed", "#E0E7FF"), ("disabled", PANEL)],
        foreground=[("disabled", SUBTLE)],
    )

    style.configure(
        "Accent.TButton",
        background=ACCENT,
        foreground="white",
        padding=(10, 6),
        relief="flat",
        focusthickness=0,
        bordercolor=ACCENT,
    )
    style.map(
        "Accent.TButton",
        background=[("active", ACCENT_HOVER), ("pressed", "#1E40AF"), ("disabled", "#93C5FD")],
        foreground=[("disabled", "white")],
    )

    style.configure(
        "TEntry",
        fieldbackground="white",
        foreground=INK,
        bordercolor=BORDER,
        lightcolor=BORDER,
        darkcolor=BORDER,
        padding=4,
    )
    style.map("TEntry", bordercolor=[("focus", ACCENT)])

    style.configure(
        "TCombobox",
        fieldbackground="white",
        foreground=INK,
        bordercolor=BORDER,
        lightcolor=BORDER,
        darkcolor=BORDER,
        arrowcolor=INK,
        padding=3,
    )
    style.map("TCombobox", bordercolor=[("focus", ACCENT)])

    style.configure(
        "TNotebook",
        background=BG,
        borderwidth=0,
        tabmargins=[2, 6, 2, 0],
    )
    style.configure(
        "TNotebook.Tab",
        padding=[16, 8],
        font=bold_font,
        background=PANEL,
        foreground=SUBTLE,
        borderwidth=0,
    )
    style.map(
        "TNotebook.Tab",
        background=[("selected", BG)],
        foreground=[("selected", ACCENT)],
        expand=[("selected", [1, 1, 1, 0])],
    )

    style.configure("TCheckbutton", background=BG, foreground=INK)
    style.map("TCheckbutton", background=[("active", BG)])

    style.configure(
        "Vertical.TScrollbar",
        background=PANEL,
        bordercolor=BORDER,
        arrowcolor=INK,
        troughcolor=BG,
    )
    style.configure(
        "Horizontal.TScale",
        background=BG,
        troughcolor=BORDER,
    )

    style.configure("TSeparator", background=BORDER)


def style_axes(ax):
    """Apply a final pass of polish to a matplotlib Axes object."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_color(INK)
    ax.tick_params(colors=INK)
    ax.grid(True, alpha=0.25, linewidth=0.4)
    ax.set_axisbelow(True)
