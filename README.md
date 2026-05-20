# Electrochemical Impedance & Voltammetry Analyzer

A unified, Python-based desktop application for advanced electrochemical data analysis. This software integrates two powerful modules into a single tabbed interface:
1. **EIS Processing & Fitting (OSIF-Revised):** For rigorous Kramers-Kronig validation, automated data averaging, and non-Faradaic impedance fitting to extract Catalyst Layer Resistance ($R_{cl}$) and High-Frequency Resistance ($HFR$).
2. **LSV Decoupling:** For dissecting Linear Sweep Voltammetry (LSV) curves into kinetic, ohmic, catalyst-layer, and residual overpotentials using robust, iterative Tafel analysis.

---

## 🚀 Installation & Quick Start

**Prerequisites:** Python 3.8+ installed on your system.

1. **Install dependencies:**
   Open your terminal in the project directory and run:
   ```bash
   pip install -r requirements.txt
   ```
   *(Required packages: `numpy`, `pandas`, `matplotlib`, `scipy`, `openpyxl`, `impedance`)*

2. **Launch the software:**
   ```bash
   python main.py
   ```

---

## 🛠️ Module 1: EIS Fitting (OSIF-Revised)

This module is a heavily customized version of the [NREL Open Source Impedance Fitter (OSIF)](https://github.com/NREL/OSIF), tailored for water electrolysis Membrane Electrode Assemblies (MEAs). It adds robust pre-processing, a multi-start fit ranked by parameter accuracy, and an opt-in wire-inductance term.

### What it does:
* **Automated Data Parsing:** Intelligently scans `.txt`, `.csv`, or `.xlsx` files to locate headers and extract Frequency, Z', and Z'' columns. No manual file formatting is required.
* **Validation & Averaging:** Runs a point-by-point Linear Kramers-Kronig (Lin-KK) test to identify unphysical data points. If $\geq 3$ spectra are loaded, it automatically filters outliers ($Z > 1.5$) and calculates a final averaged curve with Relative Standard Deviation (RSD).
* **Multi-start Impedance Fitting:** Runs N optimizer launches (default 8) from perturbed initial guesses and keeps the result with the **lowest combined relative standard error on $HFR$ and $R_{cl}$** — not just the lowest residual. Restarts whose SSR is more than $2\times$ the best are rejected from the ranking as clearly worse local minima.
* **Optional Wire-Inductance Term:** A checkbox enables fitting $L_{wire}$ and $\theta$ alongside $HFR$, $R_{cl}$, $Q_{dl}$, $\phi$ (matching the original OSIF 2.0 parameterisation). When off (default), the model collapses to $Z = HFR + Z_\mathrm{diffusive}$ for bit-identical legacy behaviour.

### How to use:
1. **Load Files:** Click "Add" to select one or multiple EIS data files. Use the listbox to manage your loaded files.
2. **Preprocess:** Click **"Run preprocess & KK test"**. The app plots a Nyquist validation chart (Valid vs. Invalid points) and a Bode Error plot, and auto-fills an estimated $HFR$ from the high-frequency intercept. The fit window is narrowed to the KK-valid sub-range.
3. **Fit Model:**
   * Ensure "Enable impedance fitting" is checked.
   * (Optional) Tick **"Fit wire inductance (L_wire, Θ)"** to free the two inductive parameters; defaults seed to $L_{wire}=2\times10^{-5}\,\mathrm{H\cdot cm^2}$, $\theta=0.95$.
   * Adjust the frequency window if needed; tune **Max evaluations** and **Restarts** under the model selector.
   * Click **"Fit model"**. The status line under the controls reports the chosen restart, the max relative SE on $HFR$ / $R_{cl}$, and the SSR. Parameters whose SE exceeds 50% are highlighted in red as "high uncertainty".
4. **Export:** Click **"Export results…"** to generate an Excel workbook containing the raw/diagnostic data, the averaged dataset, the fitted curve coordinates, and the final parameter table.

> The control column is scrollable — use the mouse wheel inside the left panel if the window is too short to show everything at once.

### The Underlying Logic & Math
* **Multi-start with parameter-accuracy ranking:** Every restart's covariance is computed; the winner minimises $\max(\mathrm{SE}_{HFR}/|HFR|,\ \mathrm{SE}_{R_{cl}}/|R_{cl}|)$ among restarts within the SSR-sanity gate. This biases the choice toward fits that *determine* $HFR$ and $R_{cl}$ rather than fits that merely chase the residual.
* **Robust Error Estimation:** To prevent the algorithm from crashing on singular Hessian matrices during least-squares optimization, the code falls back to a pseudo-inverse (`np.linalg.pinv`), allowing fitting to proceed even with highly correlated parameters.
* **Wire-inductance model:** When enabled, the impedance gains a $L_{wire}\cdot(j\omega)^{\theta}$ prefactor with $L_{wire}\in[0,1]\,\mathrm{H\cdot cm^2}$ and $\theta\in[0,1]$, matching upstream OSIF 2.0. The $HFR$ bound is widened from $\pm10\%$ to $\pm20\%$ in this mode because $L_{wire}$ partially absorbs $HFR$ at high frequency. Inductive parameters are known to be ill-conditioned without sufficient high-frequency resolution — leave the toggle **off** unless your data extends well into the inductive ridge.
* **Models Included:**
  1. **Transmission Line (Default):** For porous electrodes (e.g., PEMFC catalyst layers). With inductance off:
     $$Z(\omega)=HFR+\sqrt{\frac{R_{cl}}{Q_{dl}(j\omega)^{\phi}}}\coth\left(\sqrt{R_{cl}Q_{dl}(j\omega)^{\phi}}\right)$$
     With inductance on, add $L_{wire}(j\omega)^{\theta}$ to the right-hand side.
  2. **1-D Linear Diffusion:** For planar electrode linear diffusion.
  3. **1-D Spherical Diffusion:** For nanoparticle/spherical diffusion limits.

---

## 🛠️ Module 2: LSV Decoupling

This module is a lightweight tool for dissecting polarization curves to understand the exact source of performance losses in your electrochemical cell.

### What it does:
* **Reference Conversion:** Automatically converts raw voltages to the Reversible Hydrogen Electrode (RHE) scale using built-in offsets (Ag/AgCl, SCE, HgO) and pH corrections.
* **Iterative Tafel Fitting:** Auto-hunts for a robust Tafel window and iteratively corrects for ohmic ($HFR$) and catalyst-layer ($R_{cl}$) voltage drops.
* **Overpotential Breakdown:** Decomposes the measured potential into:
  $E = E_{rev} + \eta_{kin} + \eta_{ohm} + \eta_{RCL} + \eta_{res}$
* **Interactive Column Plot:** Allows you to input a target current density (e.g., $1.0 \text{ A/cm}^2$) to instantly generate a stacked column breakdown of all overpotentials at that exact operational point.

### How to use:
1. **Load Data:** Browse for your LSV file (Column 0 = Voltage, Column 1 = Current Density).
2. **Set Parameters:** Input the Reference Electrode, pH, and Temperature to calculate $E_{rev}$.
3. **Input Resistances:** Enter the $HFR$ and $R_{cl}$ values (you can obtain these directly from the OSIF module!).
4. **Fit & Plot:** Set your desired Tafel current bounds (e.g., 0.02 to 0.8 A/cm²) and click **"Fit & Plot"**. 
5. **Analyze:** Switch between the "Decoupled Plot" (stacked LSV curves) and "Overpotential Column Plot" (target current breakdown). You can toggle the X-axis between Linear and Log scales.
6. **Export:** Generates a comprehensive Excel file containing the stacked curve data, isolated overpotential components, and Tafel fitting metadata ($b_{kin}$, $i_0$, $R^2$).

### The Underlying Logic & Math
* **Iterative Correction Loop:** The software subtracts $\eta_{ohm} = i \cdot HFR$ and computes $\eta_{RCL}$ from the current Tafel slope guess ($b_{kin}$) using a utilization law. It forms $\eta_{corr} = (V - E_{rev}) - \eta_{ohm} - \eta_{RCL}$ and performs a Tafel fit in $\log_{10}(i)$ vs $\eta$ space. It repeats this until $b_{kin}$ converges ($\Delta b < 10^{-3} \text{ V/dec}$).
* **Robust Tafel-Window Hunt:** Inside the user-defined current range, the code slides a 60 mV window and evaluates three nested subwindows (20/40/60 mV). It computes slopes/intercepts for each and chooses the candidate with the smallest combined relative standard error. This balances bias-variance and avoids cherry-picking ultra-narrow spans.
* **Catalyst Layer Utilization:**
  $$\eta_{RCL} = -b_{kin} \log_{10}(U_J)$$
  Where $U_J$ is calculated from $\frac{i \cdot \ln(10) \cdot R_{cl}}{2 \cdot b_{kin}}$.

---

## 📚 Acknowledgements & Citations
* The **OSIF** module is derived from the [NREL/OSIF GitHub Repository](https://github.com/NREL/OSIF), significantly revised for robust MEA $R_{cl}$ determination and automated data handling. 
* The **LSV Decoupling** methodology is inspired by practices detailed in:
  * *ACS Phys. Chem Au* (10.1021/acs.jpcc.9b06820)
  * *J. Electrochem. Soc.* (10.1149/1945-7111/acee25)
  * *ACS Catalysis* (10.1021/acscatal.4c02932)
* Kramers-Kronig validation utilizes the open-source `impedance.py` library.
