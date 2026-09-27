# Mwaziny 5D Continuum Mechanics Model for Galactic Dynamics

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22994011.svg)](https://doi.org/10.5281/zenodo.22994011)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official Python implementation and empirical validation dataset for the **Mwaziny Cosmological Model**, a 5D continuum-mechanics framework resolving galactic rotation curves and the Empirical Radial Acceleration Relation (RAR) without particle dark matter.

---

## Paper & Publication Details
- **Title:** A 5D Continuum Mechanics Foundation for the Empirical Radial Acceleration Relation and Galactic Rotation Dynamics: The Mwaziny Model
- **Author:** Ahmed Mwaziny (Independent Researcher, Alexandria, Egypt)
- **ORCID:** [0009-0005-4306-6397](https://orcid.org/0009-0005-4306-6397)
- **Publication DOI:** [10.5281/zenodo.22994011](https://doi.org/10.5281/zenodo.22994011)

---

## Key Empirical Findings
- **Dataset:** Evaluated on $N = 175$ disk galaxies from the **SPARC** (Spitzer Photometry and Accurate Rotation Curves) database.
- **Universal Parameter:** Calibrated dimensionless vertical tension coupling constant $\lambda_z = 0.7950 \pm 0.0361$.
- **Validation Protocol:** Pre-registered 50/50 blind train/test cross-validation split ($N_{\text{train}} = 87$, $N_{\text{test}} = 88$).
- **Generalization Performance:** Out-of-sample blind test median $R^2 = 0.7468$ (reaching $R^2 = 0.8470$ in the upper quartile of major spiral galaxies).

---

## Repository Contents

| File | Description |
| :--- | :--- |
| `sparc_mwaziny_unbiased_master.py` | Complete Master Python script executing data loading, Bayesian stellar M/L optimization, 50/50 blind cross-validation, global fitting, and benchmark plotting. |
| `Mwaziny_SPARC_Unbiased_Results.csv` | Full galaxy-by-galaxy exported dataset containing individual $R^2$ scores, reduced $\chi^2$, optimal disk mass-to-light ratios ($M/L_{\text{disk}}$), and velocity maximums ($V_{\text{max}}$). |

---

## Requirements & Execution

### Prerequisites
Make sure you have Python 3.8+ installed along with the necessary scientific libraries:

```bash
pip install numpy pandas matplotlib scipy scikit-learn
