# Optimal Epidemic Control with Mitigation–Suppression Strategy

This repository contains the numerical implementation accompanying our study on optimal short-term non-pharmaceutical interventions (NPIs) using a multi-objective optimization framework. The work focuses on balancing two key epidemic outcomes:

- **Infected Peak Prevalence (IPP)**  
- **Epidemic Final Size (EFS)**  

under fixed-duration interventions and healthcare capacity constraints.

---

## Overview

Designing effective NPIs involves a fundamental trade-off:

- Early interventions reduce peak infections (IPP) but may increase total infections (EFS) due to rebound effects.  
- Later interventions reduce EFS but may lead to higher infection peaks, placing stress on healthcare systems.  

To address this, we introduce:

- A **Total Burden** objective combining IPP and EFS  
- A two-phase **Mitigation–Suppression (MS)** strategy  
- A feasibility framework linking intervention design with healthcare capacity  

---

## Repository Structure

### 1. `V_shape.py`
- Computes optimal fixed-intensity strategies for:
  - IPP minimization  
  - EFS minimization  
- Produces the characteristic **V-shaped trade-off curve**
- Visualizes:
  - Optimal outcomes vs intervention timing  
  - Corresponding optimal control intensities  

---

### 2. `Quantify_Conflict_Scinario.py`
- Quantifies the conflict between IPP and EFS  
- Computes percentage deviation:
  - EFS under IPP-optimal control  
  - IPP under EFS-optimal control  
- Analyzes how conflict varies with intervention duration (\(\tau\))  

---

### 3. `Total_Burden_under_MS_control.py`
- Core implementation of the multi-objective optimization framework  
- Defines the **Total Burden** objective  
- Implements the **Mitigation–Suppression (MS)** strategy  
- Computes:
  - Optimal control parameters  
  - Epidemic trajectories under MS control  

---

### 4. `Feasibility_satisfying_peak_threshold.py`
- Identifies feasible weight combinations under a peak-capacity constraint  
- Determines which strategies satisfy:
  - \( I_{\max} \leq I_{\text{threshold}} \)  
- Characterizes the feasible trade-off region  

---

### 5. `Policy_Advantages.py`
- Demonstrates advantages of MS over fixed-intensity strategies  
- Shows that:
  - Early MS strategies can simultaneously reduce both IPP and EFS  

---

## Requirements

- Python 3.x  
- NumPy  
- SciPy  
- Matplotlib  

Install dependencies using:

```bash
pip install numpy scipy matplotlib
