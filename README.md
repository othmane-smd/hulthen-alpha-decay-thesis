# Alpha Decay - Hulthén WKB Model

This repository contains the Python codes developed during
my Master's thesis in nuclear physics.

The project reproduces and extends the results of:

Budaca & Budaca,
Alpha decay half-lives calculated with the Hulthén potential.

Repository structure:

original_analysis/
    fit_parameters.py
    extract_screening_parameter.py
    isotopic_chain_analysis.py

coulomb_validity/
    fit_parameters_validity.py
    extract_screening_parameter_validity.py
    isotopic_chain_analysis_validity.py

Notes:
- Nuclear datasets are directly embedded in the scripts.
- The numerical calculations were performed in Python.
The final figures included in the thesis were generated
in Mathematica using the exported numerical data.
