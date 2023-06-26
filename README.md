# Transmission_Line_Parameter_Estimation
The EGLE algorithm for parameter estimation in presence of non-Gaussian measurements.

Reference: Varghese, A. C., Pal, A., & Dasarathy, G. (2022). Transmission line parameter estimation under non-Gaussian measurement noise. IEEE Transactions on Power Systems.

Objective: Novel optimal parameter estimation algorithm (optimal even when the measurement noise is non-Gaussian)

A general regression problem of the form c = D x is considered.

The working of EGLE in a generic setting is described by the file "EGLE_MAIN.py". Simply run this file to test the performance of EGLE algorithm in a linear system where the noisy measurements available have non-Gaussian characteristics.

The related files that can be useful in running the main file are:
1) EGLE_Generate_Dataset.py
2) EGLE_Create_Noisy_Measurements.py
3) EGLE_Estimation.py
4) EGLE_Performance_comparison.py

These files, along with the MAIN file helps in finding the optimal estimates of the linear system of equations when only the noisy measurements are available. They also help in generating Non Gaussian measurement noise (if required), and comparing the performance with least squares and total least squares estimation methods.

The file EGLE_MWC_to_PS_WC.py demonstrates the use of EGLE for transmission line parameter estimation (TLPE). This file also makes use of functions from the related files mentioned above. The sample data files required for conducting TLPE can be obtained from "https://drive.google.com/drive/folders/1c2_yhQRgpz2i03L7LBnbDNj9cbpV3raJ?usp=sharing".

