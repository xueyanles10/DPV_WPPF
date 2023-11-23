import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import curve_fit

# Specify the path to the .dat file
file_path = 'si_1.dat'

# Read data using read_csv, separated by spaces
xrd_data = pd.read_csv(file_path, delim_whitespace=True, comment='#', header=None, names=['2Theta', 'Intensity'])

# Plot XRD pattern
plt.plot(xrd_data['2Theta'], xrd_data['Intensity'])
plt.title('XRD Pattern')
plt.xlabel('2Theta')
plt.ylabel('Intensity')

# Smooth data
window_length = 11
polyorder = 3
smoothed_intensity = savgol_filter(xrd_data['Intensity'].values, window_length=window_length, polyorder=polyorder)

# Adjust peak finding parameters
threshold_intensity = 1500
distance_min = 50
prominence_threshold = 0.10
width_range = (0.1, 100)

# Find diffraction peak positions
peaks, _ = find_peaks(smoothed_intensity, height=threshold_intensity, distance=distance_min, prominence=prominence_threshold, width=width_range)

# Mark peak positions on the plot
plt.plot(xrd_data['2Theta'].values[peaks], smoothed_intensity[peaks], 'ro', label='Peaks')

from scipy.optimize import curve_fit

def double_pseudo_voigt_with_background(x, ceit_1, fwhm_1, alpha_1, area_1, ceit_2, fwhm_2, alpha_2, background_slope, background_intercept):
    """Define double pseudo-Voigt function with linear background"""
    gaussian_1 = 1 / (fwhm_1 * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - ceit_1) / fwhm_1) ** 2)
    lorenzian_1 = 1 / np.pi * (fwhm_1 / ((x - ceit_1) ** 2 + fwhm_1 ** 2))
    pseudo_voigt_1 = ((1 - alpha_1) * gaussian_1 + alpha_1 * lorenzian_1) * area_1

    area_2 = area_1 / 2
    gaussian_2 = 1 / (fwhm_2 * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - ceit_2) / fwhm_2) ** 2)
    lorenzian_2 = 1 / np.pi * (fwhm_2 / ((x - ceit_2) ** 2 + fwhm_2 ** 2))
    pseudo_voigt_2 = ((1 - alpha_2) * gaussian_2 + alpha_2 * lorenzian_2) * area_2

    double_pseudo_voigt = pseudo_voigt_1 + pseudo_voigt_2

    # Linear background
    background = background_slope * x + background_intercept

    return double_pseudo_voigt + background

# Initial parameter guess
initial_guess = [0, 0.2, 0.5, 2000, 0, 0.2, 0.5, 0, 0]

def fit_all_double_peaks_with_background(x, y, peaks, width_guess, maxfev=200000):
    """Fit all double pseudo-Voigt peaks with background"""
    params_list = []

    for i, peak_index in enumerate(peaks):
        # Create a copy of the initial_guess list
        current_guess = initial_guess.copy()

        # Update relevant elements in the copy
        current_guess[0] = peak_index
        current_guess[4] = np.arcsin(1.0024 * np.sin(peak_index / 2 * np.pi / 180)) * 360 / np.pi

        # Perform curve fitting with the updated guess
        params, covariance = curve_fit(double_pseudo_voigt_with_background, x, y, p0=current_guess, maxfev=maxfev)
        params_list.append(params)

    return params_list

# Fit all double pseudo-Voigt peaks with background
width_guess = 0.5
params_list = fit_all_double_peaks_with_background(xrd_data['2Theta'].values, xrd_data['Intensity'].values, peaks, width_guess, maxfev=200000)

# Plot the original XRD pattern
plt.scatter(xrd_data['2Theta'], xrd_data['Intensity'], label='Original Data')

# # Plot the smoothed data
# plt.plot(xrd_data['2Theta'], smoothed_intensity, label='Smoothed Data')

# Plot the original XRD pattern
plt.scatter(xrd_data['2Theta'], xrd_data['Intensity'], label='Original Data')

# Plot the fitted curves for each peak
for i, peak_index in enumerate(peaks):
    # Extract the parameters for the current peak
    params = params_list[i]

    # Generate x values for the fitted curve (you may want a finer grid)
    x_fit = np.linspace(peak_index - width_guess, peak_index + width_guess, 100)

    # Calculate the corresponding y values using the fitted parameters
    y_fit = double_pseudo_voigt_with_background(x_fit, *params)

    # Plot the fitted curve for the current peak
    plt.plot(x_fit, y_fit, label=f'Fitted Peak {i + 1}')

# Add legend to the plot
plt.legend()

# Show the plot
plt.show()