from pymatgen.core.structure import Structure
from pymatgen.analysis.diffraction.xrd import XRDCalculator
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks, savgol_filter

def read_cif_and_calculate_xrd(cif_file_path):
    structure = Structure.from_file(cif_file_path, primitive=True)
    xrd_calculator = XRDCalculator()
    pattern = xrd_calculator.get_pattern(structure)
    return pattern

def plot_calculated_xrd(pattern):
    plt.figure(figsize=(16, 12))
    for i, peak in enumerate(pattern.hkls):
        hkl_str = ", ".join([f"{hkl[0]}{hkl[1]}" for hkl in peak[0]])
        label = f"{hkl_str}, d={pattern.d_hkls[i]:.2f} Å"
        plt.vlines(pattern.x[i], 0, pattern.y[i]*500,  color='red', label=label)

    plt.xlabel('2θ (degrees)')
    plt.ylabel('Intensity (a.u.)')
    plt.grid(True)
    plt.legend()
    plt.show()

def read_and_plot_experimental_data(file_path):
    xrd_data = pd.read_csv(file_path, delim_whitespace=True, comment='#', header=None, names=['2Theta', 'Intensity'])
    plt.scatter(xrd_data['2Theta'], xrd_data['Intensity'])
    plt.xlabel('2θ (degrees)')
    plt.ylabel('Intensity (a.u.)')
    plt.grid(True)
    plt.legend()
    plt.show()
    return xrd_data

def smooth_and_identify_peaks(xrd_data, window_length=50, polyorder=5, threshold_intensity=2000, distance_min=500, prominence_threshold=10.0, width_range=(30, 100)):
    smoothed_intensity = savgol_filter(xrd_data['Intensity'].values, window_length=window_length, polyorder=polyorder)
    peaks, _ = find_peaks(smoothed_intensity, height=threshold_intensity, distance=distance_min, prominence=prominence_threshold, width=width_range)
    plt.scatter(xrd_data['2Theta'], xrd_data['Intensity'])
    plt.plot(xrd_data['2Theta'].values[peaks], smoothed_intensity[peaks], 'ro', label='Peaks')
    plt.xlabel('2θ (degrees)')
    plt.ylabel('Intensity (a.u.)')
    plt.grid(True)
    plt.legend()
    plt.show()

# Example Usage:
cif_file_path = "1.cif"
pattern = read_cif_and_calculate_xrd(cif_file_path)
plot_calculated_xrd(pattern)

experimental_file_path = '1.dat'
experimental_data = read_and_plot_experimental_data(experimental_file_path)

smooth_and_identify_peaks(experimental_data)