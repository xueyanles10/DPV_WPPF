from pymatgen.core.structure import Structure
from pymatgen.analysis.diffraction.xrd import XRDCalculator
import matplotlib.pyplot as plt

# Step 1: Read CIF file
cif_file_path = "1.cif"
structure = Structure.from_file(cif_file_path, primitive=True)
# Step 2: Set up XRD calculator
xrd_calculator = XRDCalculator()

# Step 3: Calculate XRD pattern
pattern = xrd_calculator.get_pattern(structure)

# Step 4: Plot XRD pattern with individual peaks

for i, peak in enumerate(pattern.hkls):
    hkl_str = ", ".join([f"{hkl[0]}{hkl[1]}" for hkl in peak[0]])
    d_spacing = pattern.d_hkls[i]
    intensity = pattern.y[i]
    print(f"Peak {i + 1}: 2θ = {pattern.x[i]:.2f} degrees, hkl = {hkl_str}, d-spacing = {d_spacing:.2f} Å, Intensity = {intensity:.2f}")

import pandas as pd

# 指定.dat文件的路径
file_path = '1.dat'

# 用 read_csv 读取数据，以空格分隔
xrd_data = pd.read_csv(file_path, delim_whitespace=True, comment='#', header=None, names=['2Theta', 'Intensity'])

plt.figure(figsize=(16, 12))
for i, peak in enumerate(pattern.hkls):
    hkl_str = ", ".join([f"{hkl[0]}{hkl[1]}" for hkl in peak[0]])
    label = f"{hkl_str}, d={pattern.d_hkls[i]:.2f} Å"
    plt.vlines(pattern.x[i], 0, pattern.y[i]*500,  color='red', label=label)

# 绘制 XRD 图谱
plt.scatter(xrd_data['2Theta'], xrd_data['Intensity'])
plt.xlabel('2θ (degrees)')
plt.ylabel('Intensity (a.u.)')
plt.grid(True)
plt.show()

from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import curve_fit
window_length = 50  # 滑动窗口的长度
polyorder = 5  # 多项式拟合的次数
smoothed_intensity = savgol_filter(xrd_data['Intensity'].values, window_length=window_length, polyorder=polyorder)

# 调整峰的寻找参数
threshold_intensity = 2000  # 峰的高度阈值
distance_min = 500  # 两个峰之间的最小距离
prominence_threshold = 10.0  # 峰的突出度阈值
width_range = (30, 100)  # 期望的峰的宽度范围

# 找到衍射峰的位置
peaks, _ = find_peaks(smoothed_intensity, height=threshold_intensity, distance=distance_min, prominence=prominence_threshold, width=width_range)

# 在图上标记峰的位置
plt.scatter(xrd_data['2Theta'], xrd_data['Intensity'])
plt.plot(xrd_data['2Theta'].values[peaks], smoothed_intensity[peaks], 'ro', label='Peaks')


import numpy as np

# Extract X and Y values of the peaks for background fitting
x_peaks = xrd_data['2Theta'].values[peaks]
y_peaks = smoothed_intensity[peaks]

# Perform polynomial fit
degree = 3  # Degree of the polynomial fit, adjust as needed
background_coeffs = np.polyfit(x_peaks, y_peaks, degree)

# Generate the background using the fitted coefficients
background = np.polyval(background_coeffs, xrd_data['2Theta'])

# Subtract the background from the original intensity
background_subtracted_intensity = xrd_data['Intensity'] - background

# Plot the background-subtracted XRD pattern
plt.figure(figsize=(16, 12))
plt.plot(xrd_data['2Theta'], background_subtracted_intensity, label='Background Subtracted', linewidth=2)

# Plot the peaks on top of the background-subtracted pattern
plt.scatter(xrd_data['2Theta'].values[peaks], background_subtracted_intensity.values[peaks], c='red', marker='o', label='Peaks')

plt.xlabel('2θ (degrees)')
plt.ylabel('Intensity (a.u.)')
plt.grid(True)
plt.legend()
plt.show()