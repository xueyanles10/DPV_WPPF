import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import curve_fit

# 指定.dat文件的路径
file_path = 'si_1.dat'

# 用 read_csv 读取数据，以空格分隔
xrd_data = pd.read_csv(file_path, delim_whitespace=True, comment='#', header=None, names=['2Theta', 'Intensity'])

# 绘制 XRD 图谱
plt.plot(xrd_data['2Theta'], xrd_data['Intensity'])
plt.title('XRD Pattern')
plt.xlabel('2Theta')
plt.ylabel('Intensity')

# 平滑数据
window_length = 11  # 滑动窗口的长度
polyorder = 3  # 多项式拟合的次数
smoothed_intensity = savgol_filter(xrd_data['Intensity'].values, window_length=window_length, polyorder=polyorder)

# 调整峰的寻找参数
threshold_intensity = 1500  # 峰的高度阈值
distance_min = 50  # 两个峰之间的最小距离
prominence_threshold = 0.10  # 峰的突出度阈值
width_range = (0.1, 100)  # 期望的峰的宽度范围

# 找到衍射峰的位置
peaks, _ = find_peaks(smoothed_intensity, height=threshold_intensity, distance=distance_min, prominence=prominence_threshold, width=width_range)

# 在图上标记峰的位置
plt.plot(xrd_data['2Theta'].values[peaks], smoothed_intensity[peaks], 'ro', label='Peaks')

from scipy.optimize import curve_fit

def double_pseudo_voigt_fun(x, ceit_1, fwhm_1, alpha_1, area_1, ceit_2, fwhm_2, alpha_2):
    """定义双峰伪Voigt函数"""
    gaussian_1 = 1 / (fwhm_1 * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - ceit_1) / fwhm_1) ** 2)
    lorenzian_1 = 1 / np.pi * (fwhm_1 / ((x - ceit_1) ** 2 + fwhm_1 ** 2))
    pseudo_voigt_1 = ((1 - alpha_1) * gaussian_1 + alpha_1 * lorenzian_1) * area_1

    area_2 = area_1/2
    gaussian_2 = 1 / (fwhm_2 * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - ceit_2) / fwhm_2) ** 2)
    lorenzian_2 = 1 / np.pi * (fwhm_2 / ((x - ceit_2) ** 2 + fwhm_2 ** 2))
    pseudo_voigt_2 = ((1 - alpha_2) * gaussian_2 + alpha_2 * lorenzian_2) * area_2

    double_pseudo_voigt = pseudo_voigt_1 + pseudo_voigt_2

    return double_pseudo_voigt

# 初始参数猜测
initial_guess = [0, 0.2, 0.5, 2000, 0, 0.2, 0.5]

def fit_all_double_peaks(x, y, peaks, width_guess, maxfev=200000):
    """Fit all double pseudo-Voigt peaks"""
    params_list = []

    for i, peak_index in enumerate(peaks):
        # Create a copy of the initial_guess list
        current_guess = initial_guess.copy()
        
        # Update the relevant elements in the copy
        current_guess[0] = peak_index
        current_guess[4] = np.arcsin(1.0024 * np.sin(peak_index / 2 * np.pi / 180)) * 360 / np.pi

        # Perform curve fitting with the updated guess
        params, covariance = curve_fit(double_pseudo_voigt_fun, x, y, p0=current_guess, maxfev=maxfev)
        params_list.append(params)

    return params_list
    
# 拟合所有双峰伪Voigt峰
width_guess = 0.5  # 用于确定每个峰的拟合宽度
peaks = [28.43, 47.29, 56.11, 69.11, 76.36, 88.01, 94.94]
params_list = fit_all_double_peaks(xrd_data['2Theta'].values, xrd_data['Intensity'].values, peaks, width_guess, maxfev=200000)

# Plot the original XRD pattern
plt.scatter(xrd_data['2Theta'], xrd_data['Intensity'], label='Original Data')

# Plot the smoothed data
plt.plot(xrd_data['2Theta'], smoothed_intensity, label='Smoothed Data')

# Plot the fitted curves
for params in params_list:
    fitted_curve = double_pseudo_voigt_fun(xrd_data['2Theta'].values, *params)
    plt.plot(xrd_data['2Theta'], fitted_curve, '--', label='Fitted Curve')

plt.title('XRD Pattern with Fitted Curves')
plt.xlabel('2Theta')
plt.ylabel('Intensity')
plt.legend()
plt.show()