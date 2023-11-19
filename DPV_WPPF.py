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

def double_pseudo_voigt_fun(x, ceit_1, fwhm_1, alpha_1, area_1, ceit_2, fwhm_2, alpha_2, area_2):
    """定义双峰伪Voigt函数"""
    gaussian_1 = 1 / (fwhm_1 * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - ceit_1) / fwhm_1) ** 2)
    lorenzian_1 = 1 / np.pi * (fwhm_1 / ((x - ceit_1) ** 2 + fwhm_1 ** 2))
    pseudo_voigt_1 = ((1 - alpha_1) * gaussian_1 + alpha_1 * lorenzian_1) * area_1

    gaussian_2 = 1 / (fwhm_2 * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - ceit_2) / fwhm_2) ** 2)
    lorenzian_2 = 1 / np.pi * (fwhm_2 / ((x - ceit_2) ** 2 + fwhm_2 ** 2))
    pseudo_voigt_2 = ((1 - alpha_2) * gaussian_2 + alpha_2 * lorenzian_2) * area_2

    double_pseudo_voigt = pseudo_voigt_1 + pseudo_voigt_2

    return double_pseudo_voigt

# 初始参数猜测
initial_guess = [0, 0.2, 0.5, 0, 0, 0.2, 0.5, 0]

def fit_all_double_peaks(x, y, peaks, width_guess, maxfev=200000):
    """拟合所有双峰伪Voigt峰"""
    params_list = []

    for peak_index in peaks:
        # Ensure the peak index is within the valid range
        if peak_index - width_guess//2 >= 0 and peak_index + width_guess//2 < len(x):
            x_peak = x[peak_index - width_guess//2 : peak_index + width_guess//2]
            y_peak = y[peak_index - width_guess//2 : peak_index + width_guess//2]

            # Check if the number of data points is sufficient for fitting
            if len(x_peak) > len(initial_guess):
                # 更新初始参数猜测
                initial_guess[0] = x_peak[np.argmax(y_peak)]  # 使用峰的位置作为第一个峰的中心
                initial_guess[3] = y_peak.sum()  # 使用峰的总面积作为第一个峰的初始猜测值
                initial_guess[4] = np.arcsin(1.0024396183602313 * np.sin(x_peak[np.argmax(y_peak)]/2*np.pi/180))*360/np.pi
                initial_guess[7] = y_peak.sum()/2  # 使用峰的总面积的一半作为第二个峰的初始猜测值

                # 使用 curve_fit 进行峰拟合
                try:
                    params, covariance = curve_fit(double_pseudo_voigt_fun, x_peak, y_peak, p0=initial_guess, maxfev=maxfev)
                    params_list.append(params)
                except Exception as e:
                    print(f"Error fitting double peak at index {peak_index}: {e}")
            else:
                print(f"Skipping peak at index {peak_index} due to insufficient data points for fitting")
        else:
            print(f"Skipping peak at index {peak_index} due to insufficient data range")

    return params_list

# 拟合所有双峰伪Voigt峰
width_guess = 3  # 用于确定每个峰的拟合宽度
params_list = fit_all_double_peaks(xrd_data['2Theta'].values, smoothed_intensity, peaks, width_guess, maxfev=200000)

# 用拟合后的峰来重构整个曲线
fitted_curve = np.zeros_like(xrd_data['Intensity'].values)
for peak_index, params in zip(peaks, params_list):
    fitted_curve[peak_index - width_guess//2 : peak_index + width_guess//2] += double_pseudo_voigt_fun(xrd_data['2Theta'].values[peak_index - width_guess//2 : peak_index + width_guess//2], *params)

# 双峰伪Voigt减除
corrected_intensity = xrd_data['Intensity'].values - fitted_curve

# 存储拟合后的参数
fitted_params_list = []

# 输出拟合后的角度、FWHM_1、Alpha_1 和 Area_1
for i, params in enumerate(params_list):
    fitted_params_list.append({
        'Angle': params[0],
        'FWHM_1': params[1],
        'Alpha_1': params[2],
        'Area_1': params[3],
    })
    print(f"Peak {i+1} - Angle: {params[0]}, FWHM_1: {params[1]}, Alpha_1: {params[2]}, Area_1: {params[3]}")

# 显示图谱
plt.scatter(xrd_data['2Theta'], xrd_data['Intensity'])
plt.plot(xrd_data['2Theta'].values, corrected_intensity, 'r-')
plt.show()
