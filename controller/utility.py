# from scipy.optimize import curve_fit
# from scipy.integrate import quad
"""Module for helper functions of the system SeGaSA.

    Also contains function prototypes.
"""
import numpy as np

#
# def slope(y1, y2):
#     return y2 - y1


# def find_peak_start(data, max_x, threshold):
#     for i in range(2, len(data)):
#         if (max_x - i > 0 and (slope(data[max_x - i - 1], data[max_x - i]) < threshold)):
#             return [max_x - i, data[max_x - i]]
#     return [0, 0]
#
#
# def find_peak_end(data, max_x, threshold):
#     for i in range(2, len(data)):
#         if (max_x + i + 1 < len(data) and slope(data[max_x + i], data[max_x + i + 1]) > threshold):
#             return [max_x + i, data[max_x + i]]
#     return [0, 0]


def linear_model(x, a, b):
    return a * x + b


def gaussian_model(x, a, mu, sig):
    """A funciton used to calculate gaussian parameters

    PARAMETERS
        a : float
            -- gaussian height
        mu : float
            -- mean (centroid in our case)
        sig : float
            --
    """
    return a * np.exp(-(x - mu) ** 2 / (2 * sig ** 2))


# def bigaussian_lateral(x, base, height, center, width1):
#     """fitting function used for a single side of the peak"""
#     return base + height * np.exp(-0.5 * ((x - center) / width1) ** 2)

def estimate_centroid(data, max_channel):
    """Determines peak position by the five-channel method

    PARAMETERS:
        data : list of integers
        max_channel: index of the channel containing maximum amount of counts
    """
    max_count = data[max_channel]
    sub1 = data[max_channel - 1]
    sub2 = data[max_channel - 2]
    sup1 = data[max_channel + 1]
    sup2 = data[max_channel + 2]

    centroid = max_channel + (sup1 * (max_count - sub2) - sub1 * (max_count - sup2)) \
        / (sup1 * (max_count - sub2) + sub1 * (max_count - sup2))

    return centroid


# def estimate_height(data, max_channel):
#     max_count = data[max_channel]
#     sub1 = data[max_channel - 1]
#     sub2 = data[max_channel - 2]
#     sup1 = data[max_channel + 1]
#     sup2 = data[max_channel + 2]
#
#     prefit_y = [sub1, sub2, max_count, sup1, sup2]
#     prefit_x = list(range(0, len(prefit_y)))
#     gaussian_parameters, _ = curve_fit(gaussian_model, prefit_x, prefit_y)
#     height, mean, variance = gaussian_parameters
#     return height

def estimate_fwhm(data, max_channel, k):
    height = data[max_channel]
    sub1 = 0
    sub1_index = 0
    sub2 = 0
    sup1 = 0
    sup1_index = 0
    sup2 = 0

    for i in range(len(data)):

        if data[max_channel - i] > (k * height) > data[max_channel - i - 1]:
            sub1 = data[max_channel - i - 1]
            sub1_index = max_channel - i - 1
            sub2 = data[max_channel - i]
            break

    for i in range(len(data)):
        if data[max_channel + i] > (k * height) > data[max_channel + i + 1]:
            sup1 = data[max_channel + i]
            sup1_index = max_channel + i
            sup2 = data[max_channel + i + 1]
            break

    fwhm = sup1_index - sub1_index + ((sup1 - 0.5 * height) / (sup1 - sup2) - (0.5 * height - sub1) / (sub2 - sub1))

    return fwhm


# def estimate_FWHM_gaussian(data, max_channel):
#     height = estimate_height(data, max_channel)
#     sub1 = 0
#     sub1_index = 0
#     sub2 = 0
#     sup1 = 0
#     sup1_index = 0
#     sup2 = 0
#
#     for i in range(len(data)):
#
#         if data[max_channel - i] > (0.5 * height) > data[max_channel - i - 1]:
#             sub1 = data[max_channel - i - 1]
#             sub1_index = max_channel - i - 1
#             sub2 = data[max_channel - i]
#             break
#
#     for i in range(len(data)):
#         if data[max_channel + i] > (0.5 * height) > data[max_channel + i + 1]:
#             sup1 = data[max_channel + i]
#             sup1_index = max_channel + i
#             sup2 = data[max_channel + i + 1]
#             break
#
#     fwhm = sup1_index - sub1_index + ((sup1 - 0.5 * height) / (sup1 - sup2) - (0.5 * height - sub1) / (sub2 - sub1))
#
#     return fwhm


# def estimate_background_area(data, max_channel, fwhm):
#     background_start = int(max_channel - (fwhm * 3) / 2)
#     background_end = int((fwhm * 3) / 2 + max_channel)
#
#     background_width = background_end - background_start
#
#     background_area = (data[background_start] + data[background_end]) * background_width / 2
#
#     return [background_start, background_end, background_area]
#

# def estimate_gaussian_area(start, end, a, mu, sig):
#     area = quad(gaussian_model, start, end, (a, mu, sig))
#     return area


def r_squared_linear(x, y, a, b):
    x = np.array(x)
    y = np.array(y)
    residuals = (y - (linear_model(x, a, b)))
    total = ((y - np.mean(y)) ** 2)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum(total)
    return 1 - (ss_res / ss_tot)


def get_max_channel(measurement_data, peak):
    int_peak = int(peak)
    index = -1
    data = np.array(measurement_data)

    if len(measurement_data) > int_peak + 10 and int_peak - 10 > 0:
        value = max(data[int_peak - 10:int_peak + 10])
        data[0:int_peak - 10] = [0] * (int_peak - 10)
        data[int_peak + 11:len(data) - 1] = [0] * (len(data) - 1 - (int_peak + 11))
        index = data.tolist().index(value)

    return index


# def compare_neighbours(left, right, threshold, value):
#     if left + threshold < value or left - threshold > value:
#         print("nok", value, np.mean([left, right]))
#         return np.mean([left, right])
#     if right + threshold < value or right - threshold > value:
#         print("nok", value, np.mean([left, right]))
#         return np.mean([left, right])
#     print(value)
#     return value
