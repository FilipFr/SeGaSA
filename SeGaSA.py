import os
import sys
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
from matplotlib.widgets import RectangleSelector
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, peak_widths
from scipy.integrate import quad
import math

import sympy as sp
from sympy.abc import n
import msvcrt as m


"""
    vyskusat vykreslit aj total count
    Vykreslovanie parametrov v podskupinach dat.
    Pozriet sa na moznost vykreslenia iba uzivatelom zvolenej metody, pripadne 
    !!OVERIT GRAFICKY!! FWHM - nedavat na nulu, robit miesto toho priemer z najblizsich hodnot, doplnit o jednoduche odcitavanie hodnot z grafu v polovici
    !! Centroid - prelozit gaussian na 5 bodov s maximom v strede a odcitat centroid
    Pozriet sa na posledne merania aby lepsie urcovalo centroid a koniec/zaciatok peaku (skusit prelozit gaussian/bigaussian)
"""

LOADED = 0
WINDOW_SIZE = 1
OFFSET = 1


def slope(y1, y2):
    return y2-y1

def wait():
    m.getch()

def find_peak_start(data, max_x, threshold):
    for i in range(2, len(data)):
        if(max_x-i > 0 and (slope(data[max_x-i-1], data[max_x-i]) < threshold)):
            return [max_x-i, data[max_x-i]]
    return [0, 0]

def find_peak_end(data, max_x, threshold):
    for i in range(2, len(data)):
        if(max_x+i+1 < len(data) and slope(data[max_x+i], data[max_x+i+1]) > threshold):
            return [max_x+i, data[max_x+i]]
    return [0, 0]

def linear_model(x, a, b):
    return a * x + b

def gaussian_model(x, a, mu, sig):
    return a * np.exp(-(x - mu) ** 2 / (2 * sig ** 2))

def bigaussian_lateral(x, base, height, center, width1):
    return base + height * np.exp(-0.5 * ((x - center) / width1) ** 2)

def bigaussian_model(x, base, height, center, width1, width2):
    if x<center:
        return base + height * np.exp(-0.5*((x-center)/width1)**2)
    else:
        return base + height * np.exp(-0.5*((x-center)/width2)**2)

def estimate_centroid(data, max_channel):
    """Determines peak position by the five-channel method

    :param list data: list of signal counts
    :param int max_channel: local maximum count channel
    :return float: peak center position, centroid
    """
    max_count = data[max_channel]
    sub1 = data[max_channel - 1]
    sub2 = data[max_channel - 2]
    sup1 = data[max_channel + 1]
    sup2 = data[max_channel + 2]

    centroid = max_channel + (sup1 * (max_count - sub2) - sub1 * (max_count - sup2)) \
               / (sup1 * (max_count - sub2) + sub1 * (max_count - sup2))

    return centroid


def estimate_height(data, max_channel):

    max_count = data[max_channel]
    sub1 = data[max_channel - 1]
    sub2 = data[max_channel - 2]
    sup1 = data[max_channel + 1]
    sup2 = data[max_channel + 2]

    prefit_y = [sub1, sub2, max_count, sup1, sup2]
    prefit_x = list(range(0, len(prefit_y)))
    gaussian_parameters, _ = curve_fit(gaussian_model, prefit_x, prefit_y)
    height, mean, variance = gaussian_parameters
    return height


def estimate_FWHM(data, max_channel, K):
    height = data[max_channel]
    sub1 = 0
    sub1_index = 0
    sub2 = 0
    sup1 = 0
    sup1_index = 0
    sup2 = 0

    for i in range(len(data)):

        if data[max_channel - i] > (K * height) > data[max_channel - i - 1]:
            sub1 = data[max_channel - i - 1]
            sub1_index = max_channel - i-1
            sub2 = data[max_channel - i]
            break

    for i in range(len(data)):
        if data[max_channel + i] > (K * height) > data[max_channel + i + 1]:
            sup1 = data[max_channel + i]
            sup1_index = max_channel + i
            sup2 = data[max_channel + i + 1]
            break

    fwhm = sup1_index - sub1_index + ((sup1 - 0.5 * height) / (sup1 - sup2) - (0.5 * height - sub1) / (sub2 - sub1))

    return fwhm

def estimate_FWHM_gaussian(data, max_channel):
    height = estimate_height(data, max_channel)
    sub1 = 0
    sub1_index = 0
    sub2 = 0
    sup1 = 0
    sup1_index = 0
    sup2 = 0

    for i in range(len(data)):

        if data[max_channel - i] > (0.5 * height) > data[max_channel - i - 1]:
            sub1 = data[max_channel - i - 1]
            sub1_index = max_channel - i-1
            sub2 = data[max_channel - i]
            break

    for i in range(len(data)):
        if data[max_channel + i] > (0.5 * height) > data[max_channel + i + 1]:
            sup1 = data[max_channel + i]
            sup1_index = max_channel + i
            sup2 = data[max_channel + i + 1]
            break

    fwhm = sup1_index - sub1_index + ((sup1 - 0.5 * height) / (sup1 - sup2) - (0.5 * height - sub1) / (sub2 - sub1))

    return fwhm

def estimate_background_area(data, max_channel, fwhm):
    background_start = int(max_channel - (fwhm*3)/2)
    background_end = int((fwhm*3)/2 + max_channel)

    background_width = background_end - background_start

    background_area = (data[background_start] + data[background_end]) * (background_width) / 2

    return [background_start, background_end, background_area]

def estimate_gaussian_area(start, end, a, mu, sig):
    area = quad(gaussian_model, start, end, (a, mu, sig))
    return area



def r_squared_linear(x, y, a, b):
    x = np.array(x)
    y = np.array(y)
    residuals = (y - (linear_model(x, a, b)))
    total = ((y - np.mean(y)) ** 2)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum(total)
    return 1 - (ss_res / ss_tot)

def get_max_channel(measurementData, peak):

    intPeak = int(peak)
    index = -1
    data = np.array(measurementData)

    if(len(measurementData) > intPeak+10 and intPeak-10 > 0):
        value = max(data[intPeak-10:intPeak+10])
        data[0:intPeak-10] = [0] * (intPeak-10)
        data[intPeak+11:len(data)-1] = [0] * (len(data)-1-(intPeak+11))
        index = data.tolist().index(value)

    return index

def compare_neighbours(left, right, threshold, value):
    if (left + threshold < value or left - threshold > value):
        print("nok", value, np.mean([left,right]))
        return np.mean([left, right])
    if (right + threshold < value or right - threshold > value):
        print("nok", value, np.mean([left, right]))
        return np.mean([left, right])
    print(value)
    return value


class Ui_MCAData(object):

    def setupUi(self, MCAData):
        self.data = ParsedDataFromMca()
        self.merged = []
        self.alteredDataset = []
        self.alteredDatasetPeaks = []
        self.resultCentroids = []
        self.resultFWHM = []
        self.resultGaussCentroid = []
        self.resultCorrectedFWHM = []
        self.tempBoundaries = []
        self.resultGrossArea = []
        self.areaCount = []
        self.boundaries = []
        self.boundaries = [[217.88306451612902, 240.2016129032258], [217.33870967741933, 240.74596774193543], [216.79435483870964, 240.74596774193543], [218.42741935483866, 240.2016129032258], [216.79435483870964, 239.6572580645161], [217.88306451612902, 239.6572580645161], [216.79435483870964, 241.29032258064512], [214.61693548387092, 240.74596774193543], [214.61693548387092, 240.2016129032258], [217.88306451612902, 240.2016129032258], [214.61693548387092, 241.29032258064512], [215.1612903225806, 239.6572580645161], [214.61693548387092, 240.74596774193543], [214.07258064516128, 241.29032258064512], [214.07258064516128, 239.6572580645161], [214.07258064516128, 240.2016129032258], [210.80645161290317, 240.74596774193543], [212.9838709677419, 240.2016129032258], [209.17338709677415, 239.1129032258064], [206.99596774193543, 239.1129032258064], [202.09677419354836, 240.74596774193543], [202.09677419354836, 240.2016129032258], [195.5645161290322, 239.6572580645161], [202.09677419354836, 240.74596774193543], [191.7540322580645, 239.1129032258064], [191.7540322580645, 239.6572580645161], [191.7540322580645, 239.6572580645161], [192.84274193548384, 240.2016129032258], [201.55241935483866, 239.6572580645161], [203.1854838709677, 238.5685483870967], [201.55241935483866, 240.2016129032258], [197.74193548387092, 239.6572580645161], [197.19758064516128, 240.2016129032258], [196.1088709677419, 238.5685483870967], [197.74193548387092, 239.6572580645161], [196.6532258064516, 238.5685483870967], [190.66532258064512, 239.1129032258064], [182.5, 236.9354838709677], [183.58870967741933, 234.75806451612902], [180.86693548387098, 236.9354838709677], [165.0806451612903, 234.75806451612902], [181.4112903225806, 232.0362903225806], [180.86693548387098, 229.8588709677419], [158.5483870967742, 229.3145161290322], [184.67741935483872, 227.13709677419354], [161.27016129032256, 224.41532258064512], [166.71370967741933, 226.59274193548384], [171.61290322580646, 224.95967741935482], [166.16935483870964, 227.13709677419354], [166.16935483870964, 221.14919354838707], [158.0040322580645, 218.42741935483866], [152.01612903225805, 218.42741935483866], [152.56048387096774, 215.1612903225806], [154.19354838709677, 218.42741935483866], [153.10483870967738, 212.4395161290322], [158.5483870967742, 212.4395161290322], [153.64919354838707, 213.5282258064516], [158.0040322580645, 209.17338709677415], [162.3588709677419, 210.80645161290317], [167.80241935483872, 209.17338709677415]]

        self.peakStart = []
        self.peakEnd = []
        self.picked = [0, 0, 0, 0, 0, 0]
        self.analyzedPeak = -1
        self.calibratedArray = []
        self.calibrationLock = False
        self.calibrationTangent = 1
        self.calibrationOffset = 0
        self.logScale = False
        self.peak_counter = 0
        self.fig, self.ax = plt.subplots()

        MCAData.setObjectName("MCAData")
        MCAData.resize(711, 520)
        self.SubsetSize = QtWidgets.QSlider(MCAData)
        self.SubsetSize.setGeometry(QtCore.QRect(440, 430, 211, 22))
        self.SubsetSize.setOrientation(QtCore.Qt.Horizontal)
        self.SubsetSize.setObjectName("SubsetSize")
        self.SubsetSize.setMinimum(1)
        self.SubsetSize.setSingleStep(1)
        self.SubsetSize.setMaximum(1)
        self.DataOffset = QtWidgets.QSlider(MCAData)
        self.DataOffset.setGeometry(QtCore.QRect(440, 470, 211, 22))
        self.DataOffset.setOrientation(QtCore.Qt.Horizontal)
        self.DataOffset.setObjectName("DataOffset")
        self.DataOffset.setMinimum(1)
        self.DataOffset.setSingleStep(1)
        self.DataOffset.setMaximum(1)
        self.SubsetSizeLabel = QtWidgets.QLabel(MCAData)
        self.SubsetSizeLabel.setGeometry(QtCore.QRect(50, 430, 321, 20))
        self.SubsetSizeLabel.setObjectName("SubsetSizeLabel")
        self.DataOffsetLabel = QtWidgets.QLabel(MCAData)
        self.DataOffsetLabel.setGeometry(QtCore.QRect(50, 470, 291, 20))
        self.DataOffsetLabel.setObjectName("DataOffsetLabel")
        self.Load = QtWidgets.QPushButton(MCAData)
        self.Load.setGeometry(QtCore.QRect(40, 30, 141, 31))
        self.Load.setObjectName("Load")

        self.logSwitch = QtWidgets.QPushButton(MCAData)
        self.logSwitch.setGeometry(QtCore.QRect(190, 30, 141, 31))
        self.logSwitch.setObjectName("logSwitch")
        self.logSwitch.setEnabled(False)
        self.setBoundaries = QtWidgets.QPushButton(MCAData)
        self.setBoundaries.setGeometry(QtCore.QRect(340, 30, 141, 31))
        self.setBoundaries.setObjectName("setBoundaries")
        self.setBoundaries.setEnabled(False)
        self.label = QtWidgets.QLabel(MCAData)
        self.label.setGeometry(QtCore.QRect(450, 88, 211, 16))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(MCAData)
        self.label_2.setGeometry(QtCore.QRect(640, 88, 21, 16))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(MCAData)
        self.label_3.setGeometry(QtCore.QRect(400, 430, 21, 16))
        self.label_3.setObjectName("label_3")
        self.label_5 = QtWidgets.QLabel(MCAData)
        self.label_5.setGeometry(QtCore.QRect(400, 470, 21, 21))
        self.label_5.setObjectName("label_5")

        self.calibration = QtWidgets.QPushButton(MCAData)
        self.calibration.setObjectName(u"calibration")
        self.calibration.setGeometry(QtCore.QRect(180, 80, 93, 28))
        self.calibration.setEnabled(False)
        self.label_6 = QtWidgets.QLabel(MCAData)
        self.label_6.setObjectName(u"label_6")
        self.label_6.setGeometry(QtCore.QRect(50, 120, 160, 16))
        self.doubleSpinBox_2 = QtWidgets.QDoubleSpinBox(MCAData)
        self.doubleSpinBox_2.setObjectName(u"doubleSpinBox")
        self.doubleSpinBox_2.setEnabled(False)
        self.doubleSpinBox_2.setGeometry(QtCore.QRect(304, 150, 77, 21))
        self.centroid1 = QtWidgets.QLabel(MCAData)
        self.centroid1.setObjectName(u"centroid1")
        self.centroid1.setGeometry(QtCore.QRect(240, 120, 55, 16))
        self.centroid2 = QtWidgets.QLabel(MCAData)
        self.centroid2.setObjectName(u"centroid2")
        self.centroid2.setGeometry(QtCore.QRect(240, 150, 55, 16))
        self.label_10 = QtWidgets.QLabel(MCAData)
        self.label_10.setObjectName(u"label_10")
        self.label_10.setGeometry(QtCore.QRect(390, 120, 55, 16))
        self.label_11 = QtWidgets.QLabel(MCAData)
        self.label_11.setObjectName(u"label_11")
        self.label_11.setGeometry(QtCore.QRect(390, 150, 55, 16))
        self.label_4 = QtWidgets.QLabel(MCAData)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setGeometry(QtCore.QRect(41, 88, 124, 16))
        self.doubleSpinBox = QtWidgets.QDoubleSpinBox(MCAData)
        self.doubleSpinBox.setObjectName(u"doubleSpinBox_2")
        self.doubleSpinBox.setEnabled(False)
        self.doubleSpinBox.setGeometry(QtCore.QRect(304, 120, 77, 21))
        self.label_7 = QtWidgets.QLabel(MCAData)
        self.label_7.setObjectName(u"label_7")
        self.label_7.setGeometry(QtCore.QRect(50, 150, 85, 16))

        self.label_peak_3 = QtWidgets.QLabel(MCAData)
        self.label_peak_3.setObjectName(u"label_peak_3")
        self.label_peak_3.setGeometry(QtCore.QRect(50, 180, 85, 16))
        self.centroid3 = QtWidgets.QLabel(MCAData)
        self.centroid3.setObjectName(u"centroid3")
        self.centroid3.setGeometry(QtCore.QRect(240, 180, 55, 16))
        self.doubleSpinBox_3 = QtWidgets.QDoubleSpinBox(MCAData)
        self.doubleSpinBox_3.setObjectName(u"doubleSpinBox_3")
        self.doubleSpinBox_3.setEnabled(False)
        self.doubleSpinBox_3.setGeometry(QtCore.QRect(304, 180, 77, 21))
        self.label_kev_3 = QtWidgets.QLabel(MCAData)
        self.label_kev_3.setObjectName(u"label_kev_3")
        self.label_kev_3.setGeometry(QtCore.QRect(390, 180, 55, 16))

        self.label_peak_4 = QtWidgets.QLabel(MCAData)
        self.label_peak_4.setObjectName(u"label_peak_4")
        self.label_peak_4.setGeometry(QtCore.QRect(50, 210, 85, 16))
        self.centroid4 = QtWidgets.QLabel(MCAData)
        self.centroid4.setObjectName(u"centroid4")
        self.centroid4.setGeometry(QtCore.QRect(240, 210, 55, 16))
        self.doubleSpinBox_4 = QtWidgets.QDoubleSpinBox(MCAData)
        self.doubleSpinBox_4.setObjectName(u"doubleSpinBox_4")
        self.doubleSpinBox_4.setEnabled(False)
        self.doubleSpinBox_4.setGeometry(QtCore.QRect(304, 210, 77, 21))
        self.label_kev_4 = QtWidgets.QLabel(MCAData)
        self.label_kev_4.setObjectName(u"label_kev_4")
        self.label_kev_4.setGeometry(QtCore.QRect(390, 210, 55, 16))

        self.label_peak_5 = QtWidgets.QLabel(MCAData)
        self.label_peak_5.setObjectName(u"label_peak_5")
        self.label_peak_5.setGeometry(QtCore.QRect(50, 240, 85, 16))
        self.centroid5 = QtWidgets.QLabel(MCAData)
        self.centroid5.setObjectName(u"centroid5")
        self.centroid5.setGeometry(QtCore.QRect(240, 240, 55, 16))
        self.doubleSpinBox_5 = QtWidgets.QDoubleSpinBox(MCAData)
        self.doubleSpinBox_5.setObjectName(u"doubleSpinBox_5")
        self.doubleSpinBox_5.setEnabled(False)
        self.doubleSpinBox_5.setGeometry(QtCore.QRect(304, 240, 77, 21))
        self.label_kev_5 = QtWidgets.QLabel(MCAData)
        self.label_kev_5.setObjectName(u"label_kev_5")
        self.label_kev_5.setGeometry(QtCore.QRect(390, 240, 55, 16))

        self.label_peak_6 = QtWidgets.QLabel(MCAData)
        self.label_peak_6.setObjectName(u"label_peak_6")
        self.label_peak_6.setGeometry(QtCore.QRect(50, 270, 120, 16))
        self.centroid6 = QtWidgets.QLabel(MCAData)
        self.centroid6.setObjectName(u"centroid6")
        self.centroid6.setGeometry(QtCore.QRect(240, 270, 55, 16))
        self.doubleSpinBox_6 = QtWidgets.QDoubleSpinBox(MCAData)
        self.doubleSpinBox_6.setObjectName(u"doubleSpinBox_6")
        self.doubleSpinBox_6.setEnabled(False)
        self.doubleSpinBox_6.setGeometry(QtCore.QRect(304, 270, 77, 21))
        self.label_kev_6 = QtWidgets.QLabel(MCAData)
        self.label_kev_6.setObjectName(u"label_kev_6")
        self.label_kev_6.setGeometry(QtCore.QRect(390, 270, 55, 16))

        self.label_calibrationFunction = QtWidgets.QLabel(MCAData)
        self.label_calibrationFunction.setObjectName(u"label_CalibrationFunction")
        self.label_calibrationFunction.setGeometry(QtCore.QRect(50, 300, 110, 16))
        self.text_calibrationFunction = QtWidgets.QLabel(MCAData)
        self.text_calibrationFunction.setObjectName(u"text_CalibrationFunction")
        self.text_calibrationFunction.setGeometry(QtCore.QRect(240, 300, 110, 16))
        self.label_r_squared = QtWidgets.QLabel(MCAData)
        self.label_r_squared.setObjectName(u"label_CalibrationFunction")
        self.label_r_squared.setGeometry(QtCore.QRect(50, 330, 110, 16))
        self.text_r_squared = QtWidgets.QLabel(MCAData)
        self.text_r_squared.setObjectName(u"text_CalibrationFunction")
        self.text_r_squared.setGeometry(QtCore.QRect(240, 330, 110, 16))

        self.SubsetSize.raise_()
        self.DataOffset.raise_()
        self.SubsetSizeLabel.raise_()
        self.DataOffsetLabel.raise_()
        self.Load.raise_()
        self.logSwitch.raise_()
        self.setBoundaries.raise_()
        self.label.raise_()
        self.label_2.raise_()
        self.label_3.raise_()
        self.label_5.raise_()

        self.calibration.raise_()
        self.label_6.raise_()
        self.label_7.raise_()
        self.label_10.raise_()
        self.label_11.raise_()
        self.label_peak_3.raise_()
        self.label_peak_4.raise_()
        self.label_peak_5.raise_()
        self.label_peak_6.raise_()
        self.centroid1.raise_()
        self.centroid2.raise_()
        self.centroid3.raise_()
        self.centroid4.raise_()
        self.centroid5.raise_()
        self.centroid6.raise_()
        self.doubleSpinBox.raise_()
        self.doubleSpinBox_2.raise_()
        self.doubleSpinBox_3.raise_()
        self.doubleSpinBox_4.raise_()
        self.doubleSpinBox_5.raise_()
        self.doubleSpinBox_6.raise_()
        self.label_kev_3.raise_()
        self.label_kev_4.raise_()
        self.label_kev_5.raise_()
        self.label_kev_6.raise_()
        self.label_calibrationFunction.raise_()
        self.text_calibrationFunction.raise_()
        self.label_r_squared.raise_()
        self.text_r_squared.raise_()

        self.spinBoxes = [self.doubleSpinBox, self.doubleSpinBox_2, self.doubleSpinBox_3, self.doubleSpinBox_4,
                          self.doubleSpinBox_5, self.doubleSpinBox_6]

        self.retranslateUi(MCAData)
        QtCore.QMetaObject.connectSlotsByName(MCAData)

        self.Load.clicked.connect(self.load_data)
        self.logSwitch.clicked.connect(self.get_results)
        self.setBoundaries.clicked.connect(self.set_boundaries)
        self.calibration.clicked.connect(self.perform_linear_calibration)
        self.SubsetSize.valueChanged.connect(self.size_slider_update)
        self.DataOffset.valueChanged.connect(self.offset_slider_update)

    def retranslateUi(self, MCAData):
        _translate = QtCore.QCoreApplication.translate
        MCAData.setWindowTitle(_translate("MCAData", "SeGaSA"))
        self.SubsetSizeLabel.setText(_translate("MCAData", "Subset size"))
        self.DataOffsetLabel.setText(_translate("MCAData", "Current position"))
        self.Load.setText(_translate("MCAData", "Load data"))
        self.logSwitch.setText(_translate("MCAData", "Results"))
        self.setBoundaries.setText(_translate("MCAData", "Set peak boundaries"))
        self.label.setText(_translate("MCAData", "Number of .mca files loaded:"))
        self.label_2.setText(_translate("MCAData", str(LOADED)))
        self.label_3.setText(_translate("MCAData", str(WINDOW_SIZE)))
        self.label_5.setText(_translate("MCAData", str(OFFSET)))

        self.calibration.setText(_translate("MCAData", u"Calibrate"))
        self.label_4.setText(_translate("MCAData", u"Energy calibration"))
        self.label_6.setText(_translate("MCAData", u"1st peak (Peak of interest):"))
        self.centroid1.setText(_translate("MCAData", str(self.picked[0])))
        self.label_10.setText(_translate("MCAData", u"keV"))
        self.label_7.setText(_translate("MCAData", u"2nd peak:"))
        self.centroid2.setText(_translate("MCAData", str(self.picked[1])))
        self.label_11.setText(_translate("MCAData", u"keV"))
        self.label_peak_3.setText(_translate("MCAData", u"3rd peak:"))
        self.centroid3.setText(_translate("MCAData", str(self.picked[2])))
        self.label_kev_3.setText(_translate("MCAData", u"keV"))
        self.label_peak_4.setText(_translate("MCAData", u"4th peak:"))
        self.centroid4.setText(_translate("MCAData", str(self.picked[3])))
        self.label_kev_4.setText(_translate("MCAData", u"keV"))
        self.label_peak_5.setText(_translate("MCAData", u"5th peak:"))
        self.centroid5.setText(_translate("MCAData", str(self.picked[4])))
        self.label_kev_5.setText(_translate("MCAData", u"keV"))
        self.label_peak_6.setText(_translate("MCAData", u"6th peak:"))
        self.centroid6.setText(_translate("MCAData", str(self.picked[5])))
        self.label_kev_6.setText(_translate("MCAData", u"keV"))
        self.label_calibrationFunction.setText(_translate("MCAData", u"Calibration function:"))
        self.text_calibrationFunction.setText(_translate("MCAData", u"-"))
        self.label_r_squared.setText(_translate("MCAData", u"R-squared:"))
        self.text_r_squared.setText(_translate("MCAData", u"-"))

    def load_data(self):
        dirpath = QtWidgets.QFileDialog.getExistingDirectory(None, 'Select project folder:', 'F:\\',
                                                             QtWidgets.QFileDialog.ShowDirsOnly)
        if dirpath != "":
            self.data.get_data_from_directory(dirpath)
            global LOADED
            LOADED = len(self.data.data)
            global WINDOW_SIZE
            WINDOW_SIZE = 1
            global OFFSET
            OFFSET = 1
            _translate = QtCore.QCoreApplication.translate
            self.label_2.setText(_translate("MCAData", str(LOADED)))
            self.label_3.setText(_translate("MCAData", str(WINDOW_SIZE)))
            self.label_5.setText(_translate("MCAData", str(OFFSET)))
            self.SubsetSize.setValue(1)
            self.DataOffset.setValue(1)
            self.SubsetSize.setMaximum(LOADED)
            self.DataOffset.setMaximum(LOADED - 1)
            self.create_dataset()
        else:
            print("Invalid path")

    def set_logSwitch(self):
        self.logScale = not self.logScale

    def size_slider_update(self):
        global WINDOW_SIZE
        WINDOW_SIZE = self.SubsetSize.value()
        self.DataOffset.setValue(1)
        self.DataOffset.setMaximum(LOADED - WINDOW_SIZE + 1)
        _translate = QtCore.QCoreApplication.translate
        self.label_3.setText(_translate("MCAData", str(WINDOW_SIZE)))
        self.create_dataset()

    def perform_linear_calibration(self):
        if(self.doubleSpinBox.value() and self.doubleSpinBox_2.value()):
            x_list=[]
            for value in self.picked:
                if value != 0:
                    x_list.append(value)
            y_list = []
            for value in self.spinBoxes:
                if value.value() != 0:
                    y_list.append(value.value())

            plt.clf()
            plt.figure(0)
            fig_calibration, ax_calibration = plt.subplots()
            fig_calibration.canvas.set_window_title('Calibration plot')
            plt.xlabel("channels")
            plt.ylabel("energy [keV]")
            popt, _ = curve_fit(linear_model, x_list, y_list)
            a, b = popt
            x_line = np.arange(min(x_list), max(x_list), 1)
            y_line = linear_model(x_line, a, b)
            print(popt)
            plt.scatter(x_list, y_list)
            plt.plot(x_line, y_line, '--', color='red')
            plt.show()

            r_squared = r_squared_linear(x_list, y_list, a, b)

            _translate = QtCore.QCoreApplication.translate
            self.text_calibrationFunction.setText(_translate("MCAData", f"{a:.4f}x {b:+.4f}"))
            self.text_r_squared.setText(_translate("MCAData", f"{r_squared:.4f}"))

            self.calibrationTangent = a
            self.calibrationOffset = b
            # self.calibratedArray = []
            # for i in range(len(self.merged)):
            #     self.calibratedArray.append(linear_model(i, self.calibrationTangent, self.calibrationOffset))
            # self.create_dataset()

    def get_results(self):
        backgrounds = []
        starts = []
        ends = []
        fwhmError = []
        gaussError = []
        self.peakStart = []
        self.peakEnd = []
        correctedHeight = []

        if(self.boundaries):
            for i in range(0, len(self.boundaries)):
                self.peakStart.append(int(self.boundaries[i][0]))
                self.peakEnd.append(int(self.boundaries[i][1]))

        if(len(self.peakStart)==len(self.alteredDataset)):
            for i in range(0, len(self.alteredDataset)):
                ybi1 = []
                ybi2 = []
                x_subset = []
                y_subset = []
                count = 0
                x_difference = self.peakEnd[i]-self.peakStart[i]
                y_difference = self.alteredDataset[i][self.peakEnd[i]] - self.alteredDataset[i][self.peakStart[i]]
                tangent = y_difference/x_difference
                for j in range(self.peakStart[i], self.peakEnd[i]+1):
                    y_subset.append(self.alteredDataset[i][j]-(self.alteredDataset[i][self.peakStart[i]]
                        +tangent*(j-self.peakStart[i])))
                    if j == self.alteredDatasetPeaks[i]:
                         correctedHeight.append(self.alteredDataset[i][j]-(self.alteredDataset[i][self.peakStart[i]]
                        +tangent*(j-self.peakStart[i])))
                    x_subset.append(j)
                    # if j <= self.alteredDatasetPeaks[i]:
                    #     ybi1.append(self.alteredDataset[i][j])
                    # else:
                    #     ybi2.append(self.alteredDataset[i][j])
                # background_level = (self.alteredDatasetPeaks[i]-self.peakStart[i])*tangent + self.alteredDataset[i][self.peakStart[i]]
                # percentage = background_level/self.alteredDataset[i][self.alteredDatasetPeaks[i]]
                # k = percentage + ((1-percentage)/2)
                print("eh", y_subset)
                try:
                    self.resultCorrectedFWHM.append(estimate_FWHM(y_subset, self.alteredDatasetPeaks[i]-self.peakStart[i], 0.5))
                except:
                    self.resultCorrectedFWHM.append(0)
                for j in range(self.peakStart[i], self.peakEnd[i]+1):
                    count = count + self.alteredDataset[i][j]
                    # if j <= self.alteredDatasetPeaks[i]:
                    #     ybi1.append(self.alteredDataset[i][j])
                    # else:
                    #     ybi2.append(self.alteredDataset[i][j])
                self.areaCount.append(count-(int(((self.alteredDataset[i][self.peakStart[i]]+
                                                   self.alteredDataset[i][self.peakEnd[i]])*
                                                  (self.peakEnd[i]-self.peakStart[i])/2))))
                self.resultGrossArea.append(count)
                # xbi1 = list(range(0, len(ybi1)))
                # xbi2 = list(range(0, len(ybi2)))
                # try:
                #     popt1, _ = curve_fit(bigaussian_lateral, xbi1, ybi1)
                #     popt2, _ = curve_fit(bigaussian_lateral, xbi2, ybi2)
                #     base1, height1, center1, width1 = popt1
                #     base2, height2, center2, width2 = popt2
                #     self.resultBigaussFWHM.append(width1+width2)
                # except:
                #     self.resultBigaussFWHM.append(width1+width2)
        print(self.resultCorrectedFWHM)
        for i in range(0, len(self.alteredDataset)):
            x = []
            y = []
            start = find_peak_start(self.alteredDataset[i], self.alteredDatasetPeaks[i], 10)
            starts.append(start[0])
            end = find_peak_end(self.alteredDataset[i], self.alteredDatasetPeaks[i], -10)
            ends.append(end[0])
            for j in range(5):
                y.append(self.alteredDataset[i][self.alteredDatasetPeaks[i]-2+j])
                x.append(j)



            self.resultCentroids.append(estimate_centroid(self.alteredDataset[i], self.alteredDatasetPeaks[i]))
            try:
                self.resultFWHM.append(estimate_FWHM(self.alteredDataset[i], self.alteredDatasetPeaks[i], 0.5))
                backgrounds.append(estimate_background_area(self.alteredDataset[i], self.alteredDatasetPeaks[i],
                                                                self.resultFWHM[i])[2])
            except:
                backgrounds.append(0)
                self.resultFWHM.append(0)

            try:
                popt, _ = curve_fit(gaussian_model, x, y)

                a, mu, sigma = popt

                self.resultGaussCentroid.append(mu+self.alteredDatasetPeaks[i]-2)
            except:
                self.resultGaussCentroid.append(0)
            #     area = estimate_gaussian_area(0, end[0]-start[0], a, mu, sigma)[0]
            #     self.resultGaussCentroid.append(area)
            # except:
            #     self.resultGaussCentroid.append(0)


        # print(backgrounds)
        # print(self.resultCentroids)
        # print(self.resultFWHM)

        result = []
        x_result = []
        for i in range(0, len(self.alteredDataset)):
            x_result.append(i*self.data.time)
        for i in range(0, len(self.resultGaussCentroid)):

            if (self.resultGaussCentroid[i] == 0):

                if (i > 0 and i < len(self.resultGaussCentroid) - 1):
                    self.resultGaussCentroid[i] = np.mean([self.resultGaussCentroid[i - 1], self.resultGaussCentroid[i + 1]])
                gaussError.append(self.resultGaussCentroid[i])
            else:
                gaussError.append(0)

            if (self.resultFWHM[i] == 0):
                if (i > 0 and i < len(self.resultFWHM) - 1):
                    self.resultFWHM[i] = np.mean([self.resultFWHM[i - 1], self.resultFWHM[i + 1]])
                fwhmError.append(self.resultFWHM[i])
                print("ok")

            else:
                fwhmError.append(0)


        fig, ax = plt.subplots(2, 2, figsize=(18, 10))
        fig.canvas.set_window_title('Results')


        self.resultGaussCentroid=np.array(self.resultGaussCentroid)
        stdC = np.std(self.resultGaussCentroid)
        meanC = np.mean(self.resultGaussCentroid)
        print(stdC,meanC)
        self.resultFWHM = np.array(self.resultFWHM)
        stdH = np.std(self.resultFWHM)
        meanH = np.mean(self.resultFWHM)
        # for i in range(0, len(self.resultGaussCentroid)-1):
        #     if (self.resultGaussCentroid[i] > self.resultGaussCentroid[i-1] + 3*stdC
        #             or self.resultGaussCentroid[i] < self.resultGaussCentroid[i-1] - 3*stdC):
        #         self.resultGaussCentroid[i] = self.resultGaussCentroid[i-1]
        # for i in range(0, len(self.resultFWHM)-1):
        #     if (self.resultFWHM[i] > self.resultFWHM[i-1] + 3*stdH
        #             or self.resultFWHM[i] < self.resultFWHM[i-1] - 3*stdH):
        #         self.resultFWHM[i] = self.resultFWHM[i-1]

        print(self.resultGaussCentroid)

        centroid1 = np.array(self.resultCentroids);
        centroid2 = np.array(self.resultGaussCentroid)
        ax[0,0].plot(x_result, self.calibrationTangent*centroid1+self.calibrationOffset, color="g", label="Five-channel method")
        ax[0, 0].plot(x_result, self.calibrationTangent * centroid1 + self.calibrationOffset, "x", color="g",
                      markersize=5)
        ax[0,0].set_title('Centroid')
        if (self.calibrationTangent != 1):
            ax[0, 0].set(ylabel='energy [keV]', xlabel='time [seconds]')
        else:
            ax[0, 0].set(ylabel='channel', xlabel='time [seconds]')

        ax[0, 0].plot(x_result, self.calibrationTangent*centroid2+self.calibrationOffset, color="r", label="Peak apex Gaussian fit")
        ax[0, 0].plot(x_result, self.calibrationTangent * centroid2 + self.calibrationOffset, "x", color="r",
                      markersize=5)
        # ax1_1 = ax[0,0].twiny()
          # Create a dummy plot

        ax[0,0].legend(loc="lower left")
        ax[0,0].plot()
        # ax1_1.plot(range(len(self.alteredDataset)))
        #
        # ax1_1.cla()
        # ax1_1.set(xlabel='measurement')
        ax[0,0].plot()

        fwhm1 = np.array(self.resultFWHM)
        fwhm2 = np.array(self.resultCorrectedFWHM)
        # if (self.calibrationTangent != 1):
        #     fwhm1 = fwhm1*1000
        #     fwhm2 = fwhm2*1000
        ax[0,1].plot(x_result, self.calibrationTangent*fwhm1, color="g", label="Analytical interpolation with background")
        ax[0,1].plot(x_result, self.calibrationTangent*fwhm1, "x", color="g", markersize=5)
        if(self.resultCorrectedFWHM):
            ax[0,1].plot(x_result, self.calibrationTangent*fwhm2, color="r", label="Analytical interpolation with background substraction")
            ax[0,1].plot(x_result, self.calibrationTangent*fwhm2, color="r", marker="x", markersize=5)

        ax[0,1].legend(loc="upper left")
        ax[0,1].set_title('FWHM')
        if(self.calibrationTangent != 1):
            ax[0,1].set(ylabel='FWHM [keV]', xlabel='time [seconds]')
        else:
            ax[0, 1].set(ylabel='FWHM [channel]', xlabel='time [seconds]')

        maxcount = []
        for index in range(len(self.alteredDataset)):
            maxcount.append(self.alteredDataset[index][self.alteredDatasetPeaks[index]])
        maxcount = np.array(maxcount)
        ax[1, 0].plot(x_result, maxcount, color="g", label="Peak height with background")
        ax[1, 0].plot(x_result, maxcount, "x", color="g", markersize=5)
        print(correctedHeight)
        if (self.areaCount):
            ax[1, 0].plot(x_result, correctedHeight, color="r", label="Peak height with background substraction")
            ax[1, 0].plot(x_result, correctedHeight, "x", color="r", markersize=5)
            ax[1, 1].plot(x_result, self.resultGrossArea, color="g", label="Gross area")
            ax[1, 1].plot(x_result, self.resultGrossArea, "x", color="g", markersize=5)
            ax[1,1].plot(x_result, self.areaCount, color="r", label="Net area")
            ax[1,1].plot(x_result, self.areaCount, "x", color="r", markersize=5)

            ax[1,1].set_title('Peak area')
            ax[1,1].set(ylabel='counts', xlabel='time [seconds]')

            ax[1, 1].legend(loc="lower left")
            ax[1, 1].set_title('Peak area')
            ax[1, 1].set(ylabel='counts', xlabel='time [seconds]')
        ax[1, 0].legend(loc="upper right")
        ax[1, 0].set_title('Peak height')
        ax[1, 0].set(ylabel='counts', xlabel='time [seconds]')
        for i in range(0,len(fwhmError)):
            if fwhmError[i]:
                if self.calibrationTangent!=1:
                    ax[0,1].plot(i*self.data.time, self.calibrationTangent*fwhmError[i], "o", color="r")
                else:
                    ax[0, 1].plot(i * self.data.time, self.calibrationTangent * fwhmError[i], "o", color="r")
            if gaussError[i]:
                ax[0,0].plot(i*self.data.time, self.calibrationTangent*gaussError[i]+self.calibrationOffset, "o", color="r", label="gaussian fit failiure")

        # ax3.set_title('Signal count in area (Gaussian smoothing applied, background included)')
        # ax3.set(ylabel='Gross area', xlabel='time [seconds]')
        # axs[1, 1].plot(x_result, result, 'tab:red')
        # axs[1, 1].set_title('Signal count in area (Gaussian smoothing applied, subtracted background)')
        # axs[1, 1].set(ylabel='Net area', xlabel='time [seconds]')


        plt.plot()
        plt.draw()
        plt.show()

    def create_dataset(self):
        print(self.data.time)
        self.data.time = float(self.data.time)
        print(self.data.time)
        self.data.time = int(self.data.time)
        if (self.data.data != [] and self.logScale == False):
            lst = [0] * len(self.data.data[0])
            prom = 0
            self.picked=[0, 0, 0, 0, 0, 0]
            self.analyzedPeak = -1
            self.peak_counter=0
            for i in range(OFFSET - 1, OFFSET + WINDOW_SIZE - 1):
                lst = np.add(lst, self.data.data[i])
                prom = prom + 1
            plt.figure(1)
            plt.clf()
            # plt.plot(lst)
            for i in range(OFFSET - 1, OFFSET + WINDOW_SIZE - 1):
                lst = np.add(lst, self.data.data[i])
                prom = prom + 1
            self.alteredDataset = []
            self.alteredDatasetPeaks = []
            for i in range(0, len(self.data.data)-WINDOW_SIZE+1):

                tmp = [0] * len(self.data.data[0])
                for j in range(i, WINDOW_SIZE + i):

                    tmp = np.add(tmp, self.data.data[j])
                self.alteredDataset.append(tmp.tolist())


            self.ax.set_title('MCA output visualization')
            self.fig.canvas.set_window_title('Selected spectrum')

            peaks=[]
            self.merged=lst
            peaks, _ = find_peaks(lst, prominence=100 * prom)


            peaks = np.array(peaks.tolist())
            peaks = sorted(peaks, key=lambda peak: lst[peak], reverse=True)

            centroids = []
            maxima = []


            for i in range(1, len(peaks)):
                if i < 9:
                    #plt.plot(peaks[i], lst[peaks[i]], "g*")
                    maxima.append(lst[peaks[i]])
                    centroids.append(estimate_centroid(lst, peaks[i]))
            print(centroids)


            self.ax.set_title('MCA output visualization')
            plt.xlabel('channel')
            plt.ylabel('counts')


            if len(self.calibratedArray)==len(lst):
                plt.plot(self.calibratedArray, lst)

                plt.xlabel('Energy [keV]')
                plt.ylabel('Relative intensity')
                if (len(peaks) > 1):
                    plt.axis([-20*self.calibrationTangent,
                              (peaks[1] + 60) * self.calibrationTangent,
                              -200,
                              lst[peaks[1]] + 200 * prom])
            else:
                plt.plot(lst)
                if (len(peaks) > 1):
                    plt.axis([-20, 250, -200, lst[peaks[1]] + 200 * prom])

            def get_channel(event):
                print('you pressed', event.key, event.xdata, event.ydata)
                if(event.xdata and event.ydata):
                    x = int(event.xdata)
                    if x>5 and x<1018:
                        for i in range(-4,4,1):
                            if (lst[x+i-1] < lst[x+i] and lst[x+i+1] < lst[x+i]):
                                if (len(self.picked)>1):
                                    plt.clf()
                                    plt.plot(lst)
                                    self.ax.set_title('MCA output visualisation')
                                    plt.xlabel('channel')
                                    plt.ylabel('counts')
                                    if (len(peaks) > 1):
                                        plt.axis([-20, 250, -200, lst[peaks[1]] + 200 * prom])

                                if (estimate_centroid(lst, x+i) not in self.picked):
                                    if self.peak_counter < 5:
                                        self.picked[self.peak_counter] = estimate_centroid(lst, x+i)
                                        self.peak_counter = self.peak_counter + 1
                                    else:
                                        self.picked.append(estimate_centroid(lst, x+i))
                                        self.picked.pop(0)

                                for index in range(len(self.picked)):
                                    if self.picked[index] != 0:
                                        plt.axvline(self.picked[index], 0, 1)
                                plt.draw()
                                break
                for index in range(len(self.picked)):
                    if self.picked[index]:
                        self.spinBoxes[index].setEnabled(True)
                        if index == 0:
                            self.logSwitch.setEnabled(True)
                            self.setBoundaries.setEnabled(True)
                        if index == 1:
                            self.calibration.setEnabled(True)


                tmp=0
                for i in range(0, len(self.alteredDataset)):
                    if i == 0:
                        tmp = get_max_channel(self.alteredDataset[i], self.picked[0])

                    else:
                        tmp = get_max_channel(self.alteredDataset[i], tmp)

                    self.alteredDatasetPeaks.append(tmp)


                _translate = QtCore.QCoreApplication.translate
                self.centroid1.setText(_translate("MCAData", str(round(self.picked[0],3))))
                self.centroid2.setText(_translate("MCAData", str(round(self.picked[1],3))))
                self.centroid3.setText(_translate("MCAData", str(round(self.picked[2], 3))))
                self.centroid4.setText(_translate("MCAData", str(round(self.picked[3], 3))))
                self.centroid5.setText(_translate("MCAData", str(round(self.picked[4], 3))))
                self.centroid6.setText(_translate("MCAData", str(round(self.picked[5], 3))))
                self.analyzedPeak = self.picked[0]

            cid1 = self.fig.canvas.mpl_connect('button_press_event', get_channel)

            plt.draw()

            # plt.yscale('linear')

            # plt.xlabel('channel')
            # plt.ylabel('counts')

            # if (len(peaks)>1 and len(centroids)>1):
                # plt.axis([-20, 300, -200, lst[peaks[1]]+200*prom])
                # plt.vlines(centroids, -200, lst[peaks[1]]+200*prom, colors=['b','r','g','c','m','y','k','orange'])
            plt.show()



    def offset_slider_update(self):
        global OFFSET
        OFFSET = self.DataOffset.value()
        _translate = QtCore.QCoreApplication.translate
        self.label_5.setText(_translate("MCAData", str(OFFSET)))
        self.create_dataset()

    def set_boundaries(self):
        self.boundaries=[]
        for i in range(len(self.alteredDataset)):
            def line_select_callback(eclick, erelease):
                'eclick and erelease are the press and release events'
                x1, y1 = eclick.xdata, eclick.ydata
                x2, y2 = erelease.xdata, erelease.ydata
                self.tempBoundaries = [x1, x2]
                print("(%3.2f, %3.2f) --> (%3.2f, %3.2f)" % (x1, y1, x2, y2))
                print(self.tempBoundaries)
                print(" The button you used were: %s %s" % (eclick.button, erelease.button))

            def toggle_selector(event):
                print(' Key pressed.')
                if event.key and toggle_selector.RS.active:
                    print(' RectangleSelector deactivated.')
                    self.boundaries.append(self.tempBoundaries)
                    print(self.boundaries)
                    toggle_selector.RS.set_active(False)

            fig, current_ax = plt.subplots()
            fig.canvas.set_window_title('Set peak boundaries')
            plt.plot(self.alteredDataset[i])
            plt.axis([-20, 250, -200, self.alteredDataset[i][self.alteredDatasetPeaks[i]] + 200])
            plt.draw()  # draw the plot
            toggle_selector.RS = RectangleSelector(current_ax, line_select_callback,
                                                   drawtype='box', useblit=True,
                                                   button=[1, 3],  # don't use middle button
                                                   minspanx=5, minspany=5,
                                                   spancoords='pixels',
                                                   interactive=True)
            plt.connect('key_press_event', toggle_selector)
            while(plt.waitforbuttonpress()==False):
                continue


class ParsedDataFromMca:

    def __init__(self):
        self.data = []
        self.time = 0

    def parse_from_mca_file(self, filepath):
        initial_string = "<<DATA>>"
        terminal_string = "<<END>>"
        time_marker = "LIVE_TIME"

        parse_flag = False
        parsed_data = []

        if os.path.isfile(filepath) and ".mca" in filepath and "v.mca" not in filepath:
            with open(filepath) as mca_file:
                for line in mca_file:
                    if time_marker in line:
                        print(line)
                        if len(line.split()) > 2:
                            self.time = line.split()[2]
                    if initial_string in line:
                        parse_flag = True
                        continue
                    elif terminal_string in line:
                        break
                    if parse_flag:
                        parsed_data.append(int(line))
        else:
            print("Provided filepath is invalid")
        if parsed_data:
            self.data.append(parsed_data)

    def get_data_from_directory(self, directory_path):
        files = []
        for root, directories, filenames in os.walk(directory_path):
            for name in filenames:
                if ".mca" in name and "v.mca" not in name:
                    files.append(directory_path + '\\' + name)
        for file in files:
            self.parse_from_mca_file(file)


app = QtWidgets.QApplication(sys.argv)
MainWindow = QtWidgets.QMainWindow()
ui = Ui_MCAData()
ui.setupUi(MainWindow)
MainWindow.show()

sys.exit(app.exec_())

