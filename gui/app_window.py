from PyQt5 import QtCore, QtWidgets


class AppWindow(object):
    """A class containing GUI widgets of the main window.

    This class represents the View component of the SeGaSA system.
    Superclass of the EventController class.
    Contains QtWidgets and event connections.
    Majority of the class is converted from the Qt Designer .ui output file.
    Widget variables naming format: type_name.

    LIST OF WIDGETS:

    BUTTONS: -- used to trigger controller functions
        button_load_data
            -- enabled by default
            -- triggers load_data() .mca data import event
        button_set_boundaries
            -- enabled by picking a peak in a spectrum
            -- triggers set_peak_boundaries(), manual peak selection event
        button_show_results
            -- enabled by picking a peak in a spectrum
            -- triggers evaluate_sequence(), spectrometric evaluation event
        button_export_results
            -- enabled by successful sequence evaluation
            -- triggers export_results(), export event
        button_calibrate
            -- enabled by selecting more than one peak in a spectrum
            -- triggers linear_calibration(), energy calibration event

    SLIDERS:
        slider_subset_size
            -- determines the number of .mca files summed in one spectrum of the analyzed sequence
        slider_subset_position
            -- determines which spectrum of the analyzed sequence is displayed in the initial plot

    SPINBOXES:
        spinbox_peak_1
            -- input field for calibration specification (same goes for the other spinboxes)
        spinbox_peak_2
        spinbox_peak_3
        spinbox_peak_4
        spinbox_peak_5
        spinbox_peak_6
        spinbox : list of spinboxes -- contains all of the listed spinboxes

    LABELS:
        label_subset_size
        label_subset_size_text
        label_subset_position
        label_subset_position_text
        label_set_size_text
        label_set_size
        label_calibration_header
        label_peak_1
        label_centroid_1
        label_kev_1
        label_peak_2
        label_centroid_2
        label_kev_2
        label_peak_3
        label_centroid_3
        label_kev_3
        label_peak_4
        label_centroid_4
        label_kev_4
        label_peak_5
        label_centroid_5
        label_kev_5
        label_peak_6
        label_centroid_6
        label_kev_6
        label_calibration_function
        label_calibration_function_value
        label_r_squared
        label_r_squared_value

    FUNCTIONS:
        setupUi(main_window) : void
            -- performs the initialization of UI widgets
        retranslateUi(main_window) : void
            -- sets text values of buttons and labels
    """
    def setupUi(self, main_window):
        main_window.setObjectName("main_window")
        main_window.setGeometry(10, 50, 711, 520)
        main_window.setFixedSize(main_window.size())

        self.slider_subset_size = QtWidgets.QSlider(main_window)
        self.slider_subset_size.setGeometry(QtCore.QRect(440, 430, 211, 22))
        self.slider_subset_size.setOrientation(QtCore.Qt.Horizontal)
        self.slider_subset_size.setObjectName("SubsetSize")
        self.slider_subset_size.setMinimum(1)
        self.slider_subset_size.setSingleStep(1)
        self.slider_subset_size.setMaximum(1)
        self.label_subset_size = QtWidgets.QLabel(main_window)
        self.label_subset_size.setGeometry(QtCore.QRect(400, 430, 21, 16))
        self.label_subset_size.setObjectName("label_3")
        self.label_subset_size_text = QtWidgets.QLabel(main_window)
        self.label_subset_size_text.setGeometry(QtCore.QRect(50, 430, 321, 20))
        self.label_subset_size_text.setObjectName("SubsetSizeLabel")

        self.slider_subset_position = QtWidgets.QSlider(main_window)
        self.slider_subset_position.setGeometry(QtCore.QRect(440, 470, 211, 22))
        self.slider_subset_position.setOrientation(QtCore.Qt.Horizontal)
        self.slider_subset_position.setObjectName("DataOffset")
        self.slider_subset_position.setMinimum(1)
        self.slider_subset_position.setSingleStep(1)
        self.slider_subset_position.setMaximum(1)
        self.label_subset_position = QtWidgets.QLabel(main_window)
        self.label_subset_position.setGeometry(QtCore.QRect(400, 470, 21, 21))
        self.label_subset_position.setObjectName("label_5")
        self.label_subset_position_text = QtWidgets.QLabel(main_window)
        self.label_subset_position_text.setGeometry(QtCore.QRect(50, 470, 291, 20))
        self.label_subset_position_text.setObjectName("DataOffsetLabel")

        self.button_load_data = QtWidgets.QPushButton(main_window)
        self.button_load_data.setGeometry(QtCore.QRect(40, 30, 141, 31))
        self.button_load_data.setObjectName("Load")

        self.button_set_boundaries = QtWidgets.QPushButton(main_window)
        self.button_set_boundaries.setGeometry(QtCore.QRect(450, 115, 221, 31))
        self.button_set_boundaries.setObjectName("setBoundaries")
        self.button_set_boundaries.setEnabled(False)

        self.button_show_results = QtWidgets.QPushButton(main_window)
        self.button_show_results.setGeometry(QtCore.QRect(450, 160, 221, 31))
        self.button_show_results.setObjectName("show_results")
        self.button_show_results.setEnabled(False)

        self.button_export_results = QtWidgets.QPushButton(main_window)
        self.button_export_results.setGeometry(QtCore.QRect(450, 205, 221, 31))
        self.button_export_results.setObjectName("export_results")
        self.button_export_results.setEnabled(False)

        self.button_calibrate = QtWidgets.QPushButton(main_window)
        self.button_calibrate.setObjectName(u"calibrate")
        self.button_calibrate.setGeometry(QtCore.QRect(50, 360, 330, 31))
        self.button_calibrate.setEnabled(False)

        self.label_set_size_text = QtWidgets.QLabel(main_window)
        self.label_set_size_text.setGeometry(QtCore.QRect(450, 88, 211, 16))
        self.label_set_size_text.setObjectName("set_size_text")
        self.label_set_size = QtWidgets.QLabel(main_window)
        self.label_set_size.setGeometry(QtCore.QRect(640, 88, 21, 16))
        self.label_set_size.setObjectName("set_size")

        self.label_calibration_header = QtWidgets.QLabel(main_window)
        self.label_calibration_header.setObjectName(u"label_4")
        self.label_calibration_header.setGeometry(QtCore.QRect(41, 88, 124, 16))

        self.label_peak_1 = QtWidgets.QLabel(main_window)
        self.label_peak_1.setObjectName(u"calibration_header")
        self.label_peak_1.setGeometry(QtCore.QRect(50, 120, 160, 16))
        self.label_centroid_1 = QtWidgets.QLabel(main_window)
        self.label_centroid_1.setObjectName(u"label_centroid_1")
        self.label_centroid_1.setGeometry(QtCore.QRect(240, 120, 55, 16))
        self.spinbox_peak_1 = QtWidgets.QDoubleSpinBox(main_window)
        self.spinbox_peak_1.setObjectName(u"spinbox_peak_1")
        self.spinbox_peak_1.setEnabled(False)
        self.spinbox_peak_1.setGeometry(QtCore.QRect(304, 120, 77, 21))
        self.label_kev_1 = QtWidgets.QLabel(main_window)
        self.label_kev_1.setObjectName(u"label_10")
        self.label_kev_1.setGeometry(QtCore.QRect(390, 120, 55, 16))

        self.label_peak_2 = QtWidgets.QLabel(main_window)
        self.label_peak_2.setObjectName(u"label_7")
        self.label_peak_2.setGeometry(QtCore.QRect(50, 150, 85, 16))
        self.label_centroid_2 = QtWidgets.QLabel(main_window)
        self.label_centroid_2.setObjectName(u"label_centroid_2")
        self.label_centroid_2.setGeometry(QtCore.QRect(240, 150, 55, 16))
        self.spinbox_peak_2 = QtWidgets.QDoubleSpinBox(main_window)
        self.spinbox_peak_2.setObjectName(u"spinbox_peak_2")
        self.spinbox_peak_2.setEnabled(False)
        self.spinbox_peak_2.setGeometry(QtCore.QRect(304, 150, 77, 21))
        self.label_kev_2 = QtWidgets.QLabel(main_window)
        self.label_kev_2.setObjectName(u"label_11")
        self.label_kev_2.setGeometry(QtCore.QRect(390, 150, 55, 16))

        self.label_peak_3 = QtWidgets.QLabel(main_window)
        self.label_peak_3.setObjectName(u"label_peak_3")
        self.label_peak_3.setGeometry(QtCore.QRect(50, 180, 85, 16))
        self.label_centroid_3 = QtWidgets.QLabel(main_window)
        self.label_centroid_3.setObjectName(u"centroid3")
        self.label_centroid_3.setGeometry(QtCore.QRect(240, 180, 55, 16))
        self.spinbox_peak_3 = QtWidgets.QDoubleSpinBox(main_window)
        self.spinbox_peak_3.setObjectName(u"doubleSpinBox_3")
        self.spinbox_peak_3.setEnabled(False)
        self.spinbox_peak_3.setGeometry(QtCore.QRect(304, 180, 77, 21))
        self.label_kev_3 = QtWidgets.QLabel(main_window)
        self.label_kev_3.setObjectName(u"label_kev_3")
        self.label_kev_3.setGeometry(QtCore.QRect(390, 180, 55, 16))

        self.label_peak_4 = QtWidgets.QLabel(main_window)
        self.label_peak_4.setObjectName(u"label_peak_4")
        self.label_peak_4.setGeometry(QtCore.QRect(50, 210, 85, 16))
        self.label_centroid_4 = QtWidgets.QLabel(main_window)
        self.label_centroid_4.setObjectName(u"centroid4")
        self.label_centroid_4.setGeometry(QtCore.QRect(240, 210, 55, 16))
        self.spinbox_peak_4 = QtWidgets.QDoubleSpinBox(main_window)
        self.spinbox_peak_4.setObjectName(u"doubleSpinBox_4")
        self.spinbox_peak_4.setEnabled(False)
        self.spinbox_peak_4.setGeometry(QtCore.QRect(304, 210, 77, 21))
        self.label_kev_4 = QtWidgets.QLabel(main_window)
        self.label_kev_4.setObjectName(u"label_kev_4")
        self.label_kev_4.setGeometry(QtCore.QRect(390, 210, 55, 16))

        self.label_peak_5 = QtWidgets.QLabel(main_window)
        self.label_peak_5.setObjectName(u"label_peak_5")
        self.label_peak_5.setGeometry(QtCore.QRect(50, 240, 85, 16))
        self.label_centroid_5 = QtWidgets.QLabel(main_window)
        self.label_centroid_5.setObjectName(u"centroid5")
        self.label_centroid_5.setGeometry(QtCore.QRect(240, 240, 55, 16))
        self.spinbox_peak_5 = QtWidgets.QDoubleSpinBox(main_window)
        self.spinbox_peak_5.setObjectName(u"doubleSpinBox_5")
        self.spinbox_peak_5.setEnabled(False)
        self.spinbox_peak_5.setGeometry(QtCore.QRect(304, 240, 77, 21))
        self.label_kev_5 = QtWidgets.QLabel(main_window)
        self.label_kev_5.setObjectName(u"label_kev_5")
        self.label_kev_5.setGeometry(QtCore.QRect(390, 240, 55, 16))

        self.label_peak_6 = QtWidgets.QLabel(main_window)
        self.label_peak_6.setObjectName(u"label_peak_6")
        self.label_peak_6.setGeometry(QtCore.QRect(50, 270, 120, 16))
        self.label_centroid_6 = QtWidgets.QLabel(main_window)
        self.label_centroid_6.setObjectName(u"centroid6")
        self.label_centroid_6.setGeometry(QtCore.QRect(240, 270, 55, 16))
        self.spinbox_peak_6 = QtWidgets.QDoubleSpinBox(main_window)
        self.spinbox_peak_6.setObjectName(u"doubleSpinBox_6")
        self.spinbox_peak_6.setEnabled(False)
        self.spinbox_peak_6.setGeometry(QtCore.QRect(304, 270, 77, 21))
        self.label_kev_6 = QtWidgets.QLabel(main_window)
        self.label_kev_6.setObjectName(u"label_kev_6")
        self.label_kev_6.setGeometry(QtCore.QRect(390, 270, 55, 16))

        self.label_calibration_function = QtWidgets.QLabel(main_window)
        self.label_calibration_function.setObjectName(u"label_CalibrationFunction")
        self.label_calibration_function.setGeometry(QtCore.QRect(50, 300, 110, 16))
        self.label_calibration_function_value = QtWidgets.QLabel(main_window)
        self.label_calibration_function_value.setObjectName(u"text_CalibrationFunction")
        self.label_calibration_function_value.setGeometry(QtCore.QRect(240, 300, 110, 16))
        self.label_r_squared = QtWidgets.QLabel(main_window)
        self.label_r_squared.setObjectName(u"label_CalibrationFunction")
        self.label_r_squared.setGeometry(QtCore.QRect(50, 330, 110, 16))
        self.label_r_squared_value = QtWidgets.QLabel(main_window)
        self.label_r_squared_value.setObjectName(u"text_CalibrationFunction")
        self.label_r_squared_value.setGeometry(QtCore.QRect(240, 330, 110, 16))

        # raise_() makes widgets top level
        self.slider_subset_size.raise_()
        self.slider_subset_position.raise_()
        self.label_subset_size_text.raise_()
        self.label_subset_position_text.raise_()
        self.button_load_data.raise_()
        self.button_show_results.raise_()
        self.button_set_boundaries.raise_()
        self.label_set_size_text.raise_()
        self.label_set_size.raise_()
        self.label_subset_size.raise_()
        self.label_subset_position.raise_()
        self.button_calibrate.raise_()
        self.label_peak_1.raise_()
        self.label_peak_2.raise_()
        self.label_kev_1.raise_()
        self.label_kev_2.raise_()
        self.label_peak_3.raise_()
        self.label_peak_4.raise_()
        self.label_peak_5.raise_()
        self.label_peak_6.raise_()
        self.label_centroid_1.raise_()
        self.label_centroid_2.raise_()
        self.label_centroid_3.raise_()
        self.label_centroid_4.raise_()
        self.label_centroid_5.raise_()
        self.label_centroid_6.raise_()
        self.spinbox_peak_1.raise_()
        self.spinbox_peak_2.raise_()
        self.spinbox_peak_3.raise_()
        self.spinbox_peak_4.raise_()
        self.spinbox_peak_5.raise_()
        self.spinbox_peak_6.raise_()
        self.label_kev_3.raise_()
        self.label_kev_4.raise_()
        self.label_kev_5.raise_()
        self.label_kev_6.raise_()
        self.label_calibration_function.raise_()
        self.label_calibration_function_value.raise_()
        self.label_r_squared.raise_()
        self.label_r_squared_value.raise_()

        self.spinboxes = [self.spinbox_peak_1, self.spinbox_peak_2, self.spinbox_peak_3, self.spinbox_peak_4,
                          self.spinbox_peak_5, self.spinbox_peak_6]

        self.retranslateUi(main_window)
        QtCore.QMetaObject.connectSlotsByName(main_window)

        self.button_load_data.clicked.connect(self.load_data)
        self.button_show_results.clicked.connect(self.get_results)
        self.button_set_boundaries.clicked.connect(self.set_peak_boundaries)
        self.button_calibrate.clicked.connect(self.linear_calibration)
        self.slider_subset_size.valueChanged.connect(self.size_slider_update)
        self.slider_subset_position.valueChanged.connect(self.position_slider_update)
        self.button_export_results.clicked.connect(self.export_results)

    def retranslateUi(self, main_window):
        _translate = QtCore.QCoreApplication.translate
        main_window.setWindowTitle(_translate("segasa", "segasa"))
        self.label_subset_size_text.setText(_translate("segasa", "Number of .mca files summed in one spectrum"))
        self.label_subset_position_text.setText(_translate("segasa", "Current position in sequence"))
        self.button_load_data.setText(_translate("segasa", "Load data"))
        self.button_show_results.setText(_translate("segasa", "Show results"))
        self.button_set_boundaries.setText(_translate("segasa", "Set peak boundaries"))
        self.button_export_results.setText(_translate("segasa", "Save results in .txt file"))
        self.label_set_size_text.setText(_translate("segasa", "Number of .mca files loaded:"))
        self.label_set_size.setText(_translate("segasa", "0"))
        self.label_subset_size.setText(_translate("segasa", "0"))
        self.label_subset_position.setText(_translate("segasa", "0"))

        self.button_calibrate.setText(_translate("segasa", u"Calibrate"))
        self.label_calibration_header.setText(_translate("segasa", u"Energy calibration"))
        self.label_peak_1.setText(_translate("segasa", u"1st peak (Peak of interest):"))
        self.label_centroid_1.setText(_translate("segasa", "-"))
        self.label_kev_1.setText(_translate("segasa", u"keV"))
        self.label_peak_2.setText(_translate("segasa", u"2nd peak:"))
        self.label_centroid_2.setText(_translate("segasa", "-"))
        self.label_kev_2.setText(_translate("segasa", u"keV"))
        self.label_peak_3.setText(_translate("segasa", u"3rd peak:"))
        self.label_centroid_3.setText(_translate("segasa", "-"))
        self.label_kev_3.setText(_translate("segasa", u"keV"))
        self.label_peak_4.setText(_translate("segasa", u"4th peak:"))
        self.label_centroid_4.setText(_translate("segasa", "-"))
        self.label_kev_4.setText(_translate("segasa", u"keV"))
        self.label_peak_5.setText(_translate("segasa", u"5th peak:"))
        self.label_centroid_5.setText(_translate("segasa", "-"))
        self.label_kev_5.setText(_translate("segasa", u"keV"))
        self.label_peak_6.setText(_translate("segasa", u"6th peak:"))
        self.label_centroid_6.setText(_translate("segasa", "-"))
        self.label_kev_6.setText(_translate("segasa", u"keV"))
        self.label_calibration_function.setText(_translate("segasa", u"Calibration function:"))
        self.label_calibration_function_value.setText(_translate("segasa", u"-"))
        self.label_r_squared.setText(_translate("segasa", u"R-squared:"))
        self.label_r_squared_value.setText(_translate("segasa", u"-"))

# app = QtWidgets.QApplication(sys.argv)
# window = QtWidgets.QMainWindow()
# ui = MainWindow()
# ui.setupUi(window)
# window.show()
#
# sys.exit(app.exec_())
