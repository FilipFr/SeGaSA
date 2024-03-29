import sys
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import math

from .utility import *
from data.session_data import SessionData, FWHM, PeakCentroid, PeakArea, PeakHeight

from gui.app_window import AppWindow
from PyQt5 import QtCore, QtWidgets

sys.path.append("..")  # Adds higher directory to python modules path.

"""This module represents the Controller component of the SeGaSA application.

CLASSES:
    EventController : subclass of AppWindow
        -- manages event handling
"""


class EventController(AppWindow):
    """A controller class for event handling.

    A subclass of AppWindow.
    Updates the View and the Model component of the system.

    ATTRIBUTES:
        data : SessionData
            -- represents the Model component
            -- contains data of the currently running session of the application
        fig : Matplotlib Figure
            -- initial plot figure
        ax : Matplotlib Axis
            -- initial plot axis

    METHODS:
        load_data() : void
            -- loads a folder containing .mca files and calls create_sequence() and draw_plot()
        create_sequence() : void
            -- creates/updates sequence and peak data based on loaded data and slider_subset_size value
        linear_calibration() : void
            -- performs linear calibration using curve_fit and displays calibration results
        draw_plot() : void
            -- displays the selected spectrum of the sequence using the initial plot
        get_channel() : void
            -- handles peak selection on the initial plot
        offset_slider_update() : void
            -- responds to slider_subset_position change by displaying a proper plot
        size_slider_update() : void
            -- responds to slider_subset_size change by modifying the current sequence and displaying a proper plot
        set_peak_boundaries() : void
            -- enables the selection of peak boundaries using Matplotlib RectangleSelector on plot
        get_results() : void
            -- displays time dependency plot of evaluated spectrometric parameters and prepares the results for export
        export_results() : void
            -- saves the results as a .txt file
        retranslate() : void
            -- updates the main window
    """

    def __init__(self):
        super(EventController, self).__init__()

        self.data = SessionData()
        self.backup = ""

        self.fig, self.ax = plt.subplots()

    def load_data(self):
        """Loads a folder containing .mca files and calls create_sequence() and draw_plot().

        TRIGGERED BY:
            -- button_load_data click
        """

        dirpath = QtWidgets.QFileDialog.getExistingDirectory(None, 'Select project folder:', "F:\\",
                                                             QtWidgets.QFileDialog.ShowDirsOnly)
        if dirpath != "":
            plt.close('all')
            if self.data.loaded_data.get_data_from_directory(dirpath) == -1:
                msg = QtWidgets.QMessageBox()
                msg.setIcon(QtWidgets.QMessageBox.Warning)
                msg.setText("No data loaded")
                msg.setInformativeText('Directory does not contain any valid .mca files')
                msg.setWindowTitle("Warning")
                msg.exec_()
                if dirpath != self.backup:
                    self.data.loaded_data.get_data_from_directory(self.backup)
            else:
                if len(self.data.loaded_data.spectra):
                    self.backup = dirpath
                    plt.close(self.fig)
                    self.fig.clear()
                    self.fig, self.ax = plt.subplots()
                    self.data.reset()
                    self.data.sequence.subset_size = 1
                    self.data.sequence.current_position = 1
                    self.slider_subset_size.setValue(1)
                    self.slider_subset_position.setValue(1)
                    self.slider_subset_size.setMaximum(self.data.loaded_data.file_count)
                    self.slider_subset_position.setMaximum(self.data.loaded_data.file_count - 1)
                    self.data.sequence.length = self.data.loaded_data.file_count
                    self.button_export_results.setEnabled(False)
                    self.slider_subset_position.setEnabled(True)
                    self.slider_subset_size.setEnabled(True)
                    self.retranslate()
                    self.create_sequence()
                    self.draw_plot()

                    msg = QtWidgets.QMessageBox()
                    msg.setIcon(QtWidgets.QMessageBox.Information)
                    msg.setText('Select up to six peaks by '
                                'clicking near their maximum.\n\n'
                                'Spectrometric analysis will be performed on the first selected peak\n\n'
                                'Additional peaks can be selected for calibration.\n\n'
                                'Use any slider to reset the peak selection. \n'
                                '(alternatively, if only a single file was loaded, reload)')
                    msg.setWindowTitle("Data successfully loaded")
                    msg.exec_()

                    self.ax.set_title('MCA output visualization')
                    plt.xlabel('channel')
                    plt.ylabel('counts')

    def create_sequence(self):
        """Creates/updates sequence and peak data based on loaded data and slider_subset_size value.

        PRECONDITIONS:
            -- .mca files were successfully loaded

        TRIGGERED BY:
            -- load_data was successful
            -- slider_subset_size change
        """
        self.data.sequence.spectra = []
        for value in self.spinboxes:
            value.setValue(0)
            value.setMaximum(9999)
        self.data.sequence.spectrum = [0] * len(self.data.loaded_data.spectra[0])
        self.data.peaks.prominence = self.data.sequence.subset_size
        for i in range(self.data.sequence.current_position - 1,
                       self.data.sequence.current_position + self.data.sequence.subset_size - 1):
            self.data.sequence.spectrum = np.add(self.data.sequence.spectrum, self.data.loaded_data.spectra[i])

        for i in range(0, len(self.data.loaded_data.spectra) - self.data.sequence.subset_size + 1):

            tmp = [0] * len(self.data.loaded_data.spectra[0])
            for j in range(i, self.data.sequence.subset_size + i):
                tmp = np.add(tmp, self.data.loaded_data.spectra[j])
            self.data.sequence.spectra.append(tmp.tolist())

        self.data.peaks.found, _ = find_peaks(self.data.sequence.spectrum,
                                              prominence=50 * self.data.peaks.prominence)
        self.data.peaks.found = np.array(self.data.peaks.found.tolist())
        self.data.peaks.found = sorted(self.data.peaks.found,
                                       key=lambda peak: self.data.sequence.spectrum[peak], reverse=True)
        print("Found peaks: ")
        print(self.data.peaks.found)

    def linear_calibration(self):
        """Performs linear calibration using curve_fit and displays calibration results.

        Saves calibration parameters in the SessionData instance.
        Displays calibration equation, r-squared and the calibration plot.

        PRECONDITIONS:
            -- mca. files were successfully loaded
            -- slider_subset_position is set to 1
            -- at least 2 peaks were selected in the displayed spectrum

        TRIGGERED BY:
            -- button_calibrate click
        """

        if self.spinbox_peak_1.value() and self.spinbox_peak_2.value():
            x_list = []
            for value in self.data.peaks.selected:
                if value != 0:
                    x_list.append(value)
            y_list = []
            for value in self.spinboxes:
                if value.value() != 0:
                    y_list.append(value.value())

            try:
                popt, _ = curve_fit(linear_model, x_list, y_list)
            except Exception as e:
                msg = QtWidgets.QMessageBox()
                msg.setIcon(QtWidgets.QMessageBox.Warning)
                msg.setText(str(e))
                msg.setInformativeText('Calibration could not be performed. Reload your data directory and try again.')
                msg.setWindowTitle("Warning")
                msg.exec_()
                return
            plt.figure(2)
            mngr = plt.get_current_fig_manager()
            mngr.set_window_title("Calibration plot")
            plt.xlabel("channels")
            plt.ylabel("energy [keV]")

            a, b = popt
            x_line = np.arange(min(x_list), max(x_list) + 1, 1)
            y_line = linear_model(x_line, a, b)
            print(popt)
            plt.scatter(x_list, y_list)
            plt.plot(x_line, y_line, '--', color='red')
            plt.show()

            self.data.calibration.r_squared = r_squared_linear(x_list, y_list, a, b)

            _translate = QtCore.QCoreApplication.translate
            self.label_calibration_function_value.setText(_translate("segasa", f"{a:.4f}x {b:+.4f}"))
            self.label_r_squared_value.setText(_translate("segasa", f"{self.data.calibration.r_squared:.4f}"))

            self.data.calibration.tangent = a
            self.data.calibration.offset = b

    def draw_plot(self):
        """Displays the selected spectrum of the sequence in a Matplotlib plot.

        PRECONDITIONS:
            -- mca. files were successfully loaded into a sequence

        TRIGGERED BY:
            -- load_data was successful
            -- slider_subset_size change
            -- slider_subset_position change
        """
        plt.figure(1)
        manager = plt.get_current_fig_manager()
        manager.set_window_title("MCA spectrum")
        plt.clf()
        self.data.peaks.selected = [0, 0, 0, 0, 0, 0]
        self.data.peaks.counter = 0
        for spinbox in self.spinboxes:
            spinbox.setEnabled(False)
        plt.plot(self.data.sequence.spectrum)
        if len(self.data.peaks.found) > 1:
            plt.axis([-20*self.data.calibration.tangent, max(self.data.peaks.found)+20, -20 * self.data.peaks.prominence, self.data.sequence.spectrum[
                self.data.peaks.found[1]] + 200 * self.data.peaks.prominence])

        self.fig.canvas.mpl_connect('button_press_event', self.get_channel)
        mngr = plt.get_current_fig_manager()
        mngr.window.setGeometry(720, 50, 711, 520)

        plt.draw()
        plt.ion()
        plt.show()

    def get_channel(self, event):
        """Handles peak selection in the spectrum plot.

        Also updates the main window by displaying the centroid of the selected peak and enabling other events.

        PRECONDITIONS:
            -- mca. files were successfully loaded into a sequence
            -- the spectrum plot is displayed
            -- slider_subset_position is set to 1

        TRIGGERED BY:
            -- click in plot near a found peak
        """
        if self.data.sequence.current_position != 1:
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Warning)
            msg.setText("Cannot pick peaks in current spectrum\n")
            msg.setInformativeText('"Current position in sequence" slider must be set to 1')
            msg.setWindowTitle("Warning")
            msg.exec_()
        else:
            print('you pressed', event.key, event.xdata, event.ydata)
            if event.xdata and event.ydata:
                x = int(event.xdata)
                if 5 < x < 1018:
                    for i in self.data.peaks.found:
                        if math.isclose(x, i, abs_tol=3):
                            if len(self.data.peaks.selected) > 1:
                                plt.figure(1)
                                plt.clf()
                                plt.plot(self.data.sequence.spectrum)
                                self.ax.set_title('MCA output visualisation')
                                plt.xlabel('channel')
                                plt.ylabel('counts')
                                if len(self.data.peaks.found) > 1:
                                    plt.axis([-20, 250, -20 * self.data.peaks.prominence, self.data.sequence.spectrum[
                                        self.data.peaks.found[1]] + 200 * self.data.peaks.prominence])

                            if (estimate_centroid(self.data.sequence.spectrum,
                                                  i) not in self.data.peaks.selected):
                                if self.data.peaks.counter < 6:
                                    self.data.peaks.selected[self.data.peaks.counter] = estimate_centroid(
                                        self.data.sequence.spectrum,
                                        i)
                                    self.data.peaks.counter = self.data.peaks.counter + 1

                            for index in range(len(self.data.peaks.selected)):
                                if self.data.peaks.selected[index] != 0:
                                    plt.axvline(self.data.peaks.selected[index], 0, 1)
                            plt.draw()
                            break
            for index in range(len(self.data.peaks.selected)):
                if self.data.peaks.selected[index]:
                    self.spinboxes[index].setEnabled(True)
                    if index == 0:
                        self.button_show_results.setEnabled(True)
                        self.button_set_boundaries.setEnabled(True)
                    if index == 1:
                        self.button_calibrate.setEnabled(True)
            print(self.data.peaks.found)
            tmp = 0
            for i in range(0, len(self.data.sequence.spectra)):
                if i == 0:
                    tmp = get_max_channel(self.data.sequence.spectra[i], self.data.peaks.selected[0])

                else:
                    tmp = get_max_channel(self.data.sequence.spectra[i], tmp)
                if tmp > 0:
                    self.data.sequence.peaks.append(tmp)

            self.retranslate()

    def position_slider_update(self):
        """Displays the corresponding spectrum of the sequence.

        Resets the sequence by calling create_sequence().

        PRECONDITIONS:
            -- .mca files were successfully loaded into a sequence

        TRIGGERED BY:
            -- slider_subset_position change
        """
        self.data.sequence.current_position = self.slider_subset_position.value()
        self.create_sequence()
        self.draw_plot()
        self.button_set_boundaries.setEnabled(False)
        self.button_show_results.setEnabled(False)
        self.button_calibrate.setEnabled(False)
        self.button_export_results.setEnabled(False)
        self.retranslate()

    def size_slider_update(self):
        """Specifies number of summed .mca spectra and modifies the sequence by calling create_sequence.

        Also displays the the newly created spectrum in position 1 of the sequence.
        Resets the sequence and selected peaks by calling create_sequence().

        PRECONDITIONS:
            -- .mca files were successfully loaded into a sequence

        TRIGGERED BY:
            -- slider_subset_size change
        """
        self.data.reset()
        self.data.sequence.subset_size = self.slider_subset_size.value()
        self.data.sequence.current_position = 1
        self.slider_subset_position.setValue(1)
        self.create_sequence()
        self.slider_subset_position.setMaximum(self.data.loaded_data.file_count - self.data.sequence.subset_size + 1)
        self.draw_plot()
        _translate = QtCore.QCoreApplication.translate
        self.button_set_boundaries.setEnabled(False)
        self.button_show_results.setEnabled(False)
        self.button_calibrate.setEnabled(False)
        self.button_export_results.setEnabled(False)
        self.label_subset_size.setText(_translate("segasa", str(self.data.sequence.subset_size)))
        self.retranslate()


    def set_peak_boundaries(self):
        """Enables the selection of peak boundaries using Matplotlib RectangleSelector.

        Displays a selection plot for each spectrum of the sequence in a loot
        of length len(self.data.sequence.spectra).
        Selection is performed by using a mouse.
        Selection has to be confirmed by pressing some key, recommended "q".

        PRECONDITIONS:
            -- .mca files were successfully loaded into a sequence
            -- Peak of interest was selected by get_channel

        TRIGGERED BY:
            -- button_set_boundaries click

        COMPLETED BY:
            -- selecting all the boundaries.

        !!! THIS EVENT IS CURRENTLY SUSCEPTIBLE TO UNDEFINED BEHAVIOR IF DONE IMPROPERLY !!!
        """
        self.data.peaks.boundaries = []

        tempBoundaries = []
        for i in range(len(self.data.sequence.spectra)):
            def line_select_callback(eclick, erelease):  # saves selected values
                x1, y1 = eclick.xdata, eclick.ydata
                x2, y2 = erelease.xdata, erelease.ydata
                self.tempBoundaries = [x1, x2]

            def toggle_selector(event):  # selection event

                if event.key and toggle_selector.RS.active:

                    self.data.peaks.boundaries.append(self.tempBoundaries)

                    toggle_selector.RS.set_active(False)

            fig, current_ax = plt.subplots()
            number = plt.gcf().number
            fig.canvas.set_window_title('Set peak boundaries')
            plt.plot(self.data.sequence.spectra[i])
            plt.axis([-20, max(self.data.peaks.found)+20, -20 * self.data.peaks.prominence, self.data.sequence.spectrum[
                    self.data.peaks.found[1]] + 200 * self.data.peaks.prominence])
            plt.draw()  # draw the plot
            toggle_selector.RS = RectangleSelector(current_ax, line_select_callback,
                                                   drawtype='box', useblit=False,
                                                   button=[1, 3],  # don't use middle button
                                                   minspanx=5, minspany=5,
                                                   spancoords='pixels',
                                                   interactive=True)
            plt.connect('key_press_event', toggle_selector)

            if 0 < i < len(self.data.sequence.spectra):
                toggle_selector.RS.to_draw.set_visible(True)
                fig.canvas.draw()
                toggle_selector.RS.extents = (self.data.peaks.boundaries[i-1][0], self.data.peaks.boundaries[i-1][1],
                                              0, self.data.sequence.spectra[i][self.data.sequence.peaks[i]])
            while not plt.waitforbuttonpress():
                continue
            if not self.data.peaks.boundaries[i][0] < self.data.sequence.peaks[i] < self.data.peaks.boundaries[i][1]:

                msg = QtWidgets.QMessageBox()
                msg.setIcon(QtWidgets.QMessageBox.Warning)
                msg.setText("Analyzed Peak wasn't found in specified boundaries\n")
                msg.setInformativeText('Make sure that the analyzed peak is within the boundaries\n'
                                       'All your selections have been deleted to avoid program instability')
                msg.setWindowTitle("Warning")
                msg.exec_()
                self.data.peaks.boundaries = []
                break
            if plt.fignum_exists(number):
                plt.close()
        print(self.data.peaks.boundaries)

    def get_results(self):
        """Displays time dependency plot of evaluated spectrometric parameters and prepares the results for export.

        Performing the calibration and boundary selection beforehand provides more results.
        Executing get_results disables all interaction with the main window except for loading and exporting.

        PRECONDITIONS:
            -- .mca files were successfully loaded into a sequence
            -- the Peak of interest (first selected peak) must be selected by get_channel
            -- OPTIONAL -- if calibration is performed, centroids and fwhms can be calculated in keV
            -- OPTIONAL -- if boundaries are selected, results_alt is calculated for heights and fwhms and both result
            lists are created for the peak areas

        TRIGGERED BY:
            -- button_show_results click
        """
        print(self.data.peaks.boundaries)
        self.data.fwhms = FWHM()
        self.data.centroids = PeakCentroid()
        self.data.heights = PeakHeight()
        self.data.areas = PeakArea()
        if self.data.peaks.boundaries:
            print("tra;a;a")
            self.data.peaks.boundaries_to_int()




        # improving readability
        starts = self.data.peaks.starts
        ends = self.data.peaks.ends
        spectra = self.data.sequence.spectra
        peaks = self.data.sequence.peaks
        print(peaks)

        if len(starts) == len(spectra) and len(ends) == len(spectra):
            print(self.data.heights)
            self.data.heights.calculate_heights_alt(starts, ends, spectra, peaks)
            self.data.fwhms.calculate_fwhms_alt(starts, ends, spectra, peaks)
            self.data.areas.calculate_areas(starts, ends, spectra)
        self.data.heights.calculate_heights(spectra, peaks)
        self.data.fwhms.calculate_fwhms(spectra, peaks)
        print(spectra, peaks)
        self.data.centroids.calculate_centroids(spectra, peaks)
        self.data.centroids.calculate_centroids_alt(spectra, peaks)

        centroids, centroids_alt = self.data.centroids.create_ndarrays()
        fwhm1, fwhm2 = self.data.fwhms.create_ndarrays()
        areas, areas_alt = self.data.areas.create_ndarrays()
        heights, heights_alt = self.data.heights.create_ndarrays()
        gauss_error = []
        fwhm1_error = []
        fwhm2_error = []

        x_result = []
        for i in range(1, len(self.data.sequence.spectra)+1):
            x_result.append(i * self.data.loaded_data.time
                            + (self.data.sequence.subset_size-1) * self.data.loaded_data.time)

        for i in range(0, len(self.data.sequence.spectra)):
            if centroids_alt[i] == 0:
                if 0 < i < (len(centroids_alt) - 1):
                    centroids_alt[i] = np.mean(
                        [centroids_alt[i - 1], centroids_alt[i + 1]])
                gauss_error.append(centroids_alt[i])
            else:
                gauss_error.append(0)
            if fwhm1[i] == 0:
                if 0 < i < len(fwhm1) - 1:
                    fwhm1[i] = np.mean([fwhm1[i - 1], fwhm1[i + 1]])
                fwhm1_error.append(fwhm1[i])
                print("ok")
            else:
                fwhm1_error.append(0)

        if fwhm2.any():
            for i in range(0, len(self.data.sequence.spectra)):
                if fwhm2[i] == 0:
                    if 0 < i < len(fwhm2) - 1:
                        fwhm2[i] = np.mean([fwhm2[i - 1], fwhm2[i + 1]])
                    fwhm2_error.append(fwhm2[i])
                    print("ok")
                else:
                    fwhm2_error.append(0)

        print(gauss_error)
        print(fwhm1_error)
        print(fwhm2_error)

        fig, ax = plt.subplots(2, 2, figsize=(18, 10))
        manager = plt.get_current_fig_manager()
        manager.window.showMaximized()
        manager.set_window_title('Results')

        ax[0, 0].plot(x_result, self.data.calibration.tangent * centroids + self.data.calibration.offset,
                      color="g", label="Five-channel method")
        ax[0, 0].plot(x_result,
                      self.data.calibration.tangent * centroids + self.data.calibration.offset,
                      "x", color="g", markersize=5)
        ax[0, 0].set_title('Centroid')
        if self.data.calibration.tangent != 1:
            ax[0, 0].set(ylabel='energy [keV]', xlabel='time [seconds]')
        else:
            ax[0, 0].set(ylabel='channel', xlabel='time [seconds]')
        ax[0, 0].plot(x_result,
                      self.data.calibration.tangent * centroids_alt + self.data.calibration.offset, color="r",
                      label="Peak apex Gaussian fit")
        ax[0, 0].plot(x_result,
                      self.data.calibration.tangent * centroids_alt + self.data.calibration.offset, "x", color="r",
                      markersize=5)
        ax[0, 0].legend(loc="lower left")
        ax[0, 0].plot()

        ax[0, 1].plot(x_result, self.data.calibration.tangent * fwhm1, color="g",
                      label="Analytical interpolation with background")
        ax[0, 1].plot(x_result, self.data.calibration.tangent * fwhm1, "x", color="g", markersize=5)
        if fwhm2.any():
            ax[0, 1].plot(x_result, self.data.calibration.tangent * fwhm2, color="r",
                          label="Analytical interpolation with background subtraction")
            ax[0, 1].plot(x_result, self.data.calibration.tangent * fwhm2, color="r", marker="x", markersize=5)
        ax[0, 1].legend(loc="upper left")
        ax[0, 1].set_title('FWHM')
        if self.data.calibration.tangent != 1 or self.data.calibration.offset != 0:
            ax[0, 1].set(ylabel='FWHM [keV]', xlabel='time [seconds]')
            self.data.export.unit = "[keV]"
        else:
            ax[0, 1].set(ylabel='FWHM [channel]', xlabel='time [seconds]')

        ax[1, 0].plot(x_result, heights, color="g", label="Peak height with background")
        ax[1, 0].plot(x_result, heights, "x", color="g", markersize=5)
        if areas_alt.any():
            ax[1, 0].plot(x_result, heights_alt, color="r", label="Peak height with background substraction")
            ax[1, 0].plot(x_result, heights_alt, "x", color="r", markersize=5)
            ax[1, 1].plot(x_result, areas, color="g", label="Gross area")
            ax[1, 1].plot(x_result, areas, "x", color="g", markersize=5)
            ax[1, 1].plot(x_result, areas_alt, color="r", label="Net area")
            ax[1, 1].plot(x_result, areas_alt, "x", color="r", markersize=5)

            ax[1, 1].set_title('Peak area')
            ax[1, 1].set(ylabel='counts', xlabel='time [seconds]')

            ax[1, 1].legend(loc="lower left")
            ax[1, 1].set_title('Peak area')
            ax[1, 1].set(ylabel='counts', xlabel='time [seconds]')
        ax[1, 0].legend(loc="upper right")
        ax[1, 0].set_title('Peak height')
        ax[1, 0].set(ylabel='counts', xlabel='time [seconds]')
        print(len(fwhm2))
        print(len(fwhm1))

        for i in range(0, len(fwhm1_error)):
            if fwhm1_error[i]:
                if self.data.calibration.tangent != 1:
                    ax[0, 1].plot(i * self.data.loaded_data.time
                                  + self.data.loaded_data.time*self.data.sequence.subset_size,
                                  self.data.calibration.tangent * fwhm1_error[i], "o", color="r")
                else:
                    ax[0, 1].plot(i * self.data.loaded_data.time
                                  + self.data.loaded_data.time*self.data.sequence.subset_size,
                                  self.data.calibration.tangent * fwhm1_error[i], "o", color="r")
        for i in range(0, len(fwhm2_error)):
            if fwhm2_error[i]:
                if self.data.calibration.tangent != 1:
                    ax[0, 1].plot(i * self.data.loaded_data.time
                                  + self.data.loaded_data.time*self.data.sequence.subset_size,
                                  self.data.calibration.tangent * fwhm2_error[i], "o", color="r")
                else:
                    ax[0, 1].plot(i * self.data.loaded_data.time
                                  + self.data.loaded_data.time*self.data.sequence.subset_size,
                                  self.data.calibration.tangent * fwhm2_error[i], "o", color="r")
        for i in range(0, len(gauss_error)):
            if gauss_error[i]:
                ax[0, 0].plot(i * self.data.loaded_data.time
                              + self.data.loaded_data.time*self.data.sequence.subset_size,
                              self.data.calibration.tangent * gauss_error[i] + self.data.calibration.offset,
                              "o",
                              color="r", label="gaussian fit failure")

        export = []
        export_preparation_1 = []
        export_preparation_2 = []
        export_preparation_3 = []
        for i in range(1, len(self.data.sequence.spectra)+1):
            export_preparation_1.append(0)
            export_preparation_2.append(str((i*self.data.loaded_data.time)
                                        - self.data.loaded_data.time
                                        + self.data.loaded_data.time*self.data.sequence.subset_size))
            export_preparation_3.append(i)

        fwhm2_kev = []
        fwhm1_kev = []
        centroids_kev = []
        centroids_alt_kev = []
        calibration_flag = 0
        if self.data.calibration.tangent > 1 or self.data.calibration.offset != 0:
            calibration_flag = 1
            self.data.export.unit = "[keV]"
            if areas_alt.any() and fwhm2.any() and heights_alt.any():
                fwhm2_kev = fwhm2 * self.data.calibration.tangent
            fwhm1_kev = fwhm1 * self.data.calibration.tangent
            centroids_kev = centroids * self.data.calibration.tangent + self.data.calibration.offset
            centroids_alt_kev = centroids_alt * self.data.calibration.tangent + self.data.calibration.offset
        else:
            self.data.export.unit = "[channel]"

        fwhm2_percentages = []
        for i in range(0, len(fwhm2)):
            fwhm2_percentages.append(fwhm2[i]*100/(self.data.peaks.boundaries[i][1] - self.data.peaks.boundaries[i][0]))

        if areas_alt.any() and fwhm2.any() and heights_alt.any() and calibration_flag == 1:
            export = [export_preparation_3,
                      export_preparation_2,
                      centroids, centroids_kev,
                      centroids_alt, centroids_alt_kev,
                      fwhm1, fwhm1_kev,
                      fwhm2, fwhm2_kev,
                      fwhm2_percentages,
                      heights, heights_alt,
                      areas, areas_alt]
        elif areas_alt.any() and fwhm2.any() and heights_alt.any() and calibration_flag == 0:
            export = [export_preparation_3,
                      export_preparation_2,
                      centroids, export_preparation_1,
                      centroids_alt, export_preparation_1,
                      fwhm1, export_preparation_1,
                      fwhm2, export_preparation_1,
                      fwhm2_percentages,
                      heights, heights_alt,
                      areas, areas_alt]
        elif calibration_flag == 1:
            export = [export_preparation_3,
                      export_preparation_2,
                      centroids, centroids_kev,
                      centroids_alt, centroids_alt_kev,
                      fwhm1, fwhm1_kev,
                      export_preparation_1, export_preparation_1,
                      export_preparation_1,
                      heights, export_preparation_1,
                      export_preparation_1, export_preparation_1]
        else:
            export = [export_preparation_3,
                      export_preparation_2,
                      centroids, export_preparation_1,
                      centroids_alt, export_preparation_1,
                      fwhm1, export_preparation_1,
                      export_preparation_1, export_preparation_1,
                      export_preparation_1,
                      heights, export_preparation_1,
                      export_preparation_1, export_preparation_1]
        print(export)
        self.button_set_boundaries.setEnabled(False)
        self.button_show_results.setEnabled(False)
        self.button_calibrate.setEnabled(False)
        self.slider_subset_position.setEnabled(False)
        self.slider_subset_size.setEnabled(False)
        self.data.export.data = export
        print(self.data.peaks.boundaries)
        self.button_export_results.setEnabled(True)

    def export_results(self):
        """Saves result in a .txt file.

        Parameters that were not evaluated are set to 0 for all spectra of the sequence.

        PRECONDITIONS:
            -- get_results was executed

        TRIGGERED BY:
            -- button_export_results click
        """
        header = f"Number of loaded files: {self.data.loaded_data.file_count}\n" \
                 + f"Calibration plot equation: {self.data.calibration.tangent}x " \
                   f"+ {self.data.calibration.offset}\n" \
                 + f"Number of files summed in one spectrum: {self.data.sequence.subset_size}\n\n"

        table_head = "Columns should be interpreted as follows:\n" \
                     + "1. Position in sequence\n" \
                     + "2. Time of acquisition [seconds]\n" \
                     + f"3. Centroid - Five-channel method [channel]\n" \
                     + f"4. Centroid - Five-channel method [kiloelectron volts]\n" \
                     + f"5. Centroid - Gaussian fit [channel]\n" \
                     + f"6. Centroid - Gaussian fit [kiloelectron volts]\n" \
                     + f"7. FWHM - analytic interpolation with background [channel]\n" \
                     + f"8. FWHM - analytic interpolation with background [kiloelectron volts]\n" \
                     + f"9. FWHM - analytic interpolation without background [channel]\n" \
                     + f"10. FWHM - analytic interpolation without background [kiloelectron volts]\n" \
                     + f"11. FWHM - analytic interpolation without background [percentage of peak width]\n" \
                     + "12. Peak height with background [counts]\n" \
                     + f"13. Peak height without background [counts]\n" \
                     + "14. Gross area [counts]\n" \
                     + f"15. Net area [counts]\n" \
                     + "\n"
        export_string = "Position" + 2*" " \
                        + "Time [s]" + 2*" " \
                        + "Cen-5ch [ch]" + 3*" " \
                        + "Cen-5ch [keV]" + 2*" " \
                        + "Cen-G [ch]" + 5*" " \
                        + "Cen-G [keV]" + 4*" " \
                        + "FWHM+b [ch]" + 4*" " \
                        + "FWHM+b [keV]" + 3*" " \
                        + "FWHM-b [ch]" + 4*" " \
                        + "FWHM-b [keV]" + 3*" " \
                        + "FWHM [%]" + 7*" " \
                        + "PH+b [counts]" + 2*" " \
                        + "PH-b [counts]" + 2*" " \
                        + "GA [counts]" + 4*" " \
                        + "NA [counts]" + 4*" " + "\n"
        for i in range(len(self.data.sequence.spectra)):
            counter = 0
            for j in self.data.export.data:
                counter += 1
                if counter > 2:
                    export_string += f"{float(j[i]):.5f}"
                    if len(f"{j[i]:.5f}") < 15:
                        export_string += (15-len(f"{j[i]:.5f}"))*" "
                else:
                    export_string += str(j[i]) + (10-len(str(j[i])))*" "
            export_string += "\n"
        name = QtWidgets.QFileDialog.getSaveFileName(None, 'Save File')
        file = open(name[0]+".txt", 'w')
        file.write(header+table_head+export_string)

        #         counter = 0
        # for i in self.data.sequence.spectra:
        #     counter = counter + 1
        #
        # if self.data.calibration.tangent > 1 or self.data.calibration.offset != 0:
        #     if

    def retranslate(self):
        _translate = QtCore.QCoreApplication.translate
        self.label_subset_size_text.setText(_translate("segasa", "Number of .mca files summed in one spectrum"))
        self.label_subset_position_text.setText(_translate("segasa", "Current position in sequence"))
        self.button_load_data.setText(_translate("segasa", "Load data"))
        self.button_show_results.setText(_translate("segasa", "Show results"))
        self.button_set_boundaries.setText(_translate("segasa", "Set peak boundaries"))
        self.button_export_results.setText(_translate("segasa", "Save results in .txt file"))
        self.label_set_size_text.setText(_translate("segasa", "Number of .mca files loaded:"))
        self.label_set_size.setText(_translate("segasa", str(self.data.loaded_data.file_count)))
        self.label_subset_size.setText(_translate("segasa", str(self.data.sequence.subset_size)))
        self.label_subset_position.setText(_translate("segasa", str(self.data.sequence.current_position)))

        self.button_calibrate.setText(_translate("segasa", u"Calibrate"))
        self.label_calibration_header.setText(_translate("segasa", u"Energy calibration"))
        self.label_peak_1.setText(_translate("segasa", u"1st peak (Peak of interest):"))
        self.label_centroid_1.setText(_translate("segasa", str(round(self.data.peaks.selected[0], 3))))
        self.label_kev_1.setText(_translate("segasa", u"keV"))
        self.label_peak_2.setText(_translate("segasa", u"2nd peak:"))
        self.label_centroid_2.setText(_translate("segasa", str(round(self.data.peaks.selected[1], 3))))
        self.label_kev_2.setText(_translate("segasa", u"keV"))
        self.label_peak_3.setText(_translate("segasa", u"3rd peak:"))
        self.label_centroid_3.setText(_translate("segasa", str(round(self.data.peaks.selected[2], 3))))
        self.label_kev_3.setText(_translate("segasa", u"keV"))
        self.label_peak_4.setText(_translate("segasa", u"4th peak:"))
        self.label_centroid_4.setText(_translate("segasa", str(round(self.data.peaks.selected[3], 3))))
        self.label_kev_4.setText(_translate("segasa", u"keV"))
        self.label_peak_5.setText(_translate("segasa", u"5th peak:"))
        self.label_centroid_5.setText(_translate("segasa", str(round(self.data.peaks.selected[4], 3))))
        self.label_kev_5.setText(_translate("segasa", u"keV"))
        self.label_peak_6.setText(_translate("segasa", u"6th peak:"))
        self.label_centroid_6.setText(_translate("segasa", str(round(self.data.peaks.selected[5], 3))))
        self.label_kev_6.setText(_translate("segasa", u"keV"))
        self.label_calibration_function.setText(_translate("segasa", u"Calibration function:"))
        self.label_calibration_function_value.setText(_translate("segasa", u"-"))
        self.label_r_squared.setText(_translate("segasa", u"R-squared:"))
        self.label_r_squared_value.setText(_translate("segasa", u"-"))
