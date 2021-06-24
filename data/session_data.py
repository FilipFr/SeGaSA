from os import path, walk
from controller import utility
from scipy.optimize import curve_fit
import sys
import numpy as np
sys.path.append("..")  # Adds higher directory to python modules path.

"""The main Model component module of the SeGaSA system.

Contains data classes.

CLASSES:

    SessionData
        -- data on the current state of the application session.
    Peak
        -- data on peaks found and selected in the currently analyzed sequence
    Calibration
        -- calibration data
    Sequence
        -- data on the currently analyzed sequence
    MCAData
        -- data parsed from loaded files
    Result 
        -- stores results of the spectra evaluation
    PeakCentroid
        -- peak centroid evaluation result data
    FWHM
        -- fwhm evaluation result data
    PeakHeight
        -- peak height evaluation result data
    PeakArea
        -- peak area evaluation result data
    Export
        -- result data intended to be exported 
"""


class SessionData:
    """A class storing the current state of the application.

    Contains instances of data classes.
    This class represents the Model component of the SeGaSA system.
    Instances are maintained for the duration of a single application session.
    Check docstrings of respective data classes for their specification.

    ATTRIBUTES:

        loaded_data : MCAData()
        sequence : Sequence()
        peaks : Peak()
        calibration : Calibration()
        fwhms : FWHMs()
        centroids : PeakCentroids()
        heights : PeakHeights()
        areas : PeakAreas()

    METHODS:

        reset() : void
            -- resets the application to a partially uninitialized state.
    """

    def __init__(self):
        self.loaded_data = MCAData()
        self.sequence = Sequence()
        self.peaks = Peak()
        self.calibration = Calibration()
        self.fwhms = FWHM()
        self.centroids = PeakCentroid()
        self.heights = PeakHeight()
        self.areas = PeakArea()
        self.export = Export()

    def reset(self):
        """Returns application to a partially uninitialized state.

        Loaded data remains unaffected."""
        self.sequence = Sequence()
        self.peaks = Peak()
        self.calibration = Calibration()
        self.fwhms = FWHM()
        self.centroids = PeakCentroid()
        self.heights = PeakHeight()
        self.areas = PeakArea()
        self.export = Export()


class Peak:
    """A class storing the data of peaks found and selected in the currently analyzed sequence.

    ATTRIBUTES:
        found : list of integers
            -- peaks found in the current spectrum using SciPy function find_peaks()
        selected : list of floats
            -- peaks selected by the user
        prominence : integer
            -- current level of prominence, identical to the spectrum size (possible redundancy?)
        boundaries : list of lists of floats
            -- currently selected peak boundaries
        starts : list of integers
            -- starting channels of the currently selected peak for the whole sequence
        ends : list of integers
            -- ending channels of the currently selected peak for the whole sequence

    METHODS:
        boundaries_to_int() : void
            -- converts the list of floats representing peak boundaries into two lists of integers
    """

    def __init__(self):
        self.found = []
        self.selected = [0, 0, 0, 0, 0, 0]
        self.prominence = 0
        self.boundaries = []
        self.starts = []
        self.ends = []

    def boundaries_to_int(self):
        """Converts the tuple of floats representing peak boundaries into two lists of integers."""
        self.starts = []
        self.ends = []
        for boundary in self.boundaries:
            if boundary[0] > 5 and boundary[1] < 1020:
                self.starts.append(int(boundary[0]))
                self.ends.append(int(boundary[1]))


class Calibration:
    """Contains calibration data.

    ATTRIBUTES:
        tangent : float
        offset : float
        r_squared : float (on the interval [0-1])
            -- coefficient of determination
    """

    def __init__(self):
        self.tangent = 1
        self.offset = 0
        self.r_squared = 0


class Sequence:
    """Contains data of the currently analyzed sequence.

    ATTRIBUTES:
        subset_size : integer
            -- number of summed .mca files in one spectrum of the sequence
        current_position : integer
            -- index of the currently displayed spectrum of the sequence (indexing from 1)
        spectra : list of lists of integers
            -- all spectra of the sequence
        length : integer
            -- number of spectra in the analyzed sequence
        peaks : list of integers
            -- list of channel numbers of the analyzed peak maximum
        spectrum : list of integers
            -- currently selected spectrum
    """

    def __init__(self):
        self.subset_size = 0
        self.current_position = 0

        self.spectra = []
        self.length = 0
        self.peaks = []
        self.spectrum = []


class MCAData:
    """A class to contain and manage the imported data from .mca files.

    ATTRIBUTES:
        _spectra : list of lists
            -- contains the spectra (spectrum : list of integers) imported from .mca files
        _time : integer
            -- the duration of a single spectrum acquisition

    METHODS:
        parse_data_from_mca(filepath)
            -- parses and imports relevant data from the specified file
        get_data_from_directory(directory_path)
            -- calls parse_data_from_mca on each file in the directory
    """

    def __init__(self):
        self._file_count = 0
        self._spectra = []
        self._time = 0

    @property
    def file_count(self):
        return self._file_count

    @property
    def spectra(self):
        return self._spectra

    @property
    def time(self):
        return self._time

    def parse_data_from_mca(self, filepath):
        """Parses data from a single .mca file.

        PARAMETERS:
            filepath : string
        """

        initial_string = "<<DATA>>"
        terminal_string = "<<END>>"
        time_marker = "LIVE_TIME"

        parse_flag = False
        parsed_data = []

        if path.isfile(filepath) and ".mca" in filepath and "v.mca" not in filepath:
            with open(filepath) as mca_file:
                for line in mca_file:
                    if time_marker in line:
                        print(line)
                        if len(line.split()) > 2:
                            self._time = line.split()[2]
                    if initial_string in line:
                        parse_flag = True
                        continue
                    elif terminal_string in line:
                        break
                    if parse_flag:
                        parsed_data.append(int(line))
        else:
            self._spectra = []
        if parsed_data:
            print(parsed_data)
            self._spectra.append(parsed_data)
            self._file_count += 1
            print("ok")

    def get_data_from_directory(self, directory_path):
        """Calls parse_data_from_mca on each file in the directory.

        PARAMETERS:
            directory_path : string

        RETURNS:
            integer (0 on success, -1 on failiure)
        """

        files = []
        self._file_count = 0
        self._spectra = []
        self._time = 0
        for root, directories, filenames in walk(directory_path):
            for name in filenames:
                if ".mca" in name and "v.mca" not in name:
                    files.append(directory_path + '\\' + name)
        for file in files:
            self.parse_data_from_mca(file)
        if self._time:
            self._time = int(float(self._time))
        if not self._spectra:
            return -1
        return 0


class Result:
    """A data class representing spectra evaluation results.

    This class is intended to be subclassed by classes of specific spectrometric parameters.

    ATTRIBUTES:
        _results : list of integers / list of floats
            -- contains evaluation of a specific spectrometric parameter for all loaded spectra
        _results_alt : list of integers / list of floats
            -- contains alternative evaluation of a specific parameter for all loaded spectra

    METHODS:
        create_ndarrays() : ndarray, ndarray (see NumPy type ndarray)
            -- returns NumPy arrays made from results
    """

    def __init__(self):
        self._results = []
        self._results_alt = []

    def create_ndarrays(self):
        """Returns NumPy arrays made from results.

        RETURNS:
            ndarray, ndarray
        """
        return np.array(self._results), np.array(self._results_alt)


class PeakCentroid(Result):
    """A subclass of Results representing peak centroids.

    METHODS:
        calculate_centroids()
            -- uses five-channel method to evaluate current spectra and produces a list of centroids
        calculate_centroids_alt()
            -- uses gaussian fitting on top of the peak of interest to produce a list of centroids
    """
    def __init__(self):
        super().__init__()

    def calculate_centroids(self, sequence, peaks):
        """Uses five-channel method to evaluate current spectra and produces a list of centroids.

        The operation is done on all spectra of the sequence, each result is appended to the respective result list.

        PARAMETERS:
            sequence : list of lists of integers
                -- all analyzed spectra
            peaks : list of integers
                -- list of channel numbers of the analyzed peak maximum
        """
        for i in range(0, len(sequence)):
            self._results.append(utility.estimate_centroid(sequence[i], peaks[i]))
        print(self._results)

    def calculate_centroids_alt(self, sequence, peaks):
        """Uses gaussian fitting on top of the peak of interest to produce a list of centroids.

        The operation is done on all spectra of the sequence, each result is appended to the respective result list.

        PARAMETERS:
            sequence : list of lists of integers
                -- all analyzed spectra
            peaks : list of integers
                -- list of channel numbers of the analyzed peak maximum
        """
        for i in range(0, len(sequence)):
            x = []
            y = []
            for j in range(5):
                y.append(sequence[i][peaks[i] - 2 + j])
                x.append(j)
            try:
                popt, _ = curve_fit(utility.gaussian_model, x, y)
                a, mu, sigma = popt
                self._results_alt.append(mu + peaks[i] - 2)
            except:
                self._results_alt.append(0)
        print(self._results_alt)


class FWHM(Result):
    """A subclass of Results representing peak FWHMs (Full Width at Half Maximum).

    METHODS:
        calculate_fwhms()
            -- uses analytic interpolation without background subtraction to produce a list of fwhms
        calculate_fwhms_alt() !!! REQUIRES PEAK BOUNDARIES TO BE SET
            -- uses analytic interpolation with background subtraction to produce a list of fwhms
    """
    def __init__(self):
        super().__init__()

    def calculate_fwhms(self, sequence, peaks):
        """Uses analytic interpolation without background subtraction to produce a list of fwhms.

        The operation is done on all spectra of the sequence, each result is appended to the respective result list.

        PARAMETERS:
            sequence : list of lists of integers
                -- all analyzed spectra
            peaks : list of integers
                -- list of channel numbers of the analyzed peak maximum
        """
        for i in range(0, len(sequence)):
            x = []
            y = []
            for j in range(5):
                y.append(sequence[i][peaks[i] - 2 + j])
                x.append(j)

            try:
                self._results.append(utility.estimate_fwhm(sequence[i], peaks[i], 0.5))
            except:
                self._results.append(0)
        print(self._results)

    def calculate_fwhms_alt(self, starts, ends, sequence, peaks):
        """Uses analytic interpolation with background subtraction to produce a list of fwhms.

        !!! REQUIRES PEAK BOUNDARIES TO BE SET
        The operation is done on all spectra of the sequence, each result is appended to the respective result list.

        PARAMETERS:
            starts : list of integers
                -- starting channel of the analyzed peak
            ends : list of integers
                -- ending channel of the analyzed peak
            sequence : list of lists of integers
                -- all analyzed spectra
            peaks : list of integers
                -- list of channel numbers of the analyzed peak maximum
        """
        self._results_alt = []
        for i in range(0, len(sequence)):

            y_subset = []

            x_difference = ends[i] - starts[i]
            y_difference = sequence[i][ends[i]] - sequence[i][starts[i]]
            tangent = y_difference / x_difference
            for j in range(starts[i], ends[i] + 1):
                y_subset.append(sequence[i][j]
                                - (sequence[i][starts[i]]
                                   + tangent * (j - starts[i])))

            try:
                self._results_alt.append(
                    utility.estimate_fwhm(y_subset, peaks[i] - starts[i], 0.5))
            except:
                self._results_alt.append(0)
        print(self._results_alt)


class PeakHeight(Result):
    """A subclass of Results representing peak heights.

    METHODS:
        calculate_heights(sequence, peaks)
            -- saves the maximum counts of the peak to the result list
        calculate_heights_alt(starts, ends, sequence, peaks) !!! REQUIRES PEAK BOUNDARIES TO BE SET
            -- saves the maximum counts of the peak to the result list (with background subtracted)
    """
    def __init__(self):
        super().__init__()

    def calculate_heights(self, sequence, peaks):
        """Saves the maximum counts of the peak to the result list.

        The operation is done on all spectra of the sequence, each result is appended to the respective result list.

        PARAMETERS:
            sequence : list of lists of integers
                -- all analyzed spectra
            peaks : list of integers
                -- list of channel numbers of the analyzed peak maximum
        """
        self._results = []
        for index in range(len(sequence)):
            self._results.append(sequence[index][peaks[index]])
        print(self._results)

    def calculate_heights_alt(self, starts, ends, sequence, peaks):
        """Saves the maximum counts (with background subtracted) of the peak to the result list.

        !!! REQUIRES PEAK BOUNDARIES TO BE SET
        The operation is done on all spectra of the sequence, each result is appended to the respective result list.

        PARAMETERS:
            starts : list of integers
                -- starting channel of the analyzed peak
            ends : list of integers
                -- ending channel of the analyzed peak
            sequence : list of lists of integers
                -- all analyzed spectra
            peaks : list of integers
                -- list of channel numbers of the analyzed peak maximum
        """
        for i in range(0, len(sequence)):
            x_difference = ends[i] - starts[i]
            y_difference = sequence[i][ends[i]] - \
                sequence[i][starts[i]]
            tangent = y_difference / x_difference
            for j in range(starts[i], ends[i] + 1):

                if j == peaks[i]:
                    self._results_alt.append(sequence[i][j]
                                             - (sequence[i][starts[i]]
                                             + tangent * (j - starts[i])))
        print(self._results_alt)


class PeakArea(Result):
    """A subclass of Results representing peak areas.

    METHODS:
        calculate_areas(starts, ends, sequence) !!! REQUIRES PEAK BOUNDARIES TO BE SET
            -- sums the counts between boundaries with and without background subtraction
    """
    def __init__(self):
        super().__init__()

    def calculate_areas(self, starts, ends, sequence):
        """Sums the counts between boundaries with and without background subtraction.

        !!! REQUIRES PEAK BOUNDARIES TO BE SET
        The operation is done on all spectra of the sequence, each result is appended to the respective result list.

        PARAMETERS:
            starts : list of integers
                -- starting channel of the analyzed peak
            ends : list of integers
                -- ending channel of the analyzed peak
            sequence : list of lists of integers
                -- all analyzed spectra
        """

        self._results = []
        self._results_alt = []

        for i in range(0, len(sequence)):
            count = 0
            for j in range(starts[i], ends[i] + 1):
                count = count + sequence[i][j]
            self._results_alt.append(count - (int(((sequence[i][starts[i]] +
                                                    sequence[i][ends[i]]) * (ends[i] - starts[i]) / 2))))
            self._results.append(count)
        print(self._results)
        print(self._results_alt)


class Export:
    """Contains all result data in a matrix.

    ATTRIBUTES:
        unit : string
            -- unit of centroids and fwhms
        data : list of lists of floats/integers
            -- contains list of spectrum numbers, list of times of acquisition.
            -- additionally contains lists of all the result parameters
    """
    def __init__(self):
        self.unit = "channel"
        self.data = []
