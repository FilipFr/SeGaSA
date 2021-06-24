from PyQt5 import QtCore, QtGui, QtWidgets
import sys
from gui.app_window import AppWindow
from controller.event_controller import EventController


"""The main module of the SeGaSA system.

SeGaSA is a software for sequential gamma spectra analysis developed at STU as a part of a bachelor's thesis.
For instructions on how to use the software refer to the user manual at https://github.com/FilipFr/SeGaSA
Running this module starts the SeGaSA application."""


def start_application():
    """Sets up the GUI and starts the application"""
    app = QtWidgets.QApplication(sys.argv)
    window = QtWidgets.QMainWindow()
    # ui = AppWindow()
    # ui.setupUi(window)
    ui = EventController()
    ui.setupUi(window)
    print()
    window.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    start_application()


