from PyQt5 import QtCore, QtGui, QtWidgets
import sys
from gui.app_window import AppWindow
from controller.event_controller import EventController


def start_application():
    """Sets up the GUI"""
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


