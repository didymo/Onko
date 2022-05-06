from PySide6 import QtWidgets, QtGui, QtCore
from src.Model.PatientDictContainer import PatientDictContainer
from src.View.ProgressWindow import ProgressWindow
from src.Model.ISO2ROI import ISO2ROI

isodose_percentages = [107, 105, 100, 95, 90, 80, 70, 60, 30, 10]


class IsodoseTab(QtWidgets.QWidget):

    request_update_isodoses = QtCore.Signal()
    request_update_ui = QtCore.Signal(tuple)

    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        self.patient_dict_container = PatientDictContainer()
        self.rx_dose_in_cgray = self.patient_dict_container.get("rx_dose_in_cgray")
        self.color_dict = self.init_color_isod()
        self.color_squares = self.init_color_squares()
        self.checkboxes = self.init_checkboxes()

        # Create and initialise ISO2ROI button and layout
        self.iso2roi_button = QtWidgets.QPushButton()
        self.iso2roi_button.setText("Convert Isodoses to ROIs")
        self.iso2roi_button.clicked.connect(self.iso2roi_button_clicked)

        self.iso2roi_layout = QtWidgets.QHBoxLayout()
        self.iso2roi_layout.setContentsMargins(0, 0, 0, 0)
        self.iso2roi_layout.addWidget(self.iso2roi_button)

        self.isodose_tab_layout = QtWidgets.QVBoxLayout()
        self.isodose_tab_layout.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignTop)
        self.isodose_tab_layout.setSpacing(0)
        self.init_layout()
        self.iso2roi = ISO2ROI()

        # Add button to tab
        self.isodose_tab_layout.addStretch()
        self.isodose_tab_layout.addLayout(self.iso2roi_layout)

        self.setLayout(self.isodose_tab_layout)
        self.progress_window = ProgressWindow(
            self, QtCore.Qt.WindowTitleHint | QtCore.Qt.WindowCloseButtonHint)
        self.progress_window.signal_loaded.connect(self.on_loaded_iso2roi)

    def init_layout(self):
        for i in range(0, len(self.checkboxes)):
            widget_isodose = QtWidgets.QWidget()
            layout_isodose = QtWidgets.QHBoxLayout(widget_isodose)
            layout_isodose.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignLeft)
            layout_isodose.addWidget(self.color_squares[i])
            layout_isodose.addWidget(self.checkboxes[i])
            self.isodose_tab_layout.addWidget(widget_isodose)

    def init_color_isod(self):
        """
        Create a list containing the colors for each isodose.

        :return: Dictionary where the key is the percentage of isodose and the value a QColor object.
        """
        roi_color = {
            107: QtGui.QColor(131, 0, 0),
            105: QtGui.QColor(185, 0, 0),
            100: QtGui.QColor(255, 46, 0),
            95: QtGui.QColor(255, 161, 0),
            90: QtGui.QColor(253, 255, 0),
            80: QtGui.QColor(0, 255, 0),
            70: QtGui.QColor(0, 143, 0),
            60: QtGui.QColor(0, 255, 255),
            30: QtGui.QColor(33, 0, 255),
            10: QtGui.QColor(11, 0, 134)
        }

        return roi_color

    def init_color_squares(self):
        """
        Create a color square.
        """
        list_of_squares = []
        for key, color in self.color_dict.items():
            list_of_squares.append(self.draw_color_square(color))

        return list_of_squares

    def init_checkboxes(self):
        """
        Initialize the checkbox objects.
        """
        list_of_checkboxes = []
        # Values of Isodoses
        list_of_doses = []
        for percentage in isodose_percentages:
            dose = int(self.rx_dose_in_cgray * (percentage / 100))
            list_of_doses.append(dose)

        # Checkboxes
        def generate_clicked_handler(text):
            def handler(state):
                self.checked_dose(state, text)
            return handler
        first_iteration = True
        for i in range(10):
            if first_iteration:
                checkbox = QtWidgets.QCheckBox("%s %% / %s cGy [Max]" % (str(isodose_percentages[i]),
                                                                         str(list_of_doses[i])))
                first_iteration = False
            else:
                checkbox = QtWidgets.QCheckBox("%s %% / %s cGy" % (str(isodose_percentages[i]), str(list_of_doses[i])))
            checkbox.clicked.connect(generate_clicked_handler(isodose_percentages[i]))
            checkbox.setStyleSheet("font: 10pt \"Laksaman\";")
            list_of_checkboxes.append(checkbox)

        return list_of_checkboxes

    # Function triggered when a dose level selected
    # Updates the list of selected isodoses and dicom view
    def checked_dose(self, state, isod_value):
        """
        Function triggered when the checkbox of a structure is checked / unchecked.
        Update the list of selected structures.
        Update the DICOM view.

        :param state: True if the checkbox is checked, False otherwise.
        :param isod_value: Percentage of isodose.
        """

        selected_doses = self.patient_dict_container.get("selected_doses")

        if state:
            # Add the dose to the list of selected doses
            selected_doses.append(isod_value)
        else:
            # Remove dose from list of previously selected doses
            selected_doses.remove(isod_value)

        self.patient_dict_container.set("selected_doses", selected_doses)

        # Update the dicom view
        self.request_update_isodoses.emit()

    def draw_color_square(self, color):
        """
        Create a color square.
        :param color: QColor object
        :return: Color square widget.
        """
        color_square_label = QtWidgets.QLabel()
        color_square_pix = QtGui.QPixmap(15, 15)
        color_square_pix.fill(color)
        color_square_label.setPixmap(color_square_pix)

        return color_square_label

    def iso2roi_button_clicked(self):
        """
        Clicked action handler for the ISO2ROI button.
        Opens a progress window and Initiates the
        ISO2ROI conversion process.
        """
        self.progress_window.start(self.iso2roi.start_conversion)

    def on_loaded_iso2roi(self):
        """
        Called when progress bar has finished.
        Closes the progress window and refreshes
        the main screen.
        """
        self.request_update_ui.emit((
            self.patient_dict_container.get('dataset_rtss'), {"draw": None}))
        self.progress_window.close()
