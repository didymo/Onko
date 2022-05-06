import csv
import platform
from pydicom import dcmread
from PySide6 import QtCore, QtWidgets
from src.Controller.PathHandler import data_path, resource_path
from fuzzywuzzy import process


class ROINameCleaningOptionComboBox(QtWidgets.QComboBox):
    """
    This class inherits QComboBox to create a custom widget for the Name
    Cleaning ROI table that includes two options - modify and delete.
    """

    def __init__(self):
        QtWidgets.QComboBox.__init__(self)
        self.addItem("Ignore")
        self.addItem("Modify")
        self.addItem("Delete")
        self.setObjectName("BatchROICleaning")


class ROINameCleaningOrganComboBox(QtWidgets.QComboBox):
    """
    This class inherits QComboBox to create a custom widget for the Name
    Cleaning ROI table that includes a list of standard organ names.
    """

    def __init__(self, organ_names, volume_prefixes, roi_name):
        """
        Initialises the object, setting the combo box options to be a
        list of the standard organ names.
        :param organs: a list of standard organ names.
        """
        QtWidgets.QComboBox.__init__(self)
        self.setObjectName("BatchROICleaning")

        # Get and add suggested roi names
        roi_suggestions = self.roi_suggestions(roi_name,
                                               organ_names,
                                               volume_prefixes)
        self.addItem(roi_suggestions[0][0])
        self.addItem(roi_suggestions[1][0])
        self.addItem(roi_suggestions[2][0])
        self.insertSeparator(4)

        # Populate combo box options
        for organ in organ_names:
            self.addItem(organ)

    @QtCore.Slot(int)
    def change_enabled(self, index):
        """
        A slot for changing whether this combo box is enabled or
        disabled depending on whether "Modify" or "Delete" has been
        selected for the associated ROI.
        :param index: the new index of the combo box.
        """
        if index == 1:
            self.setEnabled(True)
        else:
            self.setEnabled(False)

    def roi_suggestions(self, roi_name, organ_names, volume_prefixes):
        """
        Get the top 3 suggestions for the selected ROI based on
        string matching with standard ROIs provided in .csv format.

        :return: two dimensional list with ROI name and string match percent
        i.e [('MANDIBLE', 100), ('SUBMAND_L', 59), ('LIVER', 51)]
        """

        roi_list = organ_names + volume_prefixes
        suggestions = process.extract(roi_name, roi_list,
                                      limit=3)  # will get the top 3 matches

        return suggestions


class ROINameCleaningDatasetComboBox(QtWidgets.QComboBox):
    """
    This class inherits QComboBox to create a custom widget for displaying
    a list of RTSS files that contain a listed ROI name. This combobox can
    be dropped down, but nothing can be selected.
    """

    def __init__(self, rtss_list):
        """
        Initialises the object, setting the combo box options to be a
        list of the datasets.
        :param rtss_list: a list RTSTRUCT paths.
        """
        QtWidgets.QComboBox.__init__(self)

        # Populate combo box options
        for rtss in rtss_list:
            self.addItem(rtss)

        self.setObjectName("BatchROICleaning")
        # TODO: make changing option do nothing


class ROINameCleaningPrefixEntryField(QtWidgets.QLineEdit):
    """
    This class inherits QLabel to create a custom widget for the Name
    Cleaning ROI table that allows it to be enabled or disabled when the
    QComboBox in the same row changes. This widget displays an entry field
    for entering in a new name for the ROI.
    """

    def __init__(self):
        """
        Simple sets the object name to properly set styling.
        """
        QtWidgets.QLineEdit.__init__(self)
        self.setObjectName("BatchROICleaning")

    @QtCore.Slot(int)
    def change_enabled(self, index):
        print(index)
        if index == 1:
            self.setEnabled(True)
        else:
            self.setEnabled(False)


class ROINameCleaningOptions(QtWidgets.QWidget):
    """
    ROI Name Cleaning options for batch processing.
    """

    def __init__(self):
        """
        Initialise the class
        """
        QtWidgets.QWidget.__init__(self)

        # Create the main layout
        self.main_layout = QtWidgets.QVBoxLayout()

        # Get the stylesheet
        if platform.system() == 'Darwin':
            self.stylesheet_path = "res/stylesheet.qss"
        else:
            self.stylesheet_path = "res/stylesheet-win-linux.qss"
        self.stylesheet = open(resource_path(self.stylesheet_path)).read()

        # Class variables
        self.organ_names = []
        self.organ_names_lowercase = []
        self.volume_prefixes = []

        self.get_standard_names()
        self.create_table_view()
        self.setLayout(self.main_layout)

    def get_standard_names(self):
        """
        Get standard organ names and prefix types.
        """
        # Get standard organ names
        with open(data_path('organName.csv'), 'r') as f:
            csv_input = csv.reader(f)
            header = next(f)  # Ignore the "header" of the column
            for row in csv_input:
                self.organ_names.append(row[0])
                self.organ_names_lowercase.append(row[0].lower())
            f.close()

        # Get standard volume prefixes
        with open(data_path('volumeName.csv'), 'r') as f:
            csv_input = csv.reader(f)
            header = next(f)  # Ignore the "header" of the column
            for row in csv_input:
                self.volume_prefixes.append(row[1])
            f.close()

    def create_table_view(self):
        """
        Create a table to display all of the non-standard ROIs and
        options for what to do with them.
        """
        # Create table
        self.table_roi = QtWidgets.QTableWidget(self)
        self.table_roi.setStyleSheet(
            "background-color: rgb(255, 255, 255);")
        self.table_roi.setColumnCount(4)
        self.table_roi.verticalHeader().hide()
        # Note - "New Name" is only enabled if the option "Rename" is
        # selected.
        self.table_roi.setHorizontalHeaderLabels(
            [" ROI Name ", " Option ", " New Name ", " Dataset Location "])

        # Set text align
        self.table_roi.horizontalHeaderItem(0).setTextAlignment(
            QtCore.Qt.AlignLeft)
        self.table_roi.horizontalHeaderItem(1).setTextAlignment(
            QtCore.Qt.AlignLeft)
        self.table_roi.horizontalHeaderItem(2).setTextAlignment(
            QtCore.Qt.AlignLeft)
        self.table_roi.horizontalHeaderItem(3).setTextAlignment(
            QtCore.Qt.AlignLeft)

        # Set header stretch
        roi_name_header = self.table_roi.horizontalHeader()
        roi_name_header.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        roi_name_header.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        roi_name_header.setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)
        roi_name_header.setSectionResizeMode(3, QtWidgets.QHeaderView.Stretch)

        # Removing the ability to edit tables with immediate click
        self.table_roi.setEditTriggers(
            QtWidgets.QTreeView.NoEditTriggers |
            QtWidgets.QTreeView.NoEditTriggers)

        # Add table to the main layout
        self.main_layout.addWidget(self.table_roi)

    def populate_table(self, dicom_structure, batch_directory):
        """
        Populates the table with ROI names and options once datasets
        have been loaded. Called when datasets have finished loading.
        :param dicom_structure: DICOM structure object representing all
                                patients loaded.
        :param batch_directory: The directory selected for batch processing.
        """
        # Update table column view
        roi_name_header = self.table_roi.horizontalHeader()
        roi_name_header.setSectionResizeMode(
            0, QtWidgets.QHeaderView.Stretch)
        roi_name_header.setSectionResizeMode(
            1, QtWidgets.QHeaderView.ResizeToContents)
        roi_name_header.setSectionResizeMode(
            2, QtWidgets.QHeaderView.ResizeToContents)
        roi_name_header.setSectionResizeMode(
            3, QtWidgets.QHeaderView.ResizeToContents)

        # Loop through each patient, get every RTSTRUCT
        rtstruct_list = []
        for patient in dicom_structure.patients:
            studies = dicom_structure.patients[patient].studies
            for study in studies:
                rtstructs = studies[study].rtstructs
                for rtss in rtstructs:
                    images = rtstructs[rtss].images
                    for image in images:
                        if images[image].class_id == \
                                '1.2.840.10008.5.1.4.1.1.481.3':
                            rtstruct_list.append(images[image].path)

        # Return if no RT Structs found
        if not len(rtstruct_list):
            self.table_roi.setRowCount(0)
            return

        # Loop through each RT Struct
        # Structure of rois dictionary:
        #   key: ROI name
        #   value: a list of dataset paths containing this ROI
        rois = {}
        for rtss in rtstruct_list:
            rtstruct = dcmread(rtss)
            # Loop through each ROI in the RT Struct
            for i in range(len(rtstruct.StructureSetROISequence)):
                # Get the ROI name
                roi_name = rtstruct.StructureSetROISequence[i].ROIName

                # Add ROI name to the dictionary
                if roi_name not in rois.keys():
                    rois[roi_name] = []

                # Add dataset to the list if the ROI name is not a
                # standard organ name or has a standard prefix
                if roi_name not in self.organ_names or \
                        roi_name[0:3] in self.volume_prefixes:
                    rois[roi_name].append(rtss)

        # Return if no ROIs found
        rois_to_process = False
        for roi in rois:
            if len(rois[roi]):
                rois_to_process = True
                break

        if not rois_to_process:
            self.table_roi.setRowCount(0)
            return

        # Populate table
        self.table_roi.setRowCount(0)

        # Loop through each ROI
        i = 0
        for roi_name in rois:
            # Create option combo box
            combo_box = ROINameCleaningOptionComboBox()
            combo_box.setStyleSheet(self.stylesheet)

            # Create dataset combo box
            # Get list of RTSTRUCTs
            dataset_list = rois[roi_name]
            if len(dataset_list) == 0:
                continue

            # Remove common path from RTStructs
            for index in range(len(dataset_list)):
                dataset_list[index] = \
                    dataset_list[index].replace(batch_directory, '')

            rtss_combo_box = ROINameCleaningDatasetComboBox(dataset_list)
            rtss_combo_box.setStyleSheet(self.stylesheet)

            # Create text entry field the ROI has a standard prefix.
            # Generate organ combobox otherwise.
            if roi_name[0:3] in self.volume_prefixes \
                    or roi_name[0:4] in self.volume_prefixes:
                name_box = ROINameCleaningPrefixEntryField()
                name_box.setEnabled(False)
            else:
                name_box = \
                    ROINameCleaningOrganComboBox(self.organ_names,
                                                 self.volume_prefixes,
                                                 roi_name)
                # Set default combo box entry to organ name in proper case
                # if the organ name is a standard one.
                if roi_name.lower() in self.organ_names_lowercase:
                    index = self.organ_names_lowercase.index(roi_name.lower())
                    name_box.setCurrentIndex(index)
                    name_box.setEnabled(True)
                    combo_box.setCurrentIndex(1)
                else:
                    name_box.setEnabled(False)

            combo_box.currentIndexChanged.connect(name_box.change_enabled)
            name_box.setStyleSheet(self.stylesheet)

            # Add row to table
            self.table_roi.insertRow(i)
            self.table_roi.setRowHeight(i, 50)
            self.table_roi.setItem(
                i, 0, QtWidgets.QTableWidgetItem(roi_name))
            self.table_roi.setCellWidget(i, 1, combo_box)
            self.table_roi.setCellWidget(i, 2, name_box)
            self.table_roi.setCellWidget(i, 3, rtss_combo_box)
            i += 1

        # Set row height
        vertical_header = self.table_roi.verticalHeader()
        vertical_header.setSectionResizeMode(QtWidgets.QHeaderView.Fixed)
        vertical_header.setDefaultSectionSize(40)
