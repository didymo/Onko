from src.View.mainpage.DeleteROIWindow import *
from src.View.mainpage.DrawROIWindow import *
from src.Controller.AddOnOptionsController import *
from pydicom import Dataset


# Create the ROI Delete Options class based on the UI from the file in View/ROI Delete Option
class RoiDeleteOptions(QtWidgets.QMainWindow, UIDeleteROIWindow):
    deleting_rois_structure_tuple = QtCore.pyqtSignal(tuple)  # new PyDicom dataset

    def __init__(self, rois, dataset_rtss):
        super(RoiDeleteOptions, self).__init__()

        self.setup_ui(self, rois, dataset_rtss, self.deleting_rois_structure_tuple)


# The class that will be called by the main page to access the ROI Options controller
class ROIDelOption:

    def __init__(self, rois, dataset_rtss, structure_modified_function):
        super(ROIDelOption, self).__init__()
        self.rois = rois
        self.dataset_rtss = dataset_rtss
        self.structure_modified_function = structure_modified_function

    def show_roi_delete_options(self):
        self.options_window = RoiDeleteOptions(self.rois, self.dataset_rtss)
        self.options_window.deleting_rois_structure_tuple.connect(self.structure_modified_function)
        self.options_window.show()


# Create the ROI Draw Options class based on the UI from the file in View/ROI Draw Option
class RoiDrawOptions(QtWidgets.QMainWindow, UIDrawROIWindow):
    def __init__(self, rois, dataset_rtss):
        super(RoiDrawOptions, self).__init__()
        self.setup_ui(self, rois, dataset_rtss)


# The class that will be called by the main page to access the ROI Options controller
class ROIDrawOption:

    def __init__(self, rois, dataset_rtss):
        super(ROIDrawOption, self).__init__()
        self.rois = rois
        self.dataset_rtss = dataset_rtss

    def show_roi_draw_options(self):
        self.draw_window = RoiDrawOptions(self.rois, self.dataset_rtss)
        self.draw_window.show()
