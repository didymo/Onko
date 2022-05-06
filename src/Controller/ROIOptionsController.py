from src.View.ImageFusion.UITransferROIWindow import UITransferROIWindow
from src.View.mainpage.DeleteROIWindow import *
from src.View.mainpage.DrawROIWindow.SelectROIPopUp import SelectROIPopUp
from src.View.mainpage.DrawROIWindow.UIDrawROIWindow import UIDrawROIWindow
from src.View.mainpage.ManipulateROIWindow import *


class RoiDeleteOptions(QtWidgets.QMainWindow, UIDeleteROIWindow):
    """
    Create the ROI Delete Options class based on the UI from the file in
    View/ROI Delete Option
    """
    deleting_rois_structure_tuple = QtCore.Signal(tuple)  # new PyDicom dataset

    def __init__(self, rois, dataset_rtss):
        super(RoiDeleteOptions, self).__init__()

        self.setup_ui(
            self, rois, dataset_rtss, self.deleting_rois_structure_tuple)

    def update_ui(self, rois, dataset_rtss):
        self.setup_ui(
            self, rois, dataset_rtss, self.deleting_rois_structure_tuple)


class ROIDelOption:
    """
    The class that will be called by the main page to access the ROI Options
    controller
    """

    def __init__(self, structure_modified_function):
        super(ROIDelOption, self).__init__()
        self.structure_modified_function = structure_modified_function

    def show_roi_delete_options(self):
        patient_dict_container = PatientDictContainer()
        rois = patient_dict_container.get("rois")
        dataset_rtss = patient_dict_container.get("dataset_rtss")
        if not hasattr(self, "options_window"):
            self.options_window = RoiDeleteOptions(rois, dataset_rtss)
            self.options_window.deleting_rois_structure_tuple.connect(
                self.structure_modified_function)
        else:
            self.options_window.update_ui(rois, dataset_rtss)
        self.options_window.show()


class RoiDrawOptions(QtWidgets.QMainWindow, UIDrawROIWindow):
    """
    Create the ROI Draw Options class based on the UI from the file in
    View/ROI Draw Option
    """
    signal_roi_drawn = QtCore.Signal(tuple)

    def __init__(self, rois, dataset_rtss):
        super(RoiDrawOptions, self).__init__()
        self.setup_ui(self, rois, dataset_rtss, self.signal_roi_drawn)

    def update_ui(self, rois, dataset_rtss):
        self.setup_ui(self, rois, dataset_rtss, self.signal_roi_drawn)


class ROIDrawOption:
    """
    The class that will be called by the main page to access the ROI
    Options controller
    """

    def __init__(self, structure_modified_function):
        super(ROIDrawOption, self).__init__()
        self.structure_modified_function = structure_modified_function

    def show_roi_draw_options(self):
        self.choose_roi_name_window = SelectROIPopUp()
        self.choose_roi_name_window.signal_roi_name.connect(
            self.roi_name_selected)
        self.choose_roi_name_window.show()

    def roi_name_selected(self, roi_name):
        patient_dict_container = PatientDictContainer()
        rois = patient_dict_container.get("rois")
        dataset_rtss = patient_dict_container.get("dataset_rtss")

        if not hasattr(self, "draw_window"):
            self.draw_window = RoiDrawOptions(rois, dataset_rtss)
            self.draw_window.signal_roi_drawn.connect(
                self.structure_modified_function)
        else:
            self.draw_window.update_ui(rois, dataset_rtss)

        self.draw_window.set_selected_roi_name(roi_name)
        self.draw_window.show()


class RoiManipulateOptions(QtWidgets.QMainWindow, UIManipulateROIWindow):
    """
    Create the ROI Manipulate Options class based on the UI from the file in
    View/ROI Delete Option
    """
    signal_roi_manipulated = QtCore.Signal(tuple)

    def __init__(self, rois, dataset_rtss, roi_color):
        """
        :param rois: dictionary of ROI info
        :param dataset_rtss: RTSS pydicom dataset
        :param roi_color: dictionary of QColor for each ROI
        """
        super(RoiManipulateOptions, self).__init__()
        self.roi_color = roi_color
        self.setup_ui(self, rois, dataset_rtss, roi_color,
                      self.signal_roi_manipulated)

    def update_ui(self, rois, dataset_rtss, roi_color):
        """
        :param rois: dictionary of ROI info
        :param dataset_rtss: RTSS pydicom dataset
        :param roi_color: dictionary of QColor for each ROI
        """
        self.roi_color = roi_color
        self.setup_ui(self, rois, dataset_rtss, self.roi_color,
                      self.signal_roi_manipulated)


class ROIManipulateOption:
    """
    The class that will be called by the main page to access the ROI
    Options controller
    """

    def __init__(self, structure_modified_function):
        """
        :param structure_modified_function: Function to modify list of ROIs on
        Structure Tab
        """
        super(ROIManipulateOption, self).__init__()
        self.structure_modified_function = structure_modified_function

    def show_roi_manipulate_options(self, roi_color):
        """
        Display ROI manipulate window
        """
        patient_dict_container = PatientDictContainer()
        rois = patient_dict_container.get("rois")
        dataset_rtss = patient_dict_container.get("dataset_rtss")
        if not hasattr(self, "manipulate_window"):
            self.manipulate_window = RoiManipulateOptions(rois, dataset_rtss,
                                                          roi_color)
            self.manipulate_window.signal_roi_manipulated.connect(
                self.structure_modified_function)
        else:
            self.manipulate_window.update_ui(rois, dataset_rtss, roi_color)
        self.manipulate_window.show()


class ROITransferOptionUI(QtWidgets.QMainWindow, UITransferROIWindow):
    """
    Create the ROI Manipulate Options class based on the UI from the file in
    UITransferROIWindow
    """
    signal_roi_transferred_to_fixed_container = QtCore.Signal(tuple)
    signal_roi_transferred_to_moving_container = QtCore.Signal(tuple)

    def __init__(self):
        super(ROITransferOptionUI, self).__init__()
        self.setup_ui(self, self.signal_roi_transferred_to_fixed_container,
                      self.signal_roi_transferred_to_moving_container)


class ROITransferOption:
    """
        The class that will be called by ImageFusion to access the ROI
        Transfer controller
        """
    def __init__(self, fixed_dict_structure_modified_function,
                 moving_dict_structure_modified_function):
        """

        :param fixed_dict_structure_modified_function: function to call when
        the fixed image's rtss is modified
        :param moving_dict_structure_modified_function: function to call when
        the moving image's rtss is modified

        """
        super(ROITransferOption, self).__init__()
        self.fixed_dict_structure_modified_function = \
            fixed_dict_structure_modified_function
        self.moving_dict_structure_modified_function = \
            moving_dict_structure_modified_function

    def show_roi_transfer_options(self):
        """

        function to display ROI Transfer window when
        ROI Transfer button is clicked.

        """
        self.roi_transfer_option_pop_up_window = ROITransferOptionUI()
        self.roi_transfer_option_pop_up_window. \
            signal_roi_transferred_to_moving_container\
            .connect(self.moving_dict_structure_modified_function)
        self.roi_transfer_option_pop_up_window. \
            signal_roi_transferred_to_fixed_container\
            .connect(self.fixed_dict_structure_modified_function)
        self.roi_transfer_option_pop_up_window.show()
