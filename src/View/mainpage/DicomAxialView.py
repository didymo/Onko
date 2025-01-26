from PySide6 import QtWidgets, QtCore, QtGui
from skimage import measure

from src.View.mainpage.DicomView import DicomView
from src.Model.Isodose import get_dose_grid
from src.Model.PatientDictContainer import PatientDictContainer
from src.Controller.PathHandler import data_path, resource_path


class DicomAxialView(DicomView):

    suv2roi_signal = QtCore.Signal()

    def __init__(self, roi_color=None, iso_color=None,
                 metadata_formatted=False, cut_line_color=None,
                 is_four_view=False):
        """
        Initialise the DICOM Axial View.
        :param roi_color: List of ROI colors.
        :param iso_color: List of isodose colors.
        :param metadata_formatted: Whether the metadata needs to be formatted
                                   (only metadata in the four view need to be
                                   formatted)
        :param cut_line_color: The color of the cut line.
        :param is_four_view: Whether the current object is part of the four
                             view.
        """
        self.metadata_formatted = metadata_formatted
        self.slice_view = 'axial'
        super(DicomAxialView, self).__init__(
            roi_color=roi_color, iso_color=iso_color,
            cut_line_color=cut_line_color)

        # Set parent
        self.is_four_view = is_four_view

        # Init metadata widgets
        self.metadata_layout = QtWidgets.QVBoxLayout(self.view)
        self.label_image_id = QtWidgets.QLabel()
        self.label_image_pos = QtWidgets.QLabel()
        self.label_wl = QtWidgets.QLabel()
        self.label_image_size = QtWidgets.QLabel()
        self.label_zoom = QtWidgets.QLabel()
        self.label_patient_pos = QtWidgets.QLabel()
        self.button_suv2roi = QtWidgets.QPushButton()
        self.init_metadata()

        self.update_view()

    def init_metadata(self):
        """
        Create and place metadata on the view widget.
        """
        # Position of the labels on the DICOM view.
        self.label_image_id.setAlignment(
            QtCore.Qt.AlignTop | QtCore.Qt.AlignTop)
        self.label_image_pos.setAlignment(
            QtCore.Qt.AlignTop | QtCore.Qt.AlignTop)
        self.label_wl.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignRight)
        self.label_image_size.setAlignment(
            QtCore.Qt.AlignBottom | QtCore.Qt.AlignBottom)
        self.label_zoom.setAlignment(
            QtCore.Qt.AlignBottom | QtCore.Qt.AlignBottom)
        self.label_patient_pos.setAlignment(
            QtCore.Qt.AlignRight | QtCore.Qt.AlignRight)

        # SUV2ROI button (only when PET is opened)
        patient_dict_container = PatientDictContainer()
        datasets = patient_dict_container.dataset
        pet_opened = False
        for ds in datasets:
            if datasets[ds].SOPClassUID == "1.2.840.10008.5.1.4.1.1.128":
                pet_opened = True
                break
        if pet_opened:
            icon_suv2roi = QtGui.QIcon()
            icon_suv2roi.addPixmap(
                QtGui.QPixmap(
                    resource_path("res/images/btn-icons/suv2roi.png")),
                QtGui.QIcon.Normal,
                QtGui.QIcon.On
            )
            self.button_suv2roi.setObjectName("SUV2ROI_Button")
            self.button_suv2roi.setIcon(icon_suv2roi)
            self.button_suv2roi.setToolTip("Convert SUVs to ROIs")
            self.button_suv2roi.setFixedSize(50, 50)
            self.button_suv2roi.setCursor(
                QtGui.QCursor(QtCore.Qt.PointingHandCursor))
            self.button_suv2roi.clicked.connect(self.suv2roi_handler)

        # Set all labels to white
        stylesheet = "QLabel { color : white; }"
        self.format_metadata_labels(stylesheet)

        # The following layout was originally accomplished using a QGridLayout with QSpaceItems to anchor the labels
        # to the corners of the DICOM view. This caused a reintroduction of the tedious memory issues that were fixed
        # with the restructure. The following was rewritten to not use QSpaceItems because they, for reasons unknown,
        # caused a memory leak resulting in the entire patient dictionary not being cleared from memory correctly,
        # leaving hundreds of additional megabytes unused in memory each time a new patient was opened.

        # Create a widget to contain the two top-left labels
        top_left_widget = QtWidgets.QWidget()
        top_left = QtWidgets.QVBoxLayout(top_left_widget)
        top_left.addWidget(self.label_image_id,
                           QtCore.Qt.AlignTop | QtCore.Qt.AlignTop)
        top_left.addWidget(self.label_image_pos,
                           QtCore.Qt.AlignTop | QtCore.Qt.AlignTop)

        # Create a widget to contain the top-right label
        top_right_widget = QtWidgets.QWidget()
        top_right = QtWidgets.QVBoxLayout(top_right_widget)
        top_right.addWidget(
            self.label_wl, QtCore.Qt.AlignTop | QtCore.Qt.AlignTop)

        # Create a widget to contain the two top widgets
        top_widget = QtWidgets.QWidget()
        top = QtWidgets.QHBoxLayout(top_widget)
        # Set margin for axial view
        if self.metadata_formatted:
            top_widget.setFixedHeight(50)
            top_widget.setContentsMargins(0, 0, 0, 0)
            top.setContentsMargins(0, 0, 0, 0)
            top.setSpacing(0)
        else:
            top_widget.setFixedHeight(100)
        top.addWidget(top_left_widget, QtCore.Qt.AlignLeft |
                      QtCore.Qt.AlignLeft)
        top.addWidget(top_right_widget, QtCore.Qt.AlignRight |
                      QtCore.Qt.AlignRight)

        # Create a widget to contain the two bottom-left labels
        bottom_left_widget = QtWidgets.QWidget()
        bottom_left = QtWidgets.QVBoxLayout(bottom_left_widget)
        bottom_left.addWidget(self.label_image_size,
                              QtCore.Qt.AlignBottom | QtCore.Qt.AlignBottom)
        bottom_left.addWidget(
            self.label_zoom, QtCore.Qt.AlignBottom | QtCore.Qt.AlignBottom)

        # Create a widget to contain the two bottom-right widgets
        bottom_right_widget = QtWidgets.QWidget()
        bottom_right = QtWidgets.QVBoxLayout(bottom_right_widget)
        bottom_right.setAlignment(QtCore.Qt.AlignRight)
        bottom_right.addWidget(self.label_patient_pos,
                               QtCore.Qt.AlignBottom | QtCore.Qt.AlignBottom)

        # Add the SUV2ROI button if PET opened and in single view
        if pet_opened and not self.is_four_view:
            bottom_right_button_layout = \
                QtWidgets.QHBoxLayout(bottom_right_widget)
            bottom_right_button_layout.addStretch(1)
            bottom_right_button_layout.addWidget(
                self.button_suv2roi)
            bottom_right.addLayout(bottom_right_button_layout)

        # Create a widget to contain the two bottom widgets
        bottom_widget = QtWidgets.QWidget()
        bottom = QtWidgets.QHBoxLayout(bottom_widget)
        # Set margin for axial view
        if self.metadata_formatted:
            bottom_widget.setFixedHeight(50)
            bottom_widget.setContentsMargins(0, 0, 0, 0)
            bottom.setContentsMargins(0, 0, 0, 0)
            bottom.setSpacing(0)
        else:
            bottom_widget.setFixedHeight(100)
        bottom.addWidget(bottom_left_widget,
                         QtCore.Qt.AlignLeft | QtCore.Qt.AlignLeft)
        bottom.addWidget(bottom_right_widget,
                         QtCore.Qt.AlignRight | QtCore.Qt.AlignRight)

        # Add the bottom and top widgets to the view
        self.metadata_layout.addWidget(
            top_widget, QtCore.Qt.AlignTop | QtCore.Qt.AlignTop)
        self.metadata_layout.addStretch()
        self.metadata_layout.addWidget(
            bottom_widget, QtCore.Qt.AlignBottom | QtCore.Qt.AlignBottom)

    def format_metadata_labels(self, stylesheet):
        """
        Format the meta data's labels
        """
        self.label_image_id.setStyleSheet(stylesheet)
        self.label_image_pos.setStyleSheet(stylesheet)
        self.label_wl.setStyleSheet(stylesheet)
        self.label_image_size.setStyleSheet(stylesheet)
        self.label_zoom.setStyleSheet(stylesheet)
        self.label_patient_pos.setStyleSheet(stylesheet)

    def format_metadata_margin(self):
        """
        Update the margin of the metadata depending on the size of the view and the scene.
        """
        if self.metadata_formatted:
            view_height = self.view.size().height()
            view_width = self.view.size().width()
            scene_height = self.scene.height() * self.zoom
            scene_width = self.scene.width() * self.zoom

            if view_height >= scene_height and view_width >= scene_width:
                # Remove all margin because there is no slider
                self.metadata_layout.setSpacing(0)
                self.metadata_layout.setContentsMargins(0, 0, 0, 0)
            else:
                # Add margin if the vertical and/or horizontal sliders appear
                self.metadata_layout.setSpacing(6)
                if view_height >= scene_height:
                    self.metadata_layout.setContentsMargins(0, 0, 0, 11)
                elif view_width >= scene_width:
                    self.metadata_layout.setContentsMargins(0, 0, 11, 0)
                else:
                    self.metadata_layout.setContentsMargins(0, 0, 11, 11)

    def format_metadata(self, size: QtCore.QSize):
        """
        Update the font size of the meta data's labels depending on the StackedWidget's size.
        :param size: size of the StackedWidget used in the MainPage.
        """
        if self.metadata_formatted:
            # TODO: generalise 1200 and 600
            if size.width() < 1200 and size.height() < 600:
                stylesheet = "QLabel { color : white; font-size: 10px }"
            else:
                stylesheet = "QLabel { color : white; }"
            self.format_metadata_labels(stylesheet)

    def update_view(self, zoom_change=False):
        """
            Update the view of the DICOM Image.
            :param zoom_change: Boolean indicating whether the user wants to change the zoom. False by default.
        """
        super().update_view(zoom_change)
        self.update_metadata()

    def update_metadata(self):
        """
        Update metadata displayed on the DICOM Image view.
        """
        # Retrieve dictionary from the dataset of the slice
        id = self.slider.value()
        dataset = self.patient_dict_container.dataset[id]

        # Set margin for axial view
        self.format_metadata_margin()

        # Information to display
        self.current_slice_number = dataset['InstanceNumber'].value
        total_slices = len(self.patient_dict_container.get("pixmaps_axial"))
        row_img = dataset['Rows'].value
        col_img = dataset['Columns'].value
        window = self.patient_dict_container.get("window")
        level = self.patient_dict_container.get("level")
        slice_pos = dataset['ImagePositionPatient'].value[2]

        if hasattr(dataset, 'PatientPosition'):
            patient_pos = dataset['PatientPosition'].value
            self.label_patient_pos.setText(
                "Patient Position: %s" % (str(patient_pos)))

        # Update labels
        self.label_image_id.setText(
            "Image: %s / %s" % (str(self.current_slice_number), str(total_slices)))
        self.label_image_pos.setText("Position: %s mm" % (str(slice_pos)))
        self.label_wl.setText("W/L: %s/%s" % (str(window), str(level)))
        self.label_image_size.setText(
            "Image Size: %sx%spx" % (str(row_img), str(col_img)))
        self.label_zoom.setText(
            "Zoom: " + "{:.2f}".format(self.zoom * 100) + "%")

    def roi_display(self):
        """
        Display ROI structures on the DICOM Image.
        """
        slider_id = self.slider.value()
        curr_slice = self.patient_dict_container.get("dict_uid")[slider_id]

        selected_rois = self.patient_dict_container.get("selected_rois")
        rois = self.patient_dict_container.get("rois")
        selected_rois_name = []
        for roi in selected_rois:
            selected_rois_name.append(rois[roi]['name'])

        for roi in selected_rois:
            roi_name = rois[roi]['name']
            polygons = self.patient_dict_container.get("dict_polygons_axial")[
                roi_name][curr_slice]
            super().draw_roi_polygons(roi, polygons)

    def isodose_display(self):
        """
        Display isodoses on the DICOM Image.
        """
        slider_id = self.slider.value()
        curr_slice_uid = self.patient_dict_container.get("dict_uid")[slider_id]
        z = self.patient_dict_container.dataset[slider_id].ImagePositionPatient[2]
        dataset_rtdose = self.patient_dict_container.dataset['rtdose']
        grid = get_dose_grid(dataset_rtdose, float(z))

        if not (len(grid) == 0):
            # sort selected_doses in ascending order so that the high dose isodose washes
            # paint over the lower dose isodose washes
            for sd in sorted(self.patient_dict_container.get("selected_doses")):
                dose_level = sd * self.patient_dict_container.get("rx_dose_in_cgray") / \
                    (dataset_rtdose.DoseGridScaling * 10000)
                contours = measure.find_contours(grid, dose_level)

                polygons = self.calc_dose_polygon(
                    self.patient_dict_container.get("dose_pixluts")[curr_slice_uid], contours)

                brush_color = self.iso_color[sd]
                with open(data_path('line&fill_configuration'), 'r') as stream:
                    elements = stream.readlines()
                    if len(elements) > 0:
                        iso_line = int(elements[2].replace('\n', ''))
                        iso_opacity = int(elements[3].replace('\n', ''))
                        line_width = float(elements[4].replace('\n', ''))
                    else:
                        iso_line = 2
                        iso_opacity = 5
                        line_width = 2.0
                    stream.close()
                iso_opacity = int((iso_opacity / 100) * 255)
                brush_color.setAlpha(iso_opacity)
                pen_color = QtGui.QColor(
                    brush_color.red(), brush_color.green(), brush_color.blue())
                pen = self.get_qpen(pen_color, iso_line, line_width)
                for i in range(len(polygons)):
                    self.scene.addPolygon(
                        polygons[i], pen, QtGui.QBrush(brush_color))

    def calc_dose_polygon(self, dose_pixluts, contours):
        """
        Calculate a list of polygons to display for a given isodose.
        :param dose_pixluts:
         lookup table (LUT) to get the image pixel values
        :param contours:
          trace outline of the isodose to be displayed
        :return: List of polygons of type QPolygonF.
        """
        list_polygons = []
        for contour in contours:
            list_qpoints = []
            # Slicing controls how many points considered for visualization
            # Essentially affects sharpness of edges, fewer points equals "smoother" edges
            for point in contour[::2]:
                curr_qpoint = QtCore.QPoint(
                    dose_pixluts[0][int(point[1])], dose_pixluts[1][int(point[0])])
                list_qpoints.append(curr_qpoint)
            curr_polygon = QtGui.QPolygonF(list_qpoints)
            list_polygons.append(curr_polygon)

        return list_polygons

    def suv2roi_handler(self):
        """
        Clicked action handler for the SUV2ROI button. Opens a progress
        window and initiates the SUV2ROI conversion process.
        """
        self.suv2roi_signal.emit()
