from src.View.mainpage.DicomView import DicomView


class DicomSagittalView(DicomView):
    def __init__(self, roi_color=None, iso_color=None, cut_line_color=None):
        self.slice_view = 'sagittal'
        super(DicomSagittalView, self).__init__(roi_color,
                                                iso_color,
                                                cut_line_color)
        self.update_view()

    def roi_display(self):
        slider_id = self.slider.value()
        selected_rois = self.patient_dict_container.get("selected_rois")
        rois = self.patient_dict_container.get("rois")
        selected_rois_name = []
        for roi in selected_rois:
            selected_rois_name.append(rois[roi]['name'])

        for roi in selected_rois:
            roi_name = rois[roi]['name']
            polygons = self.patient_dict_container.get(
                "dict_polygons_sagittal")[roi_name][slider_id]
            super().draw_roi_polygons(roi, polygons)

    def isodose_display(self):
        # TODO: Display ISODose on Sagittal View
        pass
