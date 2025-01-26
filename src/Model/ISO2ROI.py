from skimage import measure
from src.Controller.PathHandler import data_path
from src.Model import ImageLoading
from src.Model import ROI
from src.Model.Isodose import get_dose_grid
from src.Model.PatientDictContainer import PatientDictContainer


class ISO2ROI:
    """This class is for converting isodose levels to ROIs."""

    def start_conversion(self, interrupt_flag, progress_callback):
        """
        Goes the the steps of the iso2roi conversion.
        :param interrupt_flag: interrupt flag to stop process
        :param progress_callback: signal that receives the current
                                  progress of the loading.
        """
        progress_callback.emit(("Validating Datasets", 0))

        # Stop loading
        if interrupt_flag.is_set():
            # TODO: convert print to logging
            print("Stopped ISO2ROI")
            return False

        # Get isodose levels to turn into ROIs
        isodose_levels = self.get_iso_levels(data_path('isodoseRoi.csv'))

        # Stop loading
        if interrupt_flag.is_set():
            # TODO: convert print to logging
            print("Stopped ISO2ROI")
            return False

        # Calculate dose boundaries
        progress_callback.emit(("Calculating Boundaries", 50))
        boundaries = self.calculate_isodose_boundaries(isodose_levels)

        # Return if boundaries could not be calculated
        if not boundaries:
            # TODO: convert print to logging
            print("Boundaries could not be calculated.")
            return

        # Stop loading
        if interrupt_flag.is_set():
            # TODO: convert print to logging
            print("Stopped ISO2ROI")
            return False

        progress_callback.emit(("Generating ROIs", 75))
        self.generate_roi(boundaries, progress_callback)
        progress_callback.emit(("Reloading Window. Please Wait...", 95))

    def get_iso_levels(self, path):
        """
        Opens /data/csv/isodoseRoi.csv to find the isodose levels that
        the user wants turned into ROIs. Creates a
        :return: isodose_levels, a dictionary  where the key is the
                 isoname, and the value is a list containing a boolean
                 (cGy: 0, %: 1) and an integer (cGy/% value).
        """
        isodose_levels = {}

        # Open isodoseRoi.csv
        with open(path, "r") as fileInput:
            for row in fileInput:
                items = row.split(',')
                isodose_levels[items[2]] = [items[1] == 'cGy',
                                            int(items[0])]
        return isodose_levels

    def calculate_isodose_boundaries(self, isodose_levels):
        """
        Calculates isodose boundaries for each isodose level.
        :return: coutours, a list containing the countours for each
                 isodose level.
        """
        # Initialise variables needed to find isodose levels
        patient_dict_container = PatientDictContainer()
        pixmaps = patient_dict_container.get("pixmaps_axial")
        slider_min = 0
        slider_max = len(pixmaps)

        rt_plan_dose = patient_dict_container.dataset['rtdose']
        rt_dose_dose = patient_dict_container.get("rx_dose_in_cgray")

        # If rt_dose_dose does not exist, return None
        if not rt_dose_dose:
            return None

        contours = {}

        for item in isodose_levels:
            # Calculate boundaries for each isodose level for each slice
            contours[item] = []
            for slider_id in range(slider_min, slider_max):
                contours[item].append([])
                temp_ds = patient_dict_container.dataset[slider_id]
                z = temp_ds.ImagePositionPatient[2]
                grid = get_dose_grid(rt_plan_dose, float(z))

                if not (len(grid) == 0):
                    if isodose_levels[item][0]:
                        dose_level = isodose_levels[item][1] / \
                                     (rt_plan_dose.DoseGridScaling * 100)
                        contours[item][slider_id] =\
                            (measure.find_contours(grid, dose_level))
                    else:
                        dose_level = isodose_levels[item][1] * \
                                     rt_dose_dose / \
                                     (rt_plan_dose.DoseGridScaling * 10000)
                        contours[item][slider_id] = \
                            (measure.find_contours(grid, dose_level))

        # Return list of contours for each isodose level for each slice
        return contours

    def generate_roi(self, contours, progress_callback):
        """
        Generates new ROIs based on contour data.
        :param contours: dictionary of contours to turn into ROIs.
        :param progress_callback: signal to update loading progress
        """
        # Initialise variables needed for function
        patient_dict_container = PatientDictContainer()
        dataset_rtss = patient_dict_container.get("dataset_rtss")
        pixmaps = patient_dict_container.get("pixmaps_axial")
        slider_min = 0
        slider_max = len(pixmaps) - 1

        # Get existing ROIs
        existing_rois = []
        rois = patient_dict_container.get("dataset_rtss")
        if rois:
            for roi in rois.StructureSetROISequence:
                existing_rois.append(roi.ROIName)

        # Loop through each isodose level
        for item in contours:
            # Delete ROI if it already exists to recreate it
            if item in existing_rois:
                dataset_rtss = ROI.delete_roi(dataset_rtss, item)

                # Update patient dict container
                current_rois = patient_dict_container.get("rois")
                keys = []
                for key, value in current_rois.items():
                    if value["name"] == item:
                        keys.append(key)
                for key in keys:
                    del current_rois[key]
                patient_dict_container.set("rois", current_rois)

            # Calculate isodose ROI for each slice, skip if slice has no
            # contour data
            for i in range(slider_min, slider_max):
                if not len(contours[item][i]):
                    continue

                # Get required data for calculating ROI
                dataset = patient_dict_container.dataset[i]
                pixlut = patient_dict_container.get("pixluts")
                pixlut = pixlut[dataset.SOPInstanceUID]
                z_coord = dataset.SliceLocation
                curr_slice_uid = patient_dict_container.get("dict_uid")[i]
                dose_pixluts = patient_dict_container.get("dose_pixluts")
                dose_pixluts = dose_pixluts[curr_slice_uid]

                # Loop through each contour for each slice.
                # Convert the pixel points to RCS points, append z value
                single_array = []

                # Loop through each contour
                for j in range(len(contours[item][i])):
                    single_array.append([])
                    # Loop through every second point in the contour
                    for point in contours[item][i][j][::2]:
                        # Transform into dose pixel
                        dose_pixels = [dose_pixluts[0][int(point[1])],
                                       dose_pixluts[1][int(point[0])]]
                        # Transform into RCS pixel
                        rcs_pixels = ROI.pixel_to_rcs(pixlut,
                                                      round(dose_pixels[0]),
                                                      round(dose_pixels[1]))
                        # Append point coordinates to the single array
                        single_array[j].append(rcs_pixels[0])
                        single_array[j].append(rcs_pixels[1])
                        single_array[j].append(z_coord)

                # Create the ROI(s)
                for array in single_array:
                    rtss = ROI.create_roi(dataset_rtss, item,
                                          [{'coords': array, 'ds': dataset}],
                                          "DOSE_REGION")

                    # Save the updated rtss
                    patient_dict_container.set("dataset_rtss", rtss)
                    patient_dict_container.set("rois",
                                               ImageLoading.get_roi_info(rtss))

        progress_callback.emit(("Writing to RT Structure Set", 85))
