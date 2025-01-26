import os
import pytest

from src.Model import ROI
from src.Model.ISO2ROI import ISO2ROI
from src.Model.Isodose import get_dose_grid
from src.Model.PatientDictContainer import PatientDictContainer
from src.Model import ImageLoading

from pathlib import Path
from pydicom import dcmread
from pydicom.errors import InvalidDicomError
from skimage import measure

def find_DICOM_files(file_path):
    """Function to find DICOM files in a given folder.
    :param file_path: File path of folder to search.
    :return: List of file paths of DICOM files in given folder.
    """

    dicom_files = []

    # Walk through directory
    for root, dirs, files in os.walk(file_path, topdown=True):
        for name in files:
            # Attempt to open file as a DICOM file
            try:
                dcmread(os.path.join(root, name))
            except (InvalidDicomError, FileNotFoundError):
                pass
            else:
                dicom_files.append(os.path.join(root, name))
    return dicom_files


class TestIso2Roi:
    """
    Class to set up the OnkoDICOM main window for testing
    ISO2ROI functionality.
    """
    __test__ = False

    def __init__(self):
        # Load test DICOM files
        desired_path = Path.cwd().joinpath('test', 'testdata')

        # list of DICOM test files
        selected_files = find_DICOM_files(desired_path)
        # file path of DICOM files
        file_path = os.path.dirname(os.path.commonprefix(selected_files))
        read_data_dict, file_names_dict = \
            ImageLoading.get_datasets(selected_files)

        # Create patient dict container object
        self.patient_dict_container = PatientDictContainer()
        self.patient_dict_container.clear()
        self.patient_dict_container.set_initial_values\
            (file_path, read_data_dict, file_names_dict)

        # Set additional attributes in patient dict container
        # (otherwise program will crash and test will fail)
        if "rtss" in file_names_dict:
            dataset_rtss = dcmread(file_names_dict['rtss'])
            self.rois = ImageLoading.get_roi_info(dataset_rtss)
            dict_raw_contour_data, dict_numpoints = \
                ImageLoading.get_raw_contour_data(dataset_rtss)
            dict_pixluts = ImageLoading.get_pixluts(read_data_dict)

            self.patient_dict_container.set("rois", self.rois)
            self.patient_dict_container.set("raw_contour",
                                            dict_raw_contour_data)
            self.patient_dict_container.set("num_points", dict_numpoints)
            self.patient_dict_container.set("pixluts", dict_pixluts)

            # Set location of rtss file
            file_paths = self.patient_dict_container.filepaths
            self.patient_dict_container.set("file_rtss", file_paths['rtss'])

        # Create ISO2ROI object
        self.iso2roi = ISO2ROI()


@pytest.fixture(scope="module")
def test_object():
    """Function to pass a shared TestIso2Roi object to each test."""
    test = TestIso2Roi()
    return test


def test_select_prescription_dose_data(test_object):
    """
    Test selecting prescription dose data from both an RT Plan and an
    RT Dose. Assumes the test data set does not contain dose data.
    :param test_object: test_object function, for accessing the shared
                        TestIso2Roi object.
    """
    # Test selecting RT Plan dose data
    rt_plan_dose = test_object.patient_dict_container.dataset['rtdose']
    slider_id = 0

    dataset = test_object.patient_dict_container.dataset[slider_id]
    z = dataset.ImagePositionPatient[2]
    grids = get_dose_grid(rt_plan_dose, float(z))

    # Assert that the dose grid has 121 entries
    assert len(grids) == 121

    # Assert that each entry in the dose grid has 220 entries
    for grid in grids:
        assert len(grid) == 220

    # Test selecting RT Dose dose data
    rt_dose_dose = test_object.patient_dict_container.get("rx_dose_in_cgray")

    # Assert that the RT Dose dose data is None
    assert not rt_dose_dose


def test_calculate_prescription_dose_boundaries(test_object):
    """
    Test calculating prescription dose boundaries. Assumes that the test
    data set does not contain any dose data.
    :param test_object: test_object function, for accessing the shared
                        TestIso2Roi object.
    """
    boundaries = None

    # Get required dose data
    rt_plan_dose = test_object.patient_dict_container.dataset['rtdose']
    slider_id = 0
    dataset = test_object.patient_dict_container.dataset[slider_id]
    z = dataset.ImagePositionPatient[2]
    grid = get_dose_grid(rt_plan_dose, float(z))
    rt_dose_dose = 100

    isodose_percentages = \
        [10, 25, 50, 75, 80, 85, 90, 95, 100, 105]

    # Calculate prescription dose boundaries
    if not (len(grid) == 0):
        for sd in isodose_percentages:
            dose_level = sd * rt_dose_dose / \
                         (rt_plan_dose.DoseGridScaling * 10000)
            boundaries = measure.find_contours(grid, dose_level)

    # Assert that there are no boundaries
    assert len(boundaries) == 0


def test_generate_roi_from_iso(test_object):
    """
    Test for generating ROIs from ISO data.
    :param test_object: test_object function, for accessing the shared
                        TestIso2Roi object.
    """
    # Initialise variables needed for function
    slider_id = 0
    dataset = test_object.patient_dict_container.dataset[slider_id]
    pixlut = test_object.patient_dict_container.get("pixluts")
    pixlut = pixlut[dataset.SOPInstanceUID]
    z_coord = dataset.SliceLocation

    # Create fake contour data
    contours = [
        [0, 1], [1, 2], [2, 3],
        [3, 4], [4, 5]
    ]

    # Create fake pixlut data
    dose_pixluts = [[41, 43, 45, 47, 49, 51], [151, 153, 155, 159, 161, 163]]

    # Contour data to pixel points
    list_points = []
    for item in contours:
        list_points.append \
            ([dose_pixluts[0][int(item[1])],
              dose_pixluts[1][int(item[0])]])

    # Convert the pixel points to RCS points
    points = []
    for i, item in enumerate(list_points):
        points.append \
            (ROI.pixel_to_rcs(pixlut,
                              round(item[0]),
                              round(item[1])))

    # Add z coord to contour data
    contour_data = []
    for p in points:
        coords = (p[0], p[1], z_coord)
        contour_data.append(coords)

    # Transform RCS points into 1D array
    single_array = []
    for sublist in contour_data:
        for item in sublist:
            single_array.append(item)

    # Assert ROI points exist
    assert len(single_array) == 15
    assert len(single_array) == 3 * len(contour_data)


def test_find_rtss(test_object):
    """
    Test for finding existing RT Struct files. Assumes the test data
    contains an RT Struct file.
    :param test_object: test_object function, for accessing the shared
                        TestIso2Roi object.
    """
    rtss_directory = Path(test_object.patient_dict_container.get("file_rtss"))

    assert rtss_directory

def test_create_rtss(test_object):
    """
    Test for creating an RT Struct file if one does not exist.
    :param test_object: test_object function, for accessing the shared
                        TestIso2Roi object.
    """
    # Create file path
    file_path = test_object.patient_dict_container.filepaths.values()
    file_path = Path(os.path.commonpath(file_path))
    file_path = str(file_path.joinpath("rtss.dcm"))

    # Get CT UID list
    ct_uid_list = ImageLoading.get_image_uid_list(
        test_object.patient_dict_container.dataset)

    # Generate RTSS
    rtss = ROI.create_initial_rtss_from_ct(
        test_object.patient_dict_container.dataset[0], file_path, ct_uid_list)

    # Get test dataset
    test_ds = test_object.patient_dict_container.dataset[0]

    # Assert that the ds exists and is not empty
    assert rtss

    # Assert that certain values are correct
    assert rtss.PatientName == test_ds.PatientName
    assert rtss.PatientID == test_ds.PatientID
    assert rtss.PatientBirthDate == test_ds.PatientBirthDate
    assert rtss.PatientSex == test_ds.PatientSex
    assert rtss.StudyInstanceUID == test_ds.StudyInstanceUID
    assert rtss.Modality == 'RTSTRUCT'
    assert rtss.SOPClassUID == '1.2.840.10008.5.1.4.1.1.481.3'
