"""
Created on Thu Dec  5 05:25:04 2019

@author: sjswerdloff
"""
import pytest
import os
import numpy as np
from pathlib import Path
from pydicom import dataset, dcmread
from pydicom.errors import InvalidDicomError
from pydicom.tag import Tag

from src.Model import ImageLoading
from src.Model.PatientDictContainer import PatientDictContainer
from src.Model.ROI import add_to_roi, calculate_matrix, create_roi, roi_to_geometry, \
    get_roi_contour_pixel, manipulate_rois, geometry_to_roi, create_initial_rtss_from_ct


def find_DICOM_files(file_path):
    """
    Function to find DICOM files in a given folder.
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


class TestROI:
    """
    Class to set up the OnkoDICOM main window for testing
    ROI functionality
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
        self.patient_dict_container.set_initial_values \
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


@pytest.fixture(scope="module")
def test_object():
    """Function to pass a shared TestROI object to each test."""
    test = TestROI()
    return test


def test_calculate_matrix():
    image_ds = dataset.Dataset()
    image_ds.PixelSpacing = [1, 1]
    image_ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    image_ds.ImagePositionPatient = [0, 0, 0]
    image_ds.Rows = 4
    image_ds.Columns = 4
    array_x, array_y = calculate_matrix(image_ds)
    assert np.all(array_x == np.array([0, 1, 2, 3]))
    assert np.all(array_y == np.array([0, 1, 2, 3]))


def test_add_to_roi():
    rt_ss = dataset.Dataset()

    rt_ss.StructureSetROISequence = []
    rt_ss.StructureSetROISequence.append(dataset.Dataset())
    rt_ss.StructureSetROISequence[0].ReferencedFrameOfReferenceUID = "1.2.3"
    rt_ss.StructureSetROISequence[0].ROINumber = "1"
    rt_ss.StructureSetROISequence[0].ROIName = "NewTestROI"

    rt_ss.ROIContourSequence = []

    rt_ss.RTROIObservationsSequence = []

    roi_name = "NewTestROI"
    roi_coordinates = [0, 0, 0, 0, 1, 0, 1, 0, 0]  # a right triangle
    image_ds = dataset.Dataset()
    image_ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    image_ds.SOPInstanceUID = "1.2.3.4.5.6.7.8.9"
    patient_dict_container = PatientDictContainer()
    # container has to be initialised with kwargs content
    # for the get/set to not fail for lack of an "additional_parameters" dict.
    patient_dict_container.set_initial_values(None, None, None,
                                              blah="blah", rois={})
    if patient_dict_container.get("rois") is not None:
        print("rois are present in patient dict container")
    updated_rtss = create_roi(rt_ss, roi_name,
                              [{'coords': roi_coordinates, 'ds': image_ds}])
    # clearly the above is an opportunity to factor out to a fixture or similar
    rtss_with_added_roi = add_to_roi(updated_rtss, roi_name,
                                     roi_coordinates, image_ds)
    first_contour = rtss_with_added_roi.ROIContourSequence[0].\
        ContourSequence[0]
    second_contour = rtss_with_added_roi.ROIContourSequence[0].\
        ContourSequence[1]
    assert (
        second_contour
        .ContourImageSequence[0]
        .ReferencedSOPClassUID
        == image_ds.SOPClassUID
    )
    assert(first_contour.ContourGeometricType == "OPEN_PLANAR")
    assert (second_contour.ContourGeometricType == "OPEN_PLANAR")


def test_create_roi():
    rt_ss = dataset.Dataset()

    rt_ss.StructureSetROISequence = []
    rt_ss.StructureSetROISequence.append(dataset.Dataset())
    rt_ss.StructureSetROISequence[0].ReferencedFrameOfReferenceUID = "1.2.3"
    rt_ss.StructureSetROISequence[0].ROINumber = "1"

    rt_ss.ROIContourSequence = []

    rt_ss.RTROIObservationsSequence = []

    roi_name = "NewTestROI"
    # a closed right triangle
    roi_coordinates = [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0]
    image_ds = dataset.Dataset()
    image_ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    image_ds.SOPInstanceUID = "1.2.3.4.5.6.7.8.9"
    patient_dict_container = PatientDictContainer()
    # container has to be initialised with kwargs content
    # for the get/set to not fail for lack of an "additional_parameters" dict.
    patient_dict_container.set_initial_values(None, None, None,
                                              blah="blah", rois={})
    if patient_dict_container.get("rois") is not None:
        print("rois are present in patient dict container")
    updated_rtss = create_roi(rt_ss, roi_name,
                              [{'coords': roi_coordinates, 'ds': image_ds}])
    first_contour = updated_rtss.ROIContourSequence[0].ContourSequence[0]
    assert (
        first_contour
        .ContourImageSequence[0]
        .ReferencedSOPClassUID
        == image_ds.SOPClassUID
    )
    assert (first_contour.ContourGeometricType == "CLOSED_PLANAR")
    assert (rt_ss.RTROIObservationsSequence[0].RTROIInterpretedType == "ORGAN")


def test_roi_to_geometry(test_object):
    roi_names = [roi['name']
                 for roi in test_object.
                     patient_dict_container.get("rois").values()
                 if 'ISO' not in roi['name']]
    dict_rois_contours = get_roi_contour_pixel(test_object.
                                               patient_dict_container.
                                               get("raw_contour"),
                                               roi_names,
                                               test_object.
                                               patient_dict_container.
                                               get("pixluts"))
    for roi_name in roi_names:
        roi_geometry = roi_to_geometry(dict_rois_contours[roi_name])
        for slice_id, geometry in roi_geometry.items():
            if geometry.geom_type == "Polygon":
                assert len(geometry.exterior.coords) - \
                       len(dict_rois_contours[roi_name][slice_id][0]) <= 1
            else:
                for i in range(len(geometry.geoms)):
                    if geometry.geom_type == "Polygon":
                        assert len(geometry.geoms[i].exterior.coords) - \
                               len(dict_rois_contours[roi_name][slice_id][i]) \
                               <= 1


def test_roi_manipulation(test_object):
    dict_rois_contours = get_roi_contour_pixel(
        test_object.patient_dict_container.get("raw_contour"),
        ['LUNG_R', 'HEART'],
        test_object.patient_dict_container.get("pixluts"))

    lung_r_geometry = roi_to_geometry(dict_rois_contours['LUNG_R'])
    heart_geometry = roi_to_geometry(dict_rois_contours['HEART'])

    union_result = manipulate_rois(lung_r_geometry, heart_geometry, "UNION")
    result_contours = geometry_to_roi(union_result)
    assert len(result_contours.keys()) == \
           len(dict_rois_contours['LUNG_R'].keys())


def test_create_initial_rtss_from_ct(qtbot, test_object, init_config):
    # Create a test rtss
    path = test_object.patient_dict_container.path
    rtss_path = Path(path).joinpath('rtss.dcm')
    uid_list = ImageLoading.get_image_uid_list(
        test_object.patient_dict_container.dataset)
    rtss = create_initial_rtss_from_ct(
        test_object.patient_dict_container.dataset[1], rtss_path, uid_list)

    # type 1 tags - must exist and not be empty
    type_1_tags: list = [Tag("StudyInstanceUID"),
                         Tag("Modality"),
                         Tag("SeriesInstanceUID"),
                         Tag("StructureSetLabel"),
                         Tag("SOPClassUID"),
                         Tag("SOPInstanceUID")
                         ]
    # type 2 tags - must exist and be at least an empty string
    type_2_tags: list = [Tag("PatientName"),
                         Tag("PatientBirthDate"),
                         Tag("PatientSex"),
                         Tag("StudyDate"),
                         Tag("StudyTime"),
                         Tag("AccessionNumber"),
                         Tag("ReferringPhysicianName"),
                         Tag("StudyID"),
                         Tag("OperatorsName"),
                         Tag("SeriesNumber"),
                         Tag("Manufacturer"),
                         Tag("StructureSetDate"),
                         Tag("StructureSetTime")
                         ]

    # type 1 sequence tags - must exist
    type_1_sequence_tags: list = [Tag("StructureSetROISequence"),
                                  Tag("ROIContourSequence"),
                                  Tag("RTROIObservationsSequence")
                                  ]

    # Checking type 1 tags
    for tag in type_1_tags:
        assert (tag in rtss) is True
        assert rtss[tag].is_empty is False

    # Checking type 2 tags
    for tag in type_2_tags:
        assert (tag in rtss) is True
        if rtss[tag].value != "":
            assert rtss[tag].is_empty is False

    # Checking type 1 sequence tags
    for tag in type_1_sequence_tags:
        assert (tag in rtss) is True
