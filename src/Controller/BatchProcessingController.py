import datetime
import threading
from PySide6.QtCore import QThreadPool
from src.Model import DICOMDirectorySearch
from src.Model.batchprocessing.BatchProcessClinicalDataSR2CSV import \
    BatchProcessClinicalDataSR2CSV
from src.Model.batchprocessing.BatchProcessCSV2ClinicalDataSR import \
    BatchProcessCSV2ClinicalDataSR
from src.Model.batchprocessing.BatchProcessDVH2CSV import BatchProcessDVH2CSV
from src.Model.batchprocessing.BatchProcessISO2ROI import BatchProcessISO2ROI
from src.Model.batchprocessing.BatchProcessPyRad2CSV import \
    BatchProcessPyRad2CSV
from src.Model.batchprocessing.BatchProcessPyrad2PyradSR import \
    BatchProcessPyRad2PyRadSR
from src.Model.batchprocessing.BatchProcessROIName2FMAID import \
    BatchProcessROIName2FMAID
from src.Model.batchprocessing.BatchProcessROINameCleaning import \
    BatchProcessROINameCleaning
from src.Model.batchprocessing.BatchProcessSUV2ROI import BatchProcessSUV2ROI
from src.Model.DICOMStructure import Image, Series
from src.Model.PatientDictContainer import PatientDictContainer
from src.Model.Worker import Worker
from src.View.batchprocessing.BatchSummaryWindow import BatchSummaryWindow
from src.View.ProgressWindow import ProgressWindow


class BatchProcessingController:
    """
    This class is the controller for batch processing. It starts and
    ends processes, and controls the progress window.
    """
    def __init__(self):
        """
        Class initialiser function.
        """
        self.batch_path = ""
        self.dvh_output_path = ""
        self.pyrad_output_path = ""
        self.clinical_data_input_path = ""
        self.clinical_data_output_path = ""
        self.processes = []
        self.dicom_structure = None
        self.suv2roi_weights = None
        self.name_cleaning_options = None
        self.patient_files_loaded = False
        self.progress_window = ProgressWindow(None)
        self.timestamp = ""
        self.batch_summary = [{}, ""]

        # Threadpool for file loading
        self.threadpool = QThreadPool()
        self.interrupt_flag = threading.Event()

    def set_file_paths(self, file_paths):
        """
        Sets all the required paths
        :param file_paths: dict of directories
        """
        self.batch_path = file_paths.get('batch_path')
        self.dvh_output_path = file_paths.get('dvh_output_path')
        self.pyrad_output_path = file_paths.get('pyrad_output_path')
        self.clinical_data_input_path = \
            file_paths.get('clinical_data_input_path')
        self.clinical_data_output_path = \
            file_paths.get('clinical_data_output_path')

    def set_processes(self, processes):
        """
        Sets the selected processes
        :param processes: list of selected processes
        """
        self.processes = processes

    def set_suv2roi_weights(self, suv2roi_weights):
        """
        Function used to set suv2roi_weights.
        :param suv2roi_weights: Dictionary of patient IDs and patient weight
                                in grams.
        """
        self.suv2roi_weights = suv2roi_weights

    def start_processing(self):
        """
        Starts the batch process.
        """
        # Create new instance of ProgressWindow
        self.progress_window = ProgressWindow(None)

        # Connect callbacks
        self.progress_window.signal_error.connect(
            self.error_processing)
        self.progress_window.signal_loaded.connect(
            self.completed_processing)

        # Start performing processes on patient files
        self.progress_window.start(self.perform_processes)

    def load_patient_files(self, path, progress_callback,
                           search_complete_callback):
        """
        Load the patient files from directory.
        """
        # Set the interrup flag
        self.interrupt_flag.set()

        # Release the current thread, and create new threadpool
        self.threadpool.releaseThread()
        self.threadpool = QThreadPool()

        # Clear the interrupt flag
        self.interrupt_flag.clear()

        # Create new worker
        worker = Worker(DICOMDirectorySearch.get_dicom_structure,
                        path,
                        self.interrupt_flag,
                        progress_callback=True)

        # Connect callbacks
        worker.signals.result.connect(search_complete_callback)
        worker.signals.progress.connect(progress_callback)

        # Start the worker
        self.threadpool.start(worker)

    def set_dicom_structure(self, dicom_structure):
        """
        Function used to set dicom_structure
        :param dicom_structure: DICOMStructure
        """
        self.dicom_structure = dicom_structure

    def set_name_cleaning_options(self, options):
        """
        Set name cleaning options for batch ROI name cleaning.
        :param options: Dictionary of datasets, ROIs, and options for
                        cleaning the ROIs.
        """
        self.name_cleaning_options = options

    @staticmethod
    def get_patient_files(patient):
        """
        Get patient files.
        :param patient: patient data.
        :return: cur_patient_files, dictionary of classes and series'.
        """
        # Get files in patient
        cur_patient_files = {}
        for study in patient.studies.values():
            for series_type in study.series.values():
                for series in series_type.values():

                    image = list(series.images.values())[0]
                    class_id = image.class_id

                    if class_id not in cur_patient_files:
                        cur_patient_files[class_id] = []

                    cur_patient_files[class_id].append(series)

        return cur_patient_files

    def perform_processes(self, interrupt_flag, progress_callback=None):
        """
        Performs each selected process to each selected patient.
        :param interrupt_flag: A threading.Event() object that tells the
                               function to stop loading.
        :param progress_callback: A signal that receives the current
                                  progress of the loading.
        """
        # Clear batch summary
        self.batch_summary = [{}, ""]

        # Dictionary of process names and functions
        self.process_functions = {
            "iso2roi": self.batch_iso2roi_handler,
            "suv2roi": self.batch_suv2roi_handler,
            "dvh2csv": self.batch_dvh2csv_handler,
            "pyrad2csv": self.batch_pyrad2csv_handler,
            "pyrad2pyrad-sr": self.batch_pyrad2pyradsr_handler,
            "csv2clinicaldata-sr": self.batch_csv2clinicaldatasr_handler,
            "clinicaldata-sr2csv": self.batch_clinicaldatasr2csv_handler,
            "roiname2fmaid": self.batch_roiname2fmaid_handler,
        }

        patient_count = len(self.dicom_structure.patients)
        cur_patient_num = 0
        self.timestamp = self.create_timestamp()

        # Loop through each patient
        for patient in self.dicom_structure.patients.values():
            # Stop loading
            if interrupt_flag.is_set():
                # TODO: convert print to logging
                print("Stopped Batch Processing")
                PatientDictContainer().clear()
                return False

            cur_patient_num += 1

            progress_callback.emit(("Loading patient ({}/{}) .. ".format(
                                     cur_patient_num, patient_count), 20))

            # Perform processes on patient
            for process in self.processes:
                if process == 'roinamecleaning':
                    continue
                self.process_functions[process](interrupt_flag,
                                                progress_callback,
                                                patient)

        # Perform batch ROI Name Cleaning on all patients
        if 'roinamecleaning' in self.processes:
            if self.name_cleaning_options:
                # Start the process
                process = \
                    BatchProcessROINameCleaning(progress_callback,
                                                interrupt_flag,
                                                self.name_cleaning_options)
                process.start()

                # Append process summary
                self.batch_summary[1] = process.summary
                progress_callback.emit(("Completed ROI Name Cleaning", 100))

        PatientDictContainer().clear()

    def update_rtss(self, patient):
        """
        Updates the patient dict container with the newly created RTSS (if a
        process generates one), so it can be used by future processes.
        :param patient: The patient with the newly-created RTSS.
        """
        # Get new RTSS
        rtss = PatientDictContainer().dataset['rtss']

        # Create a series and image from the RTSS
        rtss_series = Series(rtss.SeriesInstanceUID)
        rtss_series.series_description = rtss.get(
            "SeriesDescription")
        rtss_image = Image(
            PatientDictContainer().filepaths['rtss'],
            rtss.SOPInstanceUID,
            rtss.SOPClassUID,
            rtss.Modality)
        rtss_series.add_image(rtss_image)

        # Add the new study to the patient
        patient.studies[rtss.StudyInstanceUID].add_series(
            rtss_series)

        # Update the patient dict container
        PatientDictContainer().set("rtss_modified", False)

    def batch_iso2roi_handler(self, interrupt_flag,
                              progress_callback, patient):
        """
        Handles creating, starting, and processing the results of batch
        ISO2ROI.
        :param interrupt_flag: A threading.Event() object that tells the
                               function to stop loading.
        :param progress_callback: A signal that receives the current
                                  progress of the loading.
        :param patient: The patient to perform this process on.
        """
        # Get current patient files
        cur_patient_files = \
            BatchProcessingController.get_patient_files(patient)

        # Create and start process
        process = BatchProcessISO2ROI(progress_callback,
                                      interrupt_flag,
                                      cur_patient_files)
        success = process.start()

        # Add rtss to patient in case it is needed in future
        # processes
        if success:
            if PatientDictContainer().get("rtss_modified"):
                self.update_rtss(patient)
            reason = "SUCCESS"
        else:
            reason = process.summary

        # Append process summary
        if patient not in self.batch_summary[0].keys():
            self.batch_summary[0][patient] = {}
        self.batch_summary[0][patient]["iso2roi"] = reason
        progress_callback.emit(("Completed ISO2ROI", 100))

    def batch_suv2roi_handler(self, interrupt_flag,
                              progress_callback, patient):
        """
        Handles creating, starting, and processing the results of batch
        SUV2ROI.
        :param interrupt_flag: A threading.Event() object that tells the
                               function to stop loading.
        :param progress_callback: A signal that receives the current
                                  progress of the loading.
        :param patient: The patient to perform this process on.
        """
        # Get patient files
        cur_patient_files = \
            BatchProcessingController.get_patient_files(patient)

        # Get patient weight
        if patient.patient_id in self.suv2roi_weights.keys():
            if self.suv2roi_weights[patient.patient_id] is None:
                patient_weight = None
            else:
                patient_weight = \
                    self.suv2roi_weights[patient.patient_id] * 1000
        else:
            patient_weight = None

        process = BatchProcessSUV2ROI(progress_callback,
                                      interrupt_flag,
                                      cur_patient_files,
                                      patient_weight)
        success = process.start()

        # Add rtss to patient in case it is needed in future
        # processes
        if success:
            if PatientDictContainer().get("rtss_modified"):
                self.update_rtss(patient)
            reason = "SUCCESS"
        else:
            reason = process.summary

        # Append process summary
        if patient not in self.batch_summary[0].keys():
            self.batch_summary[0][patient] = {}
        self.batch_summary[0][patient]["suv2roi"] = reason
        progress_callback.emit(("Completed SUV2ROI", 100))

    def batch_dvh2csv_handler(self, interrupt_flag,
                              progress_callback, patient):
        """
        Handles creating, starting, and processing the results of batch
        DVH2CSV.
        :param interrupt_flag: A threading.Event() object that tells the
                               function to stop loading.
        :param progress_callback: A signal that receives the current
                                  progress of the loading.
        :param patient: The patient to perform this process on.
        """
        # Get current patient files
        cur_patient_files = \
            BatchProcessingController.get_patient_files(patient)

        # Create and start process
        process = BatchProcessDVH2CSV(progress_callback,
                                      interrupt_flag,
                                      cur_patient_files,
                                      self.dvh_output_path)
        process.set_filename('DVHs_' + self.timestamp + '.csv')
        success = process.start()

        # Set process summary
        if success:
            reason = "SUCCESS"
        else:
            reason = process.summary

        # Append process summary
        if patient not in self.batch_summary[0].keys():
            self.batch_summary[0][patient] = {}
        self.batch_summary[0][patient]['dvh2csv'] = reason
        progress_callback.emit(("Completed DVH2CSV", 100))

    def batch_pyrad2csv_handler(self, interrupt_flag,
                                progress_callback, patient):
        """
        Handles creating, starting, and processing the results of batch
        Pyrad2CSV.
        :param interrupt_flag: A threading.Event() object that tells the
                               function to stop loading.
        :param progress_callback: A signal that receives the current
                                  progress of the loading.
        :param patient: The patient to perform this process on.
        """
        # Get current files
        cur_patient_files = \
            BatchProcessingController.get_patient_files(patient)
        process = BatchProcessPyRad2CSV(progress_callback,
                                        interrupt_flag,
                                        cur_patient_files,
                                        self.pyrad_output_path)
        process.set_filename('PyRadiomics_' + self.timestamp + '.csv')
        success = process.start()

        # Set summary message
        if success:
            reason = "SUCCESS"
        else:
            reason = process.summary

        # Append process summary
        if patient not in self.batch_summary[0].keys():
            self.batch_summary[0][patient] = {}
        self.batch_summary[0][patient]['pyrad2csv'] = reason
        progress_callback.emit(("Completed PyRad2CSV", 100))

    def batch_pyrad2pyradsr_handler(self, interrupt_flag,
                                    progress_callback, patient):
        """
        Handles creating, starting, and processing the results of batch
        PyRad2PyRad-SR.
        :param interrupt_flag: A threading.Event() object that tells the
                               function to stop loading.
        :param progress_callback: A signal that receives the current
                                  progress of the loading.
        :param patient: The patient to perform this process on.
        """
        # Get current files
        cur_patient_files = \
            BatchProcessingController.get_patient_files(patient)
        process = BatchProcessPyRad2PyRadSR(progress_callback,
                                            interrupt_flag,
                                            cur_patient_files)
        success = process.start()

        # Set summary message
        if success:
            reason = "SUCCESS"
        else:
            reason = process.summary

        # Append process summary
        if patient not in self.batch_summary[0].keys():
            self.batch_summary[0][patient] = {}
        self.batch_summary[0][patient]['pyrad2pyradSR'] = reason
        progress_callback.emit(("Completed PyRad2PyRad-SR", 100))

    def batch_csv2clinicaldatasr_handler(self, interrupt_flag,
                                         progress_callback, patient):
        """
        Handles creating, starting, and processing the results of batch
        CSV2ClinicalData-SR.
        :param interrupt_flag: A threading.Event() object that tells the
                               function to stop loading.
        :param progress_callback: A signal that receives the current
                                  progress of the loading.
        :param patient: The patient to perform this process on.
        """
        # Get current files
        cur_patient_files = \
            BatchProcessingController.get_patient_files(patient)
        process = \
            BatchProcessCSV2ClinicalDataSR(progress_callback, interrupt_flag,
                                           cur_patient_files,
                                           self.clinical_data_input_path)
        success = process.start()

        # Update summary
        if success:
            reason = "SUCCESS"
        else:
            reason = process.summary

        if patient not in self.batch_summary[0].keys():
            self.batch_summary[0][patient] = {}
        self.batch_summary[0][patient]["csv2clinicaldatasr"] = reason
        progress_callback.emit(("Completed CSV2ClinicalData-SR", 100))

    def batch_clinicaldatasr2csv_handler(self, interrupt_flag,
                                         progress_callback, patient):
        """
        Handles creating, starting, and processing the results of batch
        ClinicalData-SR2CSV.
        :param interrupt_flag: A threading.Event() object that tells the
                               function to stop loading.
        :param progress_callback: A signal that receives the current
                                  progress of the loading.
        :param patient: The patient to perform this process on.
        """
        cur_patient_files = \
            BatchProcessingController.get_patient_files(patient)
        process = \
            BatchProcessClinicalDataSR2CSV(progress_callback, interrupt_flag,
                                           cur_patient_files,
                                           self.clinical_data_output_path)
        success = process.start()

        # Update summary
        if success:
            reason = "SUCCESS"
        else:
            reason = process.summary

        if patient not in self.batch_summary[0].keys():
            self.batch_summary[0][patient] = {}
        self.batch_summary[0][patient]["clinicaldatasr2csv"] = reason
        progress_callback.emit(("Completed ClinicalData-SR2CSV", 100))

    def batch_roiname2fmaid_handler(self, interrupt_flag,
                                    progress_callback, patient):
        """
        Handles creating, starting, and processing the results of batch
        ROIName2FMA-ID.
        :param interrupt_flag: A threading.Event() object that tells the
                               function to stop loading.
        :param progress_callback: A signal that receives the current
                                  progress of the loading.
        :param patient: The patient to perform this process on.
        """
        # Get patient files and start process
        cur_patient_files = \
            BatchProcessingController.get_patient_files(patient)
        process = \
            BatchProcessROIName2FMAID(progress_callback,
                                      interrupt_flag,
                                      cur_patient_files)
        process.start()

        # Append process summary
        reason = process.summary
        if patient not in self.batch_summary[0].keys():
            self.batch_summary[0][patient] = {}
        self.batch_summary[0][patient]["roiname2fmaid"] = reason
        progress_callback.emit(("Completed ROI Name to FMA ID", 100))

    def completed_processing(self):
        """
        Runs when batch processing has been completed.
        """
        self.progress_window.update_progress(("Processing complete!", 100))
        self.progress_window.close()

        # Create window to store summary info
        batch_summary_window = BatchSummaryWindow()
        batch_summary_window.set_summary_text(self.batch_summary)
        batch_summary_window.exec_()

    def error_processing(self):
        """
        Runs when there is an error during batch processing.
        """
        print("Error performing batch processing.")
        self.progress_window.close()
        return

    @classmethod
    def create_timestamp(cls):
        """
        Create a unique timestamp as a string.
        returns string
        """
        cur_time = datetime.datetime.now()
        year = cur_time.year
        month = cur_time.month
        day = cur_time.day
        hour = cur_time.hour
        min = cur_time.minute
        sec = cur_time.second

        time_stamp = str(year) + str(month) + str(day) + str(hour) + \
                     str(min) + str(sec)

        return time_stamp
