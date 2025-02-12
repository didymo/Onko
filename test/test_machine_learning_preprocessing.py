import gc
from pathlib import Path
from src.Model.batchprocessing.batchprocessingMachineLearning.Preprocessing import Preprocessing


def test_preprocessing():
    clinical_data_path = Path.cwd().joinpath('data', 'csv', 'clinical_data.csv')
    dvh_data_path = Path.cwd().joinpath('data', 'csv', 'dvh_data.csv')
    pyradiomics_data_path = Path.cwd().joinpath('data', 'csv', 'pyradiomics_data.csv')

    test_preprocessing = Preprocessing(
              path_clinical_data=clinical_data_path
            , path_pyr_data=pyradiomics_data_path
            , path_dvh_data=dvh_data_path
            , column_names=['example1']
            , type_column='category'
            , target='example_target'
            , rename_values=['Yes', 'No']
    )

    # read csv
    clinical_data, dvh, pyradiomics = test_preprocessing.read_csv()

    # check if it has all rows
    sum_of_rows = len(clinical_data)+len(dvh)+len(pyradiomics)
    # check if it has all rows rows
    assert sum_of_rows == 13

    # check if it has all columns
    assert clinical_data.shape[1] == 3
    assert dvh.shape[1] == 4
    assert pyradiomics.shape[1] == 4

    # merge datset
    full_data_set = test_preprocessing.merging_data(clinical_data, dvh, pyradiomics)

    # check if it merges correctly
    assert len(full_data_set) == 4
    assert full_data_set.shape[1] == 9

    # check rename
    clinical_data = test_preprocessing.rename(clinical_data)

    # check if only 2 'No' in dataset
    assert len(clinical_data[clinical_data['example_target'] == 'No']) == 2

    # check if dvh and pyradiomics has more than 2 cross Ids
    assert test_preprocessing.check_preprocessing_data()

    x_train, x_test, y_train, y_test = test_preprocessing.prepare_for_ml()

    # check if it is correctly split on test and train
    assert len(x_train) == 2 and len(x_test) == 2

    # Explicit cleanup of DataFrames
    del clinical_data, dvh, pyradiomics, full_data_set
    del x_train, x_test, y_train, y_test
    gc.collect()
