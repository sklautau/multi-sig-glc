'''
One solution is to treat scripts as "executables" and call them with subprocess.
Another option is to convert scripts into Python modules and instead turn
them into importable modules. Then you can orchestrate everything from a single
Python “driver” script without subprocess.
In the second caese, each script should expose a callable entry point,
typically a main() function, and guard execution with:
def main():
    # existing script logic here
    pass

if __name__ == "__main__":
    main()
'''

import shutil
import subprocess
import sys

PYTHON_EXECUTABLE = "python"


def run_script(script_path: str, *args: str):
    command = [PYTHON_EXECUTABLE, script_path] + list(args)
    print(f"#### Running command: {' '.join(command)}")
    subprocess.run(command, check=True)


def signal_processing():
    # 2) calculate new signals
    run_script(r'.\signal_processing\ecg.py')
    run_script(r'.\signal_processing\ppg.py')
    run_script(r'.\signal_processing\bioimpedance.py')

    # Organizes good segments in a Pandas dataframe CSV:
    run_script(r'.\signal_processing\save_good_segments.py')


def feature_extraction():
    # 3) Feature extraction for machine learning

    # There are different scripts, with distinct strategies. Choose one.
    method = 3

    if method == 1:
        # This is a legacy method of dealing with signals
        # Two scripts below are used when one wants a single feature vector per file
        run_script(r'.\machinelearning\features_per_file.py')
        # After running features_per_file.py, one needs to "clean" the dataframe with:
        run_script(r'.\machinelearning\clean_features_dataframe.py')
    elif method == 2:
        # For each group, find the best segment and create a single feature vector with best segment
        run_script(r'.\machinelearning\features_best_segments.py')
    elif method == 3:
        # In the code below, it is optional to work with variable-duration
        # segments or fixed-duration windows. After this decision,
        # the code will calculate all features per modality, then break
        # them later concatenate them, even in case they have a different
        # number of vectors, aiming at creating a large number of rows in
        # the output features dataframe
        run_script(r'.\machinelearning\features_per_modality.py')
        run_script(r'.\machinelearning\concatenate_multimodality_features.py')
    else:
        raise Exception("Invalid method = " + str(method))


def machine_learning_feature_selection_and_extraction():
    run_script(r'.\machinelearning\feature_selection_for_regression.py', '0', '20')
    run_script(r'.\machinelearning\dimensionality_reduction.py')


def machine_learning_regression():
    run_script(r'.\machinelearning\train_test_splits.py')
    run_script(r'.\machinelearning\regression_evaluation.py')


def main():
    # The scripts assume the configuration file is named dataset_folders.json
    # Hence, we copy the configuration file for IEB-3 dataset to the expected name:
    shutil.copyfile(r'./multimodal_dataset_folders_ieb3.json',
                    r'./dataset_folders.json')

    # run the pipeline:
    signal_processing()
    feature_extraction()
    machine_learning_feature_selection_and_extraction()
    machine_learning_regression()


if __name__ == "__main__":
    main()
