import copy
import os
import glob
import numpy as np
import pandas as pd
import yaml
import pickle
from lib import feature_extraction as fe
from lib import models
from sklearn.ensemble import BaggingClassifier


def bagging_classifier_constructor(loader, node):
    kwargs = {}
    for key_node, value_node in node.value:
        key = key_node.value
        value = int(value_node.value)
        kwargs[key] = value
    return BaggingClassifier(**kwargs)


yaml.FullLoader.add_constructor('tag:yaml.org,2002:python/BaggingClassifier', bagging_classifier_constructor)


def segment_data(data_array: np.array, segment_window: float, overlap: float, sampling_rate: float):
    window_size = int(segment_window * sampling_rate)
    starting_points = np.arange(0, data_array.shape[0], int(window_size * (1 - overlap))).astype("uint32")

    data_segments = list()
    for starting_index in starting_points:
        if (starting_index + window_size) < data_array.shape[0]:
            data_segments.append(
                data_array[starting_index:starting_index + window_size, ...])

    return np.array(data_segments)


if __name__ == '__main__':

    # Load the YAML parameters
    with open('training_config.yml', 'r') as f:
        tr_config = yaml.load(f, Loader=yaml.FullLoader)

    ############################
    # Load Data
    ############################
    # Get data paths
    data_path = os.path.join(os.getcwd(), "data")
    # Selecting Task
    weight_lifting = os.path.join(data_path, "LiftingAssessment")
    # Get all the ".csv" files
    all_parsed_files = glob.glob("**/*.csv", root_dir=weight_lifting, recursive=True)

    # Check to ensure data is present
    if len(all_parsed_files) == 0:
        raise Exception("Data Missing! No csv files found in {}".format(weight_lifting))

    # Load the data
    loaded_data = {}
    for file_path in all_parsed_files:
        # Full path to file
        full_path = os.path.join(weight_lifting, file_path)

        # Load the time of DAQ
        with open(full_path, "r") as file_handle:
            daq_time = file_handle.readline()
            daq_time = daq_time.split(" ")[-1]
            daq_time = int(daq_time[0:-2])
        # Read the csv
        df = pd.read_csv(full_path, header="infer", skiprows=1)

        # Store data
        loaded_data[full_path] = {
            "daq_time": daq_time,
            "df": df
        }

    # Print the counts
    print(f"Total number of files loaded - {len(loaded_data.keys())}")

    ############################
    # Parse data and Identify classes
    ############################
    # Group by features
    box_types = ["Crate", "CardboardBox"]
    weight_levels = ["W2", "W5", "W10", "W20", "W15", "W30"]
    labelled_data = {}
    for box_instance in box_types:
        for weight_instance in weight_levels:
            labelled_data[box_instance + "-" + weight_instance] = []

    for file_id in loaded_data.keys():
        box_instance = file_id.split(os.sep)[-4]
        weight_instance = file_id.split(os.sep)[-3]
        labelled_data[box_instance + "-" + weight_instance].append(file_id)

    # Print number of items within each group
    for class_instance in labelled_data.keys():
        print(f"For the class - {class_instance}, total number of items are {len(labelled_data[class_instance])}")

    ############################
    # Ensure data validity
    ############################
    sentinels_samplingRate = {"DAQSentinel01": [],
                              "DAQSentinel02": [],
                              "DAQSentinel03": []}
    sampling_rates = {}
    for file_path, data in loaded_data.items():
        # Choose the right sentinel
        sentinel = file_path.split(os.sep)[-1].split("_")[0]

        # Determine sampling rate
        total_time = data["daq_time"]
        samples = data["df"].shape[0]
        sentinels_samplingRate[sentinel].append(samples / total_time)

    for sentinel in sentinels_samplingRate.keys():
        print(
            "Sampling Rate for " + sentinel + " with mean " + str(round(np.mean(sentinels_samplingRate[sentinel]), 2)) +
            " and std of " + str(round(np.std(sentinels_samplingRate[sentinel]), 2)))

        # Get the mean sampling rate
        sampling_rates[sentinel] = round(np.mean(sentinels_samplingRate[sentinel]), 2)

    class_combined_dfs = {}
    num_individuals_analyzed = 3
    sentinels = ["DAQSentinel01", "DAQSentinel02", "DAQSentinel03"]

    # Group dataframes together
    for class_instance in labelled_data.keys():
        # Differentiate by Sentinels
        class_combined_dfs[class_instance] = {}

        # Sentinels data instance counters
        counters = {}

        # Go through each file
        for file_id in labelled_data[class_instance]:
            # Get the sentinel name
            sentinel = file_id.split(os.sep)[-1].split("_")[0]

            # Get the dataframe
            df = loaded_data[file_id]["df"].copy(deep=True)
            # Remove the starting and ending data instances
            df = df.iloc[int(4 * sampling_rates[sentinel]):int(df.shape[0] - (4 * sampling_rates[sentinel]))]

            if sentinel in list(class_combined_dfs[class_instance].keys()):
                class_combined_dfs[class_instance][sentinel] = pd.concat(
                    [class_combined_dfs[class_instance][sentinel], df], ignore_index=True, copy=True)
                counters[sentinel] += 1
            else:
                class_combined_dfs[class_instance][sentinel] = df
                counters[sentinel] = 1

        # Assert at the end of every class
        for s in counters.values():
            assert s == 10 * num_individuals_analyzed, "Each sentinel should add upto 20 counts for two individuals"

    ############################
    # Segmentation
    ############################
    data_cols_considered = tr_config["sensor_combinations"]
    # Params
    seg_window = tr_config["segmentation"]["window"]
    seg_overlap = tr_config["segmentation"]["overlap"]
    # Segment the data
    sentinel_segmented_data = {}
    for class_instance in class_combined_dfs.keys():
        sentinel_segmented_data[class_instance] = {}
        for sentinel in class_combined_dfs[class_instance].keys():
            sentinel_segmented_data[class_instance][sentinel] = segment_data(
                class_combined_dfs[class_instance][sentinel][data_cols_considered].to_numpy(), seg_window, seg_overlap,
                sampling_rates[sentinel])

    ############################
    # Feature Extraction - T + F + TF
    ############################
    features_extracted_data = {}
    for class_instance in sentinel_segmented_data.keys():
        features_extracted_data[class_instance] = {}
        for sentinel in sentinel_segmented_data[class_instance].keys():
            data = sentinel_segmented_data[class_instance][sentinel]

            # Select arguments based on sentinel
            freq_args = [{"axis": 0}, {"axis": 0},
                         {"axis": 0, "nperseg": 200, "noverlap": 100, "fs": sampling_rates[sentinel]}]
            freq_time_args = [{"wavelet": "db1"}, {"wavelet": "db1"}, {"wavelet": "db1"}]

            # Apply transformation to every data row
            for index, row in enumerate(data):
                computed_segments_sensors = []
                for i in range(data.shape[-1]):
                    # apply the transformation
                    computed_segments_sensors += fe.compute_all_features(row[:, i], freq_args=freq_args,
                                                                         freq_time_args=freq_time_args)

                data_array = np.array(computed_segments_sensors).T
                if index == 0:
                    features_extracted_data[class_instance][sentinel] = copy.deepcopy(data_array[np.newaxis, ...])
                else:
                    features_extracted_data[class_instance][sentinel] = np.append(
                        features_extracted_data[class_instance][sentinel], copy.deepcopy(data_array[np.newaxis, ...]),
                        axis=0)

    ############################
    # Model Training
    ############################
    model_params = tr_config["model_params"]
    # Choosing the sentinel and location to name conversion
    sensor_locations = tr_config["tr_sensor_locations"]
    # "DAQSentinel#" -> Is the internal names we used for the sensor systems
    location_to_name = {
        "L1": "DAQSentinel01",
        "L2": "DAQSentinel02",
        "L3": "DAQSentinel03"
    }
    sentinels = [location_to_name[x] for x in sensor_locations]

    # Training labels
    labels = tr_config["data_labels"]

    # Select the data based on the sentinels (or sensor locations)
    # Construct training data and labels
    for index, class_instance in enumerate(features_extracted_data.keys()):

        # Find the sentinel with min samples
        samples = []
        for sentinel in sentinels:
            samples.append(features_extracted_data[class_instance][sentinel].shape[0])

        min_samples = min(samples)

        for index2, sentinel in enumerate(sentinels):
            if index2 == 0:
                sub_X_train = features_extracted_data[class_instance][sentinel][0:min_samples, ...]
            else:
                sub_X_train = np.concatenate(
                    (sub_X_train, features_extracted_data[class_instance][sentinel][0:min_samples, ...]), axis=-1)

        if index == 0:
            X_train = copy.deepcopy(sub_X_train)
            y_train = np.array([labels[class_instance]] * sub_X_train.shape[0])[:, np.newaxis]
        else:
            X_train = np.append(X_train, copy.deepcopy(sub_X_train), axis=0)
            y_train = np.append(y_train, np.array([labels[class_instance]] * sub_X_train.shape[0])[:, np.newaxis],
                                axis=0)

    # Print results
    print(f"Shape of X-train is {X_train.shape}")
    y_train = y_train.squeeze(axis=-1)
    print(f"Shape of y-train is {y_train.shape}")

    # Create and Train Models
    # Create models repo
    models_repo = models.Models()
    # Initialize
    models_repo.create_models(model_params)

    # 5-fold CV Training
    cv_results_summary = models_repo.train_models_cvfolds(X_train, y_train, kfolds=5, summarize_results=True,
                                                          standardize=True)

    # Make a copy
    temp = copy.deepcopy(cv_results_summary)
    for index, model_name in enumerate(model_params.keys()):

        temp[model_name].columns = pd.MultiIndex.from_product([[model_name], temp[model_name].columns])
        # Append columns
        if index == 0:
            combined_cv_results = temp[model_name]
        else:
            combined_cv_results = pd.concat([combined_cv_results, temp[model_name]], axis=1)
    # Print the training results
    print(combined_cv_results)

    ############################
    # Save trained models
    ############################
    trained_model_save_location = os.path.join(os.getcwd(), tr_config["model_save_directory"])
    if not os.path.exists(trained_model_save_location):
        os.makedirs(trained_model_save_location)

    with open(os.path.join(trained_model_save_location, "cv_results.pkl"), "wb") as f:
        pickle.dump(combined_cv_results, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(trained_model_save_location, "trained_models.pkl"), "wb") as f:
        pickle.dump(models_repo.model_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

