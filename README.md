# Lifting Task Risk Assessment Study using Body-worn Sensors and Embedded Machine Learning

The research focuses on developing a body-worn sensor system for human condition monitoring in smart manufacturing. It can sample acceleration and rotation data at high frequencies and transmit wirelessly over Bluetooth Low Energy. 
The system enables synchronized data collection from multiple devices and emphasizes real-time data processing using RTOS programming and multi-threading. 
A case study on lifting risk assessment demonstrates the system's capability to analyze physical stress using machine learning algorithms, highlighting the importance of sensor placement and feature extraction. 
The proposed framework allows for real-time risk assessment using lightweight algorithms and essential data features, contributing to the wider adoption of sensor systems for human monitoring in industry.

The work has been published in NAMRC52, and the citation can be found below,



## Directory Structure

- `/lib`: Contains the libraries required to perform the analysis.
  - `feature_extraction.py`: Library to extract a total of 17 features across time, frequency, and time-frequency domains.
  - `model_evaluations.py`: Library to evaluate the trained models.
  - `models.py`: Library for model creation, training, and hyperparameter optimizations.

- `data_visualization.ipynb`: Jupyter notebook to visualize the data collected from the sensors.

- `feature_importance.ipynb`: Jupyter notebook to perform the feature importance study on the extracted features across multiple sensor systems located on the human body.

- `model_development.ipynb`: Jupyter notebook to initiate model training and evaluation.

## Running the Code

To run the code in this repository, follow these steps

1. Data to placed in the directory `/data`
2. Update the training configuration inside the file `training_config.yml`

```shell
nano training_config.yml
```

3. Training the models

```shell
python training.py
```

The training process should start running at this point and the trained model along the results from the training process will be stored under the directory specified in `training_config.yml` file.


## Data

For the data used to train the models, please reach out to the authors of the paper or the owner of this repository

## TODO

- [ ] More Information on the work performed

## Acknowledgement


## Citation


