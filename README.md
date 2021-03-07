# Preprocessing Correction

The project directory is under `MainProject`.

## Installation

placeholder

## Running the script

In the project directory, we can find a `.json` file named `config.json`, which is loaded first while running the script.

The main parameters are the ones under `run-settings`, there we can enable/disable each feature of our preprocessing.

When a feature is enabled, by using the parameter `train-###`, we can control whether we want to train the model or load a pre-trained one.

When a feature is enabled, the model parameters can be found under the corresponding `json` paramaters. eg. rotation model parameters can be found under `rotation`.\
The parameters are self-explanatory, or can be found in the PDF.\
The parameter `pre-trained-path` determines the path to the model weights from previous training, and is taken if `train-###` is set to `false`.

At the end of the run, we evaluate the model using the dataset found under `["run-setting"]["eval-dataset-path"]`\
The model run the images through all the enabled features, pre-trained or after training.


## Contributing
placeholder

## License
placeholder
