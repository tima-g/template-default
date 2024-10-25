# Sample README file for the experiment repository

## About this example

This example can help learn how mldev works.

The example solves a simple classification problem on the [iris](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html) dataset.
The solution uses the [scikit-learn](https://scikit-learn.org) library and the [pandas](https://pandas.pydata.org) library.

The classification problem is stated as follows.
A dataset with numeric attributes $`X_ {ij}`$ and class numbers $`y_i`$ is given.
It is necessary to teach the model $`f_t (X)`$ to predict the class number by the values ​​of the features.

```math
L(f_t(X), y) \to \min_t
``` 

In this example, this classification problem is solved using the online gradient descent algorithm [SGDClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html).
We use the so-called [hinge loss](https://en.wikipedia.org/wiki/Hinge_loss)
error function $`L = max (0, 1 - y ^ T f_t (X))`$.

In the process of training the model, the algorithm updates the parameters of the $`t`$ model in the direction,
inverse to the error gradient $`\partial L / \partial t`$,
so that after several updates the value of the error function is decreasing.
The process stops after performing a certain number of updates or when the error value stops changing too much.
 
## Description of the experiment

In the experiment, at the `prepare` stage, we receive data from the `sklearn` library,
split randomly into `test`, `dev` and `train` parts, and save them under the versioned
control in the `./data` folder by calling the `./src/prepare.py` script.

Next, at the `train` stage, we train our model on the `train` piece of data and save the model in the `./Models` folder.

Having trained the model, we can get predictions on the test data by calling the predict.py script.

During the experiment, training logs are collected in the folder `./logs/`, from which you can understand
how the quality of the model changes to the `dev` part of the data.
The training script also writes the current values ​​of the error function to the tensorboard for dynamic observation of the experiment.

See also [experiment-observing](#experiment-observing)

## Setting up and repeating the experiment

### Source

The data preparation code is located in src / prepare.py.
The script loads the dataset `iris` and splits into `train`, `dev`, `test`.

The model training code is located in `src/train.py`.

The code for getting predictions and estimating the model is in `src/predict.py`.

### Repeat experiment

The order of the experiment is recorded in the `experiment.yml` file processed by mldev at startup.
The experiment file contains the stages of its implementation and the parameters with which the experiment code is called:

To repeat the experiment, you need to [install](https://gitlab.com/mlrep/mldev/-/blob/develop/README.md) ``mldev``,
go to the folder with the experiment and execute

```
# Prepare and set up an experiment
$ mldev init -r.

# Performing an experiment
$ mldev run -f experiment.yml pipeline
```

The `experiment.yml` file contains a description of the experiment and its settings.

1. Data preparation stage


```yaml
prepare: &prepare_stage !Stage
  name: prepare
  params:
    size: 1
    needs_dvc: true
  inputs:
    - !path { path: "./src" }
  outputs:
    - !path { path: "./data" }
  script:
    - "python3 src/prepare.py"

```

2. Model training stage

```yaml
train: &train_stage !Stage
  name: train
  params:
    needs_dvc: true
    num_iters: 10
  inputs:
    - !path
      path: "./data"
      files:
        - "X_train.pickle"
        - "X_dev.pickle"
        - "X_test.pickle"
        - "y_train.pickle"
        - "y_dev.pickle"
        - "y_test.pickle"
  outputs: &model_data
    - !path
      path: "models/default"
      files:
        - "model.pickle"
  script:
    - "python3 src/train.py --n ${self.params.num_iters}"
```

3. Model evaluation stage

```yaml
present_model: &present_model !BasicStage
  name: present_model
  inputs: *model_data
  outputs:
    - !path
      path: "results/default"
  env:
    MLDEV_MODEL_PATH: ${path(self.inputs[0].path)}
    RESULTS_PATH: ${self.outputs[0].path}
  script:
    - |
      python3 src/predict.py
      printf "=============================\n"
      printf "Test report:\n\n"
      cat ${path(self.outputs[0].path)}/test_report.json
      printf "\n\n=============================\n"
```

## License and usage of the example

The example is licensed under the Apache License 2.0.
See also [NOTICE](https://gitlab.com/mlrep/mldev/-/blob/develop/NOTICE.md)

