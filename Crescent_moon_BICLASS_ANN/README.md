# Artificial Neural Network for Binary Classification Using the `make_moons` Dataset

## Project Overview

This project implements a feed-forward Artificial Neural Network (ANN) to solve a binary classification problem using the synthetic **`make_moons` dataset** provided by Scikit-learn. The dataset contains two interleaving, crescent-shaped classes, making it a useful example for demonstrating how a neural network can learn a nonlinear decision boundary.

The implementation is written in Python and uses TensorFlow/Keras to construct and train the neural network. It also uses Scikit-learn for dataset generation, splitting, feature standardization, and evaluation, while Pandas and Matplotlib support data inspection and visualization.

The program follows an end-to-end machine learning workflow:

1. Import the required libraries and configure random seeds.
2. Generate 10,000 two-dimensional samples with controlled noise.
3. Inspect and visualize the dataset.
4. Split the data into training and testing subsets.
5. Standardize the input features.
6. Define and compile the ANN.
7. Train the network and monitor validation performance.
8. Evaluate predictions on the held-out test set.
9. Generate classification metrics and a confusion matrix.
10. Plot training curves and the learned decision boundary.
11. Predict the class of a new sample.
12. Export the generated dataset to a CSV file.

This repository is intended as an educational implementation for understanding the core components of a neural-network classification pipeline.

---

## Table of Contents

- [Objectives](#objectives)
- [Problem Statement](#problem-statement)
- [Dataset Description](#dataset-description)
- [Technology Stack](#technology-stack)
- [Repository Structure](#repository-structure)
- [System Requirements](#system-requirements)
- [Installation and Setup](#installation-and-setup)
- [How to Run](#how-to-run)
- [Implementation Details](#implementation-details)
  - [1. Library Imports and Reproducibility](#1-library-imports-and-reproducibility)
  - [2. Dataset Generation](#2-dataset-generation)
  - [3. Dataset Visualization](#3-dataset-visualization)
  - [4. Train-Test Split](#4-train-test-split)
  - [5. Feature Standardization](#5-feature-standardization)
  - [6. ANN Architecture](#6-ann-architecture)
  - [7. Model Compilation](#7-model-compilation)
  - [8. Model Training](#8-model-training)
  - [9. Model Evaluation](#9-model-evaluation)
  - [10. Confusion Matrix and Classification Report](#10-confusion-matrix-and-classification-report)
  - [11. Training History Visualization](#11-training-history-visualization)
  - [12. Decision Boundary Visualization](#12-decision-boundary-visualization)
  - [13. Prediction on a New Sample](#13-prediction-on-a-new-sample)
  - [14. Dataset Export](#14-dataset-export)
- [Model Configuration Summary](#model-configuration-summary)
- [Evaluation Metrics Explained](#evaluation-metrics-explained)
- [Expected Outputs](#expected-outputs)
- [Reproducibility Notes](#reproducibility-notes)
- [Troubleshooting](#troubleshooting)
- [Possible Enhancements](#possible-enhancements)
- [Limitations](#limitations)
- [Conclusion](#conclusion)
- [License](#license)

---

## Objectives

The main objectives of this project are to:

- Demonstrate how an Artificial Neural Network can classify two classes that are not linearly separable.
- Implement a multilayer feed-forward neural network using TensorFlow/Keras.
- Understand the role of hidden layers and nonlinear activation functions.
- Apply feature standardization before neural-network training.
- Separate data used for model fitting from data used for final testing.
- Monitor training and validation accuracy and loss.
- Evaluate a binary classifier using accuracy, a classification report, and a confusion matrix.
- Visualize the decision regions learned by the network.
- Perform inference on a new input sample.
- Save the generated dataset for further analysis or reuse.

## Problem Statement

A binary classification model must assign each input sample to one of two possible classes. In this project, every sample has two numerical input features. The classes form interleaving half-moon shapes, so a straight-line decision boundary is generally insufficient to separate them accurately.

The task is to train an ANN that learns a nonlinear mapping from the two input features to a binary output label.

Given an input vector:

\[
X = [x_1, x_2]
\]

the model estimates the probability that the sample belongs to Class 1:

\[
P(y=1 \mid X)
\]

The output probability is converted into a class label using a threshold of 0.5:

- If the predicted probability is greater than or equal to 0.5, the predicted label is `1`.
- Otherwise, the predicted label is `0`.

The model's performance is assessed using test data that was not used to fit the network weights.

## Dataset Description

### Source

The dataset is generated programmatically with Scikit-learn's `make_moons` function. It is a synthetic dataset rather than a collection of real-world observations.

### Dataset parameters

| Parameter | Value |
|---|---|
| Dataset generator | `sklearn.datasets.make_moons` |
| Number of samples | 10,000 |
| Number of input features | 2 |
| Number of target classes | 2 |
| Noise level | 0.20 |
| Random state | 42 |
| Training proportion | 80% |
| Testing proportion | 20% |
| Training samples | 8,000 |
| Testing samples | 2,000 |

The exact training and testing counts follow from the configured 80/20 split.

### Input features

- **Feature 1:** The first coordinate of a sample in the two-dimensional feature space.
- **Feature 2:** The second coordinate of a sample in the two-dimensional feature space.

These features are synthetic coordinates and do not represent specific physical measurements.

### Target variable

The `Target` column contains the class label:

- `0` — Class 0
- `1` — Class 1

The dataset is generated with a fixed random state so that the generated samples are reproducible under the same library behavior and configuration.

### Why use `make_moons`?

The `make_moons` dataset is commonly used to demonstrate nonlinear classification. The two classes form curved regions that overlap in their coordinate ranges. This makes the problem more informative than a dataset where the classes can be separated easily with a straight line.

---

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| NumPy | Numerical arrays, transformations, and sample preparation |
| Pandas | Tabular data organization and CSV export |
| Matplotlib | Dataset, metric, and decision-boundary visualizations |
| Scikit-learn | Dataset generation, data splitting, scaling, and evaluation metrics |
| TensorFlow | Deep-learning framework |
| Keras | Neural-network construction, compilation, training, and inference |

### Main APIs used

- `make_moons()` — Generates the synthetic classification dataset.
- `train_test_split()` — Divides samples into training and testing sets.
- `StandardScaler()` — Standardizes input features.
- `Sequential()` — Defines a sequential neural-network model.
- `Dense()` — Creates fully connected neural-network layers.
- `Adam()` — Configures the Adam optimization algorithm.
- `model.fit()` — Trains the neural network.
- `model.evaluate()` — Evaluates loss and configured metrics.
- `model.predict()` — Produces model predictions.
- `accuracy_score()` — Computes classification accuracy.
- `confusion_matrix()` — Counts correct and incorrect predictions by class.
- `classification_report()` — Reports precision, recall, F1-score, and support.
- `ConfusionMatrixDisplay` — Visualizes the confusion matrix.

---

## Repository Structure

The supplied project archive contains the following Python script:

```text
Crescent_moon_BICLASS_ANN/
└── colab_code.py
```

After running the script, the program also creates a dataset file in the current working directory:

```text
make_moons_dataset.csv
```

The CSV file is generated at runtime and is not required as an input because the dataset is created by the program itself.

This README can be saved as `README.md` in the `Crescent_moon_BICLASS_ANN/` directory.

---

## System Requirements

### Recommended environment

- Python 3.10 or a compatible version supported by the installed TensorFlow release.
- Internet access for installing dependencies, if they are not already available.
- A terminal, command prompt, Jupyter environment, or Google Colab.
- A machine capable of running a small fully connected neural network.

A GPU is **not required** for this dataset and model size. Training can generally be performed on a CPU.

### Python dependencies

The implementation imports the following packages:

- `numpy`
- `pandas`
- `matplotlib`
- `tensorflow`
- `scikit-learn`

---

## Installation and Setup

### Option A: Run locally

**1. Extract the project archive.**

Extract the ZIP file and open a terminal in the `Crescent_moon_BICLASS_ANN` directory.

**2. Create a virtual environment (recommended).**

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**3. Install the required libraries.**

```bash
python -m pip install --upgrade pip
python -m pip install numpy pandas matplotlib tensorflow scikit-learn
```

If a particular TensorFlow version is required by your environment, install a compatible version according to the official TensorFlow installation guidance.

**4. Run the Python script.**

```bash
python colab_code.py
```

The script prints dataset information, the network architecture, training progress, evaluation results, and the final experiment summary. It also displays several plots and writes `make_moons_dataset.csv` to the current working directory.

### Option B: Run in Google Colab

The script includes a Google Colab-specific download step. To run it in Colab:

1. Open Google Colab.
2. Create a new notebook.
3. Upload `colab_code.py` or copy its contents into a notebook cell.
4. Ensure the imported libraries are installed. Colab commonly includes many of them already.
5. Run the script.
6. When the dataset export section executes, the script attempts to download `make_moons_dataset.csv` through `google.colab.files.download()`.

The download code is wrapped in a `try/except ImportError` block. Outside Colab, the CSV is saved locally and a message is printed instead.

---

## How to Run

The script is designed to run from beginning to end without requiring a separate dataset file.

### Execution workflow

1. **Initialization:** Load dependencies and set random seeds.
2. **Data creation:** Generate the two-class dataset.
3. **Inspection:** Print the sample count, feature count, class count, sample records, and class distribution.
4. **Visualization:** Plot both classes in two-dimensional space.
5. **Preprocessing:** Split and standardize the data.
6. **Model creation:** Construct and compile the ANN.
7. **Training:** Fit the network for the configured number of epochs.
8. **Evaluation:** Compute test loss, test accuracy, and classification metrics.
9. **Visualization:** Display the confusion matrix, training curves, and decision boundary.
10. **Inference:** Predict the class of a new sample.
11. **Export:** Save the synthetic dataset as a CSV file.
12. **Summary:** Print the final configuration and test accuracy.

The training progress and final numerical metrics may vary slightly between environments, software versions, and hardware. The code prints the actual values produced by each run; this README does not assume a specific accuracy score.

---

# Implementation Details

## 1. Library Imports and Reproducibility

The implementation begins by importing libraries for numerical computation, tabular data handling, plotting, machine learning, and deep learning.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
```

The program sets random seeds:

```python
np.random.seed(42)
tf.random.set_seed(42)
```

Using fixed seeds helps make data generation and model initialization more reproducible. It does not guarantee bit-for-bit identical training results on every device or across all TensorFlow configurations.

## 2. Dataset Generation

The data is generated with:

```python
X, y = make_moons(
    n_samples=10000,
    noise=0.20,
    random_state=42
)
```

Here:

- `n_samples=10000` creates 10,000 observations.
- `noise=0.20` adds variation to the idealized moon-shaped data.
- `random_state=42` makes dataset generation repeatable.

`X` contains the two input coordinates for every sample, while `y` contains the corresponding binary labels.

The data is then organized in a Pandas DataFrame:

```python
df = pd.DataFrame(
    X,
    columns=["Feature_1", "Feature_2"]
)
df["Target"] = y
```

This tabular representation is useful for inspecting records, checking class counts, and exporting the data.

## 3. Dataset Visualization

The initial scatter plot displays Class 0 and Class 1 using separate plotting calls:

```python
plt.scatter(
    X[y == 0, 0],
    X[y == 0, 1],
    label="Class 0",
    alpha=0.5
)

plt.scatter(
    X[y == 1, 0],
    X[y == 1, 1],
    label="Class 1",
    alpha=0.5
)
```

The plot helps verify that the data contains two curved groups. The visualization also gives an intuitive explanation for why a model needs nonlinear activation functions to learn a useful boundary.

## 4. Train-Test Split

The data is divided into training and testing subsets:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

The split uses 80% of the observations for training and 20% for testing. With 10,000 observations, this produces 8,000 training samples and 2,000 test samples.

The `stratify=y` argument preserves the approximate class proportions in both subsets. The test set is kept separate from model fitting so it can provide a more meaningful estimate of performance on unseen samples.

## 5. Feature Standardization

The input features are standardized using `StandardScaler`:

```python
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

Standardization transforms each feature using the mean and standard deviation learned from the training set. In general, a standardized feature is computed as:

\[
z = \frac{x-\mu}{\sigma}
\]

where:

- \(x\) is the original feature value,
- \(\mu\) is the training-set mean,
- \(\sigma\) is the training-set standard deviation.

The scaler is **fitted only on the training data**. The same fitted transformation is then applied to the test set. This avoids using information from the test set to calculate preprocessing parameters.

The same scaler is also applied later to the grid used for decision-boundary visualization and to the new sample used for prediction.

## 6. ANN Architecture

The network is defined with Keras' `Sequential` API:

```python
model = Sequential([
    Input(shape=(2,)),
    Dense(32, activation="relu"),
    Dense(16, activation="relu"),
    Dense(8, activation="relu"),
    Dense(1, activation="sigmoid")
])
```

### Layer-by-layer explanation

| Layer | Configuration | Purpose |
|---|---|---|
| Input | 2 features | Accepts the two standardized input coordinates |
| Hidden layer 1 | 32 neurons, ReLU | Learns initial nonlinear feature representations |
| Hidden layer 2 | 16 neurons, ReLU | Learns more complex combinations of features |
| Hidden layer 3 | 8 neurons, ReLU | Refines the representation for binary classification |
| Output layer | 1 neuron, sigmoid | Produces a probability-like score for Class 1 |

### ReLU activation

The hidden layers use the Rectified Linear Unit (ReLU) activation function:

\[
\operatorname{ReLU}(x)=\max(0,x)
\]

ReLU introduces nonlinearity, allowing the network to learn curved and more complex decision boundaries.

### Sigmoid activation

The output layer uses sigmoid activation:

\[
\sigma(x)=\frac{1}{1+e^{-x}}
\]

Its output lies between 0 and 1, making it suitable for a binary classifier trained with binary cross-entropy loss.

## 7. Model Compilation

The network is compiled as follows:

```python
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
```

### Optimizer: Adam

Adam updates the network weights using estimates of the gradients' first and second moments. The configured learning rate is `0.001`.

### Loss function: Binary cross-entropy

Binary cross-entropy measures the difference between the true binary labels and the probabilities predicted by the model. For a single example, its standard form is:

\[
L=-\left[y\log(p)+(1-y)\log(1-p)\right]
\]

where \(y\) is the true label and \(p\) is the predicted probability of Class 1.

The training process seeks to reduce this loss by adjusting the network's learnable weights.

### Metric: Accuracy

Accuracy reports the fraction of predictions that match the true labels under the metric's classification threshold. It provides a simple overall performance measure, but it should be considered alongside precision, recall, F1-score, and the confusion matrix.

## 8. Model Training

The model is trained using:

```python
history = model.fit(
    X_train,
    y_train,
    epochs=150,
    batch_size=32,
    validation_split=0.20,
    verbose=1
)
```

The configuration means:

- **Epochs:** The model can pass through the training subset 150 times.
- **Batch size:** The model processes 32 training samples per batch.
- **Validation split:** 20% of the supplied training subset is reserved by Keras for validation during training.
- **Verbose:** Training progress is printed.

Because validation is split from the 8,000-sample training subset, the network is fitted on approximately 6,400 samples per epoch, while approximately 1,600 are used for validation. The 2,000-sample test set remains separate for final evaluation.

The returned `history` object stores metrics such as training accuracy, validation accuracy, training loss, and validation loss for each epoch.

The curves can help diagnose learning behavior. For example, if training loss continues to decrease while validation loss begins to rise, the model may be overfitting. The curves alone do not prove overfitting; they should be interpreted together with the task and evaluation results.

## 9. Model Evaluation

The script evaluates the model on the held-out test set:

```python
test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)
```

It then obtains predicted probabilities and converts them into class labels:

```python
y_prob = model.predict(X_test, verbose=0).ravel()
y_pred = (y_prob >= 0.5).astype(int)
```

`y_prob` contains the model's output score for each test observation. `y_pred` contains the corresponding predicted class labels based on the 0.5 threshold.

The implementation calculates accuracy separately with Scikit-learn:

```python
accuracy = accuracy_score(y_test, y_pred)
```

The Keras evaluation accuracy and Scikit-learn accuracy should generally agree closely when using the same labels and standard threshold behavior, though small differences can occur due to metric implementation details.

## 10. Confusion Matrix and Classification Report

The classification report is generated with:

```python
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Class 0", "Class 1"],
        zero_division=0
    )
)
```

It includes:

- **Precision:** Of the observations predicted as a class, the fraction that truly belong to that class.
- **Recall:** Of the observations that truly belong to a class, the fraction correctly identified.
- **F1-score:** The harmonic mean of precision and recall.
- **Support:** The number of true observations for each class in the test set.

The confusion matrix is created with:

```python
cm = confusion_matrix(y_test, y_pred)
```

For a binary classifier, it summarizes true and false predictions for the two classes. In Scikit-learn's default label ordering, the matrix rows represent true labels and the columns represent predicted labels.

The display is produced using:

```python
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Class 0", "Class 1"]
)

disp.plot()
plt.title("ANN Confusion Matrix")
plt.tight_layout()
plt.show()
```

The diagonal cells represent correct predictions. Off-diagonal cells represent misclassifications.

## 11. Training History Visualization

The script plots training and validation accuracy, then training and validation loss:

```python
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
```

and:

```python
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
```

These graphs help explain how model performance changes across epochs.

### How to interpret the plots

- Increasing accuracy generally indicates that the classifier is correctly classifying more samples.
- Decreasing loss generally indicates that predicted probabilities are improving under the selected loss function.
- Similar training and validation trends can indicate that performance is generalizing reasonably well.
- A widening gap between training and validation metrics may indicate overfitting or a mismatch between training and validation data behavior.
- Flat curves can indicate that the model has converged, the learning rate is unsuitable, or further training is not helping.

No single curve should be interpreted in isolation. Final conclusions should also consider held-out test performance.

## 12. Decision Boundary Visualization

The script creates a mesh over the original two-dimensional feature space:

```python
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)

grid = np.c_[xx.ravel(), yy.ravel()]
grid_scaled = scaler.transform(grid)
```

Every grid point is standardized with the same scaler fitted on the training data. The model then predicts a score for each point:

```python
Z = model.predict(grid_scaled, verbose=0).ravel()
Z = (Z >= 0.5).astype(int).reshape(xx.shape)
```

Finally, `contourf()` displays the predicted class regions, while the original samples are plotted on top.

This visualization is useful because it shows **where the network assigns each class**, rather than reporting metrics alone. A successful model should learn a boundary that follows the curved separation between the moon-shaped classes, subject to the generated noise and the model's learned parameters.

## 13. Prediction on a New Sample

The script demonstrates inference with a sample:

```python
sample = np.array([[0.5, 0.2]])
sample_scaled = scaler.transform(sample)

probability = model.predict(
    sample_scaled,
    verbose=0
)[0][0]

predicted_class = int(probability >= 0.5)
```

The sample is first transformed using the previously fitted scaler. This is important: inference must use the same feature transformation that was used during training.

The program prints the original input coordinates, the predicted probability of Class 1, and the resulting predicted class.

Because the network is trained on a synthetic dataset, this example is a demonstration of the inference process rather than a prediction about a real-world phenomenon.

## 14. Dataset Export

The original generated data is converted into a DataFrame and saved:

```python
dataset = pd.DataFrame(
    X,
    columns=["Feature_1", "Feature_2"]
)

dataset["Target"] = y

dataset.to_csv(
    "make_moons_dataset.csv",
    index=False
)
```

The CSV contains three columns:

| Column | Description |
|---|---|
| `Feature_1` | First generated input coordinate |
| `Feature_2` | Second generated input coordinate |
| `Target` | Binary class label |

The `index=False` argument prevents Pandas from adding a separate index column to the exported file.

When the program runs in Google Colab, it attempts to trigger a file download. When it runs in a local Python environment, the file remains in the current working directory.

---

## Model Configuration Summary

| Component | Setting |
|---|---|
| Task | Binary classification |
| Dataset | Synthetic `make_moons` |
| Total samples | 10,000 |
| Input dimensions | 2 |
| Output classes | 2 |
| Noise | 0.20 |
| Train-test split | 80:20 |
| Feature scaling | `StandardScaler` |
| Model type | Feed-forward ANN |
| Hidden layers | 3 |
| Hidden-layer sizes | 32, 16, 8 |
| Hidden activation | ReLU |
| Output units | 1 |
| Output activation | Sigmoid |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss | Binary cross-entropy |
| Training epochs | 150 |
| Batch size | 32 |
| Validation fraction | 20% of the training subset |
| Classification threshold | 0.5 |
| Primary evaluation | Test loss and accuracy |
| Additional evaluation | Precision, recall, F1-score, confusion matrix |
| Export format | CSV |

---

### Console output

- Dataset size and number of features.
- Number of target classes.
- The first five dataset records.
- Class distribution.
- Training and testing sample counts.
- ANN model summary.
- Training progress across epochs.
- Test loss and test accuracy.
- Classification report.
- Confusion matrix values.
- Prediction results for the demonstration sample.
- Confirmation that the CSV dataset was saved.
- Final experiment configuration and test accuracy.

### Visual output

1. **Dataset scatter plot:** Displays the two classes before preprocessing.
2. **Confusion matrix:** Shows correct and incorrect classifications.
3. **Training and validation curves:** Shows accuracy and loss over the training epochs.
4. **Decision-boundary plot:** Shows the class regions learned by the neural network.

### Generated file

- `make_moons_dataset.csv`

**Note:** Numerical evaluation results are generated when the code runs. The actual test accuracy and classification metrics should be taken from that run's console output rather than assumed in advance.

---

## Reproducibility Notes

The implementation sets NumPy's and TensorFlow's random seeds and uses fixed `random_state` values for data generation and splitting. This improves repeatability and makes experimentation easier to compare.

However, reproducibility can still be affected by:

- Different versions of TensorFlow, Keras, NumPy, or Scikit-learn.
- Differences in hardware and numerical libraries.
- Nondeterministic operations in some execution environments.
- Changes to the script's random seeds or dataset parameters.

For more rigorous reproducibility, record the dependency versions, Python version, configuration values, and the evaluation results from each run.

## Troubleshooting

### 1. `ModuleNotFoundError`

**Cause:** A required library is not installed in the active Python environment.

**Solution:**

```bash
python -m pip install numpy pandas matplotlib tensorflow scikit-learn
```

Make sure the command is run in the same virtual environment or Python installation used to execute the script.

### 2. TensorFlow installation problems

TensorFlow support depends on the Python version and operating system. If installation fails, check the official TensorFlow installation documentation and choose a compatible Python/TensorFlow combination. Google Colab may be a convenient alternative for running the experiment.

### 3. Matplotlib plots do not appear

Some terminal environments do not open interactive plot windows. Try running the script in Jupyter Notebook, JupyterLab, or Google Colab. Ensure that the local plotting backend is configured appropriately for the operating system.

### 4. CSV file cannot be found

The CSV is saved relative to the process's **current working directory**, which may differ from the directory containing the script. Check the terminal's working directory before running the program. You can also replace the output filename with an explicit destination path if required.

### 5. Results differ between runs

Minor differences can occur because of software versions and execution environment behavior. Confirm that the random seeds, dataset settings, train-test split, and model architecture have not changed.

### 6. Training takes longer than expected

The script trains for 150 epochs and displays progress. For quick testing, temporarily reduce the epoch count. Restore the intended configuration when conducting the final experiment.

### 7. Accuracy is lower than expected

Check that the correct scaled arrays are passed to the model, the labels align with their corresponding samples, and the test set is evaluated only after training. Inspect the loss curves and decision-boundary plot before changing model parameters.

---

## Possible Enhancements

The current implementation provides a complete baseline. The following improvements could extend the project:

1. **Save the trained model:** Use Keras model-saving functionality to persist the network for later inference.
2. **Save the scaler:** Persist the fitted `StandardScaler` so future samples are transformed consistently.
3. **Separate preprocessing and training:** Organize data generation, preprocessing, training, evaluation, and plotting into reusable functions.
4. **Use a configuration file:** Store dataset size, noise, architecture, learning rate, epochs, and batch size in one place.
5. **Add early stopping:** Stop training when validation performance no longer improves, potentially reducing unnecessary epochs.
6. **Add model checkpointing:** Save the best model according to a chosen validation metric.
7. **Compare architectures:** Evaluate different hidden-layer sizes, activation functions, and numbers of layers.
8. **Tune hyperparameters:** Compare learning rates, batch sizes, and epoch counts systematically.
9. **Compare with baseline classifiers:** Evaluate logistic regression, support vector machines, decision trees, or other suitable models on the same split.
10. **Add ROC-AUC analysis:** Use predicted probabilities to evaluate ranking performance across classification thresholds.
11. **Improve output reporting:** Save metrics and plots to files for inclusion in a project report.
12. **Create an interactive interface:** Build a small Streamlit application that accepts two feature values and displays the predicted class and probability.
13. **Add automated tests:** Test dataset dimensions, preprocessing behavior, output shape, and inference flow.
14. **Track experiments:** Record model configuration, random seeds, package versions, and metrics for each run.

Any enhancement should be evaluated on held-out data and documented clearly so that results remain interpretable.

## Limitations

- The dataset is synthetic and does not demonstrate performance on real-world data.
- Only two input features are used.
- The model architecture and hyperparameters are fixed rather than optimized through systematic search.
- The default decision threshold is 0.5; different applications may require a threshold selected according to their objectives.
- Accuracy alone cannot describe every aspect of classifier performance.
- A good result on `make_moons` does not imply that the same model will perform well on a different dataset.
- The script exports the dataset but does not currently save the trained neural network or fitted scaler.
- Fixed random seeds improve reproducibility but do not guarantee identical results across all environments.

## Conclusion

This project demonstrates an end-to-end implementation of binary classification using an Artificial Neural Network. Starting with a synthetic, nonlinear dataset, the workflow covers data generation, inspection, visualization, splitting, feature standardization, network construction, training, evaluation, inference, and data export.

The multilayer network uses ReLU activations in its hidden layers and a sigmoid output to estimate the probability of the positive class. Adam optimization and binary cross-entropy provide the training configuration, while test accuracy, a classification report, and a confusion matrix support evaluation. Training curves and the decision-boundary visualization offer additional insight into how the model learns.

The project provides a practical foundation for further study of neural networks, nonlinear classification, model evaluation, and machine-learning experimentation.
