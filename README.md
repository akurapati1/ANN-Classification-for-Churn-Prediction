# ANN Classification for Churn Prediction

## Overview

This project implements an Artificial Neural Network (ANN) to predict customer churn in a bank. Customer churn refers to the loss of clients or customers, and predicting it is crucial for businesses to retain customers and maintain revenue. The model is developed using Python and leverages deep learning techniques to classify whether a customer will churn or not.

## Table of Contents

- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Project structure](#project-structure)
- [Testing](#testing)
- [Deploy for free](#deploy-for-free-streamlit-community-cloud)
- [Dependencies](#dependencies)
- [License](#license)

## Dataset

The dataset used is `Churn_Modelling.csv`, which contains the following features:

- **RowNumber**: Index of the row.
- **CustomerId**: Unique identifier for each customer.
- **Surname**: Customer's surname.
- **CreditScore**: Customer's credit score.
- **Geography**: Customer's country.
- **Gender**: Customer's gender.
- **Age**: Customer's age.
- **Tenure**: Number of years the customer has been with the bank.
- **Balance**: Customer's account balance.
- **NumOfProducts**: Number of products the customer has with the bank.
- **HasCrCard**: Indicates if the customer has a credit card (1: Yes, 0: No).
- **IsActiveMember**: Indicates if the customer is an active member (1: Yes, 0: No).
- **EstimatedSalary**: Customer's estimated salary.
- **Exited**: Target variable indicating if the customer has churned (1: Yes, 0: No).

## Model Architecture

The ANN is constructed using the following architecture:

| Layer | Units | Activation |
|---|---|---|
| Dense (input: 12 features) | 64 | ReLU |
| Dense | 32 | ReLU |
| Dense (output) | 1 | Sigmoid |

Compiled with the Adam optimizer (learning rate 0.01) and binary cross-entropy loss; trained with early stopping (patience 10, best weights restored) on an 80/20 split, with TensorBoard logging. Inputs: gender is label-encoded, geography one-hot encoded, and all features standardised (`StandardScaler`).

## Installation

To set up the environment, follow these steps:

1. **Clone the repository**:

   ```bash
   git clone https://github.com/akurapati1/ANN-Classification-for-Churn-Prediction.git
   cd ANN-Classification-for-Churn-Prediction
   ```

2. **Create a virtual environment**:

   ```bash
   python -m venv env
   ```

3. **Activate the virtual environment**:

   - On Windows:

     ```bash
     .\env\Scripts\activate
     ```

   - On macOS/Linux:

     ```bash
     source env/bin/activate
     ```

4. **Install the required packages**:

   ```bash
   pip install -r requirements.txt
   ```


The `requirements.txt` includes all necessary dependencies.

## Usage

1. **Data Preprocessing**:

   The `experiments.ipynb` notebook contains code for data preprocessing, including encoding categorical variables and feature scaling.

2. **Training the Model**:

   Within the same notebook, the ANN model is built and trained on the preprocessed data.

3. **Making Predictions**:

   Use the `predictions.ipynb` notebook to load the trained model and make predictions on new data.

4. **Streamlit Application**:

   To interact with the model via a web interface, run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

   This will launch a local web application where you can input customer data and receive churn predictions.

## Results

The model reaches roughly **86% validation accuracy** on the held-out 20% split (see the training log in `experiments.ipynb`). Training curves can be inspected with TensorBoard (`tensorboard --logdir log/fit`).

## Project structure
| File | Purpose |
|---|---|
| `experiments.ipynb` | Preprocessing, training, saving `model.h5` and the encoders/scaler |
| `predictions.ipynb` | Loading the model and predicting on a sample customer |
| `churn_features.py` | Shared feature-building code (loads the pickled preprocessors) |
| `app.py` | Streamlit UI |
| `tests/` | Unit tests for the feature pipeline (`pytest`) |

## Testing
```bash
pip install -r requirements-dev.txt
pytest
```

## Deploy for free (Streamlit Community Cloud)
1. Sign in at <https://share.streamlit.io> with GitHub (free, no card).
2. **New app** → repository `akurapati1/ANN-Classification-for-Churn-Prediction`, branch `main`, main file `app.py`.
3. *Advanced settings*: choose **Python 3.11** (TensorFlow 2.16 does not support newer versions).
4. Deploy. The model and encoders are bundled in the repo, so no further setup is required.

## Dependencies
TensorFlow 2.16, scikit-learn 1.5.2 (pinned - the bundled encoders were pickled with it), pandas, NumPy, Streamlit. See `requirements.txt`.

## License
See [LICENSE](LICENSE).
