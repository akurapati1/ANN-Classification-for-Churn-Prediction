# ANN Classification for Churn Prediction

## Overview

This project implements an Artificial Neural Network (ANN) to predict customer churn in a bank. Customer churn refers to the loss of clients or customers, and predicting it is crucial for businesses to retain customers and maintain revenue. The model is developed using Python and leverages deep learning techniques to classify whether a customer will churn or not.

## Table of Contents

- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Dataset

The dataset used is `Churn_Modelling.csv`, which contains the following features:

- **RowNumber**: Index of the row.
- **CustomerId**: Unique identifier for each customer.
- **Surname**: Customer's surname.
- **CreditScore**: Customer's credit score.
- **Geography**: Customer's country.
- **Gender**: Customer's gender.
- **Age**: Customer's age.
- **Tenure**: Number of years the customer has been with the bank.
- **Balance**: Customer's account balance.
- **NumOfProducts**: Number of products the customer has with the bank.
- **HasCrCard**: Indicates if the customer has a credit card (1: Yes, 0: No).
- **IsActiveMember**: Indicates if the customer is an active member (1: Yes, 0: No).
- **EstimatedSalary**: Customer's estimated salary.
- **Exited**: Target variable indicating if the customer has churned (1: Yes, 0: No).

## Model Architecture

The ANN is constructed using the following architecture:

- **Input Layer**: Accepts the input features.
- **Hidden Layers**: Two hidden layers with ReLU activation functions.
- **Output Layer**: Outputs the probability of customer churn using a sigmoid activation function.

The model is compiled with the Adam optimizer and binary cross-entropy loss function.

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


The `requirements.txt` includes all necessary dependencies.

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

The model's performance is evaluated using metrics such as accuracy, precision, recall, and the F1-score. Detailed results and visualizations are available in the `experiments.ipynb` notebook.


These dependencies are listed in the `requirements.txt` file and can be installed using the provided installation instructions.

