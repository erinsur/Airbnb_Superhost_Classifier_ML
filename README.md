# Logistic Regression Model Selection

## Overview

This project demonstrates an end-to-end machine learning workflow for building and selecting a Logistic Regression model for binary classification. The notebook walks through data preprocessing, model training, hyperparameter tuning, and model serialization to identify the best-performing model configuration.

## Objectives

* Prepare and preprocess data for machine learning
* Train a Logistic Regression classifier
* Compare model configurations and select the optimal model
* Save the trained model using Pickle for future predictions

## Project Structure

```
logistic-regression-model-selection/
├── data/
│   └── airbnbData_train.csv
├── src/
│   └── train.py
├── models/
│   └── Pickle_Airbnb_Superhost_Classification_Model.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Pickle

## Workflow

1. Load and explore the dataset
2. Clean and preprocess the data
3. Split the data into training and testing sets
4. Perform model selection and hyperparameter tuning
5. Train Logistic Regression model
6. Save the best model for deployment and future inference


## Running the Project

Clone the repository:

```bash
git clone https://github.com/erinsur/Airbnb_Superhost_Classifier_ML.git
cd Airbnb_Superhost_Classifier_ML
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the training script:

```bash
python src/train.py
```


## Author

Erin Sur
