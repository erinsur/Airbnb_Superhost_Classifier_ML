import pandas as pd
import numpy as np
import os 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

import pickle


# goal: train a machine learning model to predict whether an Airbnb host is a 'super host'. 


# building dataframe
filename = os.path.join(os.getcwd(), "data", "airbnbData_train.csv")
df = pd.read_csv(filename, header=0)
print("Built dataframe")

# create labels examples
y = df['host_is_superhost']
X = df.drop(columns='host_is_superhost', axis=1)
print("Created labeled examples")

# create train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=777)
print("Split into train and test datasets")

# To find optimal choice of hyperparameter C, we perform grid search cross-validation
cs = [10**i for i in range(-5,5)]
param_grid = {'C': cs}
model = LogisticRegression(max_iter=1000)
grid = GridSearchCV(model, param_grid, cv=5)
grid_search = grid.fit(X_train, y_train)
best_C = (grid_search.best_params_)['C']
print("Best C hyperparameter is found to be: " + str(best_C))

# train and test the optimal logistic Regression Model
model_best = LogisticRegression(C = best_C,max_iter=1000 )
model_best.fit(X_train, y_train)
class_label_predictions_best = model_best.predict(X_test)
c_m = confusion_matrix(y_test, class_label_predictions_best, labels=[True, False])
print("Confusion matrix for best model: \n")
print(pd.DataFrame(
    c_m,
    columns=['Predicted: Superhost', 'Predicted: Not Superhost'],
    index=['Actual: Superhost', 'Actual: Not Superhost']
))

# save best model
pkl_model_filename = "models/Pickle_Airbnb_Superhost_Classification_Model.pkl"
pickle.dump(model_best, open(pkl_model_filename, 'wb'))
print("Saved model to pickle file")

# verify model works
persistent_model = pickle.load(open(pkl_model_filename, 'rb'))
prediction = persistent_model.predict(X_test)
print(prediction)

print("Model is verified to be working!")
