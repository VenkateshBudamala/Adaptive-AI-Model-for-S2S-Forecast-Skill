# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 17:06:56 2023

@author: Dr. Venkatesh Budamala
"""


################################ Libraries ####################################
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from keras.models import Sequential
from keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import sqlite3
from scipy.stats import uniform, randint


############################## Global Variables #################################
var = ['Tmax','Tmin','PCP']  # Variables to process
os.chdir(r'D:\Data\Climate_model_data\S2S\Save') #Database Location

############################## Functions ######################################
def load_data(variable):
    """
    Load the data from the SQLite database and return it as a DataFrame.
    """

    
    conn = sqlite3.connect('S2S_India.db')
    query = f"SELECT * FROM {variable}_Sorted"
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Clean and preprocess data
    df = df.iloc[:, 1:]
    df = df.apply(pd.to_numeric, errors='coerce').round(2)
    
    if variable == 'PCP':
        df['S2S'] = df['S2S'] + 100
        df['Obs'] = df['Obs'] + 100
    else:
        df['S2S'] = df['S2S'] + 273.15
        df['Obs'] = df['Obs'] + 273.15
    
    return df


def prepare_data(df):
    """
    Prepare the input features (X) and target variable (y) for training.
    """
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y


def split_data(X, y):
    """
    Split the data into training and testing sets.
    """
    return train_test_split(X, y, test_size=0.2, random_state=42)


def evaluate_model(model, trainX, trainY, testX, testY, df):
    """
    Evaluate the model by computing RMSE scores for training and testing datasets.
    """
    model.fit(trainX, trainY)
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    
    trainScore = np.sqrt(np.mean((trainY - trainPredict) ** 2))
    testScore = np.sqrt(np.mean((testY - testPredict) ** 2))
    rawScore = np.sqrt(np.mean((df['Obs'] - df['S2S']) ** 2))
    
    result = f"RMSE Scores:\nRaw Score: {rawScore}\nTrain Score: {trainScore}\nTest Score: {testScore}"
    return result


def save_results(df, variable, model):
    """
    Save the model results into the SQLite database.
    """
    df['ML'] = model.predict(df.iloc[:, :-1].values) - 273.15
    conn = sqlite3.connect('S2S_India.db')
    table_name = f"{variable}_ML"
    df.to_sql(table_name, conn, index=False, if_exists='replace')
    conn.close()


def LSTM_Model(X, y):
    """
    Build and train the LSTM model with adaptive hyperparameter tuning.
    """
    trainX, testX, trainY, testY = split_data(X, y)
    
    scaler = StandardScaler()
    trainX = scaler.fit_transform(trainX)
    testX = scaler.transform(testX)
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    
    # Define hyperparameters for LSTM model
    param_dist = {
        'units': randint(50, 500),
        'batch_size': randint(16, 256),
        'epochs': randint(50, 200)
    }
    
    # Use RandomizedSearchCV for adaptive hyperparameter tuning
    model = Sequential()
    model.add(LSTM(500, return_sequences=True, input_shape=(1, trainX.shape[2])))
    model.add(LSTM(500, return_sequences=False))
    model.add(Dense(100))
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    randomized_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=10, cv=3, n_jobs=-1, verbose=2)
    randomized_search.fit(trainX, trainY)
    
    best_model = randomized_search.best_estimator_
    return best_model


def XGB_Model(X, y):
    """
    Build and train the XGBoost model with adaptive hyperparameter tuning using RandomizedSearchCV.
    """
    trainX, testX, trainY, testY = split_data(X, y)
    
    model = XGBRegressor(objective='reg:squarederror')
    
    # Hyperparameter tuning using RandomizedSearchCV
    param_dist = {
        'n_estimators': randint(100, 1000),
        'learning_rate': uniform(0.01, 0.2),
        'max_depth': randint(3, 15),
        'subsample': uniform(0.5, 0.5),
        'colsample_bytree': uniform(0.3, 0.7)
    }
    
    randomized_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=50, cv=3, n_jobs=-1, verbose=2)
    randomized_search.fit(trainX, trainY)
    
    best_model = randomized_search.best_estimator_
    return best_model


def RF_Model(X, y):
    """
    Build and train the Random Forest model with adaptive hyperparameter tuning using RandomizedSearchCV.
    """
    trainX, testX, trainY, testY = split_data(X, y)
    
    model = RandomForestRegressor(random_state=42)
    
    # Hyperparameter tuning using RandomizedSearchCV
    param_dist = {
        'n_estimators': randint(50, 300),
        'max_depth': randint(5, 15),
        'min_samples_split': randint(2, 10),
        'min_samples_leaf': randint(1, 10)
    }
    
    randomized_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=50, cv=3, n_jobs=-1, verbose=2)
    randomized_search.fit(trainX, trainY)
    
    best_model = randomized_search.best_estimator_
    return best_model


def SVM_Model(X, y):
    """
    Build and train the Support Vector Machine model with adaptive hyperparameter tuning using RandomizedSearchCV.
    """
    trainX, testX, trainY, testY = split_data(X, y)
    
    model = SVR(kernel='rbf')
    
    # Hyperparameter tuning using RandomizedSearchCV
    param_dist = {
        'C': uniform(1, 100),
        'gamma': uniform(0.01, 0.1),
        'epsilon': uniform(0.1, 1)
    }
    
    randomized_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=50, cv=3, n_jobs=-1, verbose=2)
    randomized_search.fit(trainX, trainY)
    
    best_model = randomized_search.best_estimator_
    return best_model


def CNN_Model(X, y):
    """
    Build and train the CNN model with adaptive hyperparameter tuning using RandomizedSearchCV.
    """
    trainX, testX, trainY, testY = split_data(X, y)
    
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))
    
    # Define hyperparameters for CNN model
    param_dist = {
        'filters': randint(32, 256),
        'kernel_size': randint(2, 5),
        'batch_size': randint(16, 256),
        'epochs': randint(50, 200)
    }
    
    # Create CNN model
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(trainX.shape[1], 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    randomized_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=10, cv=3, n_jobs=-1, verbose=2)
    randomized_search.fit(trainX, trainY)
    
    best_model = randomized_search.best_estimator_
    return best_model


def select_best_model(X, y):
    """
    Select the best model adaptively.
    """
    models = {
        'XGBoost': XGB_Model(X, y),
        'RandomForest': RF_Model(X, y),
        'SVM': SVM_Model(X, y),
        'LSTM': LSTM_Model(X, y),
        'CNN': CNN_Model(X, y)
    }
    
    best_model_name = None
    best_model_score = float('inf')
    
    for model_name, model in models.items():
        result = evaluate_model(model, X, y)
        model_score = float(result.split("\n")[-1].split(":")[-1].strip())
        
        if model_score < best_model_score:
            best_model_score = model_score
            best_model_name = model_name
            
    return best_model_name, models[best_model_name]


def main():
    for variable in var:
        df = load_data(variable)
        X, y = prepare_data(df)
        
        # Select the best model adaptively
        best_model_name, best_model = select_best_model(X, y)
        
        # Evaluate and print results
        result = evaluate_model(best_model, X, y)
        print(f"Best Model: {best_model_name}\n{result}")
        
        # Save the results to the database
        save_results(df, variable, best_model)


if __name__ == "__main__":
    main()
