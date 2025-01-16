import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, GlobalAveragePooling1D
from tensorflow.keras.metrics import MeanSquaredError
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error


# Custom RMSE metric
def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))


# Function to compile and train the LSTM model
def train_and_evaluate_model(X_train, y_train, X_test, y_test, n_splits=10):
    model = Sequential([
        Embedding(input_dim=3, output_dim=12),
        LSTM(units=12, return_sequences=True),
        GlobalAveragePooling1D(),
        Dense(units=1, activation='linear')
    ])

    model.compile(optimizer='adam', loss=rmse, metrics=[MeanSquaredError()])
    model.fit(X_train, y_train, epochs=500, verbose=0)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    mse_scores = []

    for train_index, test_index in kf.split(X_test):
        X_fold_test = X_test[test_index]
        y_fold_test = y_test[test_index]

        y_pred = model.predict(X_fold_test)
        mse = np.sqrt(mean_squared_error(y_fold_test, y_pred.flatten()))
        mse_scores.append(mse)

    return mse_scores


# Load phenotype data
phenotype_data = pd.read_csv('your_data.csv', header=None)

# Read haplotype files
haplotypes_by_row = [[] for _ in range(x)]  # x is the total number of haplotype blocks

for i in range(len(phenotype_data)):
    haplotype_file = f'output_{i}.txt'
    with open(haplotype_file, 'r') as file:
        patient_haplotypes = [line.strip() for line in file.readlines()]
        for row_index, haplotype in enumerate(patient_haplotypes):
            haplotypes_by_row[row_index].append([int(allele) for allele in haplotype])

# Train and evaluate model for each haplotype set
all_mse_scores = []

for haplotype_set in haplotypes_by_row:
    haplotypes = np.array(haplotype_set)
    phenotypes = np.array(phenotype_data)
    X_train, X_test, y_train, y_test = train_test_split(haplotypes, phenotypes, test_size=0.4, random_state=42)
    mse_scores = train_and_evaluate_model(X_train, y_train, X_test, y_test)
    all_mse_scores.append(mse_scores)

# Return all MSE scores
print(all_mse_scores)
