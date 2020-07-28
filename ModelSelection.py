import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

dataset = pd.read_csv('data/processed_dataset.csv')
label_col_name = dataset.columns[-1]
X = dataset.drop(columns=[dataset.columns[0], label_col_name]).to_numpy()
y = dataset[label_col_name].to_numpy()
k_fold_cross_validation = KFold(n_splits=10)
MSEs = []
fold = 0
for train_index, test_index in k_fold_cross_validation.split(X):
    fold += 1
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # select model
    # model = linear_model.Ridge(alpha=0.5)
    model = linear_model.LinearRegression()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    MSE = mean_squared_error(y_test, y_pred)
    MSEs.append(MSE)
    print('fold=%d MSE=%.6f' % (fold, MSE))

print('\nMSE_avg=%.6f MSE_std=%.6f' % (np.mean(MSEs), np.std(MSEs)))

