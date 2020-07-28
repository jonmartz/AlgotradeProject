import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
dataset = pd.read_csv('data/processed_dataset.csv')
label_col_name = dataset.columns[-1]
X = dataset.drop(columns=[dataset.columns[0], label_col_name]).to_numpy()
y = dataset[label_col_name].to_numpy()
k_fold_cross_validation = KFold(n_splits=10)
MSEs = []
model_average_mse = []
fold = 0

models = [linear_model.LinearRegression()]
# Adding different alpha values for Lasso and Ridge
for alpha in range(1,10):
    models.append(linear_model.Ridge(alpha=alpha/10.0))
    models.append(linear_model.Lasso(alpha=alpha/10.0))

# Adding SGD regressor with diferrent penalties
models.append(linear_model.SGDRegressor(penalty='l2'))
models.append(linear_model.SGDRegressor(penalty='l1'))
models.append(linear_model.SGDRegressor(penalty='elasticnet'))

# Adding DecisionTreeRegressors with different maximum number of leaves
for splitter in {'best', 'random'}:
        models.append(DecisionTreeRegressor(max_leaf_nodes=None,splitter=splitter))
        models.append(DecisionTreeRegressor(max_leaf_nodes=5,splitter=splitter))
        models.append(DecisionTreeRegressor(max_leaf_nodes=10,splitter=splitter))
        models.append(DecisionTreeRegressor(max_leaf_nodes=20,splitter=splitter))
        models.append(DecisionTreeRegressor(max_leaf_nodes=30,splitter=splitter))
        models.append(DecisionTreeRegressor(max_leaf_nodes=100,splitter=splitter))

for model in models:
    print("Model - %s" % model)
    for train_index, test_index in k_fold_cross_validation.split(X):

        fold += 1
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        MSE = mean_squared_error(y_test, y_pred)
        MSEs.append(MSE)
        print('fold=%d MSE=%.6f' % (fold, MSE))

    model_average_mse.append(np.mean(MSEs))
    print('MSE_avg=%.6f MSE_std=%.6f\n ' % (np.mean(MSEs), np.std(MSEs)))

minimal = None
best = None
for mse_idx in range(len(model_average_mse)):
    if minimal == None or minimal > model_average_mse[mse_idx]:
        minimal = model_average_mse[mse_idx]
        best = models[mse_idx]

print("The Best model %s" % best)



