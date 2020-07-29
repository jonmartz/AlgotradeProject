import csv
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt


def get_MSE_absolute(plot_predictions=False):
    """
    Take the predicted percentage changes of the test set and apply them to the values
    of the SPY stock to get an MSE from the actual predicted SPY values.
    :param plot_predictions: set to true to plot the predictions in the test set of the current fold
    :return: MSE of the test predictions
    """
    spy_test = spy_values[test_index]
    y_pred_absolute = spy_test[:, 0] * (1 + y_pred)
    y_true_absolute = spy_test[:, 1]
    if plot_predictions:
        x = range(len(y_true_absolute))
        plt.plot(x, y_true_absolute, marker='.', label='true SPY')
        plt.plot(x, y_pred_absolute, marker='.', label='predicted SPY')
        plt.xlabel('block index (5 days per block)')
        plt.ylabel('SPY value')
        plt.legend()
        plt.title('Performance of %s, test set of one fold' % model_name)
        plt.savefig('predicted vs true.png')
        plt.show()
    return mean_squared_error(y_true_absolute, y_pred_absolute)


# LOAD DATASET AND SPLIT INTO X AND Y
dataset = pd.read_csv('data/processed_dataset.csv')
label_col_name = dataset.columns[-1]
X = dataset.drop(columns=['date range', label_col_name]).to_numpy()
y = dataset[label_col_name].to_numpy()
# for getting actual values of the SPY in each block
spy_values = pd.read_csv('data/absolute_spy_values.csv').drop(columns='date range').to_numpy()

# SELECT ALL THE MODELS THAT WILL BE TESTED
models = {}
models['linear regression'] = linear_model.LinearRegression()
# Adding different alpha values for Lasso and Ridge
for alpha in range(1, 10):
    models['ridge alpha_%0.1f' % (alpha / 10.0)] = linear_model.Ridge(alpha=alpha / 10.0)
    models['lasso alpha_%0.1f' % (alpha / 10.0)] = linear_model.Lasso(alpha=alpha / 10.0)
# Adding SGD regressor with different penalties
models['SGD l2'] = linear_model.SGDRegressor(penalty='l2')
models['SGD l1'] = linear_model.SGDRegressor(penalty='l1')
models['SGD elasticnet'] = linear_model.SGDRegressor(penalty='elasticnet')
# Adding DecisionTreeRegressors with different maximum number of leaves
for splitter in {'best', 'random'}:
    models['tree splitter_%s max_leaf=None' % splitter] = DecisionTreeRegressor(max_leaf_nodes=None, splitter=splitter)
    models['tree splitter_%s max_leaf=5' % splitter] = DecisionTreeRegressor(max_leaf_nodes=5, splitter=splitter)
    models['tree splitter_%s max_leaf=10' % splitter] = DecisionTreeRegressor(max_leaf_nodes=10, splitter=splitter)
    models['tree splitter_%s max_leaf=20' % splitter] = DecisionTreeRegressor(max_leaf_nodes=20, splitter=splitter)
    models['tree splitter_%s max_leaf=30' % splitter] = DecisionTreeRegressor(max_leaf_nodes=30, splitter=splitter)
    models['tree splitter_%s max_leaf=100' % splitter] = DecisionTreeRegressor(max_leaf_nodes=100, splitter=splitter)

# PERFORM THE 10-FOLD CROSS VALIDATION FOR EACH MODEL
k_fold_cross_validation = KFold(n_splits=10)
MSEs_absolute = []  # MSEs relative to the actual SPY values
MSEs_pct_change = []  # MSEs relative to the pct_change in SPY between blocks
model_average_mse = []  # to save the MSE averaged of all the folds for each model
with open('data/results.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['model name', 'MSE_absolute_avg', 'MSE_absolute_std', 'MSE_pct_change_avg', 'MSE_pct_change_std'])
    for model_name, model in models.items():
        print("model: %s" % model)
        fold = 0
        for train_index, test_index in k_fold_cross_validation.split(X):
            fold += 1
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # get predictions
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            MSE_absolute = get_MSE_absolute()
            MSE_pct_change = mean_squared_error(y_test, y_pred)
            MSEs_absolute.append(MSE_absolute)
            MSEs_pct_change.append(MSE_pct_change)
            print('\tfold=%d MSE=%.6f' % (fold, MSE_absolute))

        model_average_mse.append(np.mean(MSEs_absolute))
        print('MSE_absolute_avg=%.6f MSE_absolute_std=%.6f | MSE_pct_change_avg=%.6f MSE_pct_change_std=%.6f\n'
              % (np.mean(MSEs_absolute), np.std(MSEs_absolute), np.mean(MSEs_pct_change), np.std(MSEs_pct_change)))

        writer.writerow([model_name, np.mean(MSEs_absolute), np.std(MSEs_absolute),
                         np.mean(MSEs_pct_change), np.std(MSEs_pct_change)])

# PRINT THE WINNING MODEL
minimal_average_MSE = None
best_model_name = None
for mse_idx in range(len(model_average_mse)):
    if minimal_average_MSE is None or minimal_average_MSE > model_average_mse[mse_idx]:
        minimal_average_MSE = model_average_mse[mse_idx]
        best_model_name = models[list(models.keys())[mse_idx]]
print("the best model is: %s" % best_model_name)
