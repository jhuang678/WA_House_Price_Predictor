######################################################################
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error

import math
import numpy as np
import random
#####################################################################

def detect_outliers(dataframe, feature):
    Q1 = dataframe[feature].quantile(0.25)
    Q3 = dataframe[feature].quantile(0.75)
    IQR = Q3 - Q1
    outliers = dataframe[(dataframe[feature] < Q1 - 1.5 * IQR) | (dataframe[feature] > Q3 + 1.5 * IQR)]
    return outliers

def print_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    adjusted_r2 = 1 - (1-r2)*(len(y_true)-1)/(len(y_true)-X.shape[1]-1)
    print('Mean Absolute Error (MAE):', mae)
    print('Mean Squared Error (MSE):', mse)
    print('Root Mean Squared Error (RMSE):', rmse)
    print('R-squared:', r2)
    print('Adjusted R-squared:', adjusted_r2)
    print()


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    # Calculating the metrics
    MAE = mean_absolute_error(y_test, y_pred)
    MSE = mean_squared_error(y_test, y_pred)
    RMSE = math.sqrt(MSE)
    R2 = r2_score(y_test, y_pred)
    adjusted_R2 = 1 - (1 - R2) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)

    print("Mean Absolute Error:", MAE)
    print("Mean Squared Error:", MSE)
    print("Root Mean Squared Error:", RMSE)
    print("R-squared:", R2)
    print("Adjusted R-squared:", adjusted_R2)
    print()

if __name__ == '__main__':
    df = pd.read_csv('city_washington.csv')

    # Exploratory Data Analysis (EDA)
    print(df.head())
    print()
    print(df.info())
    #corr_matrix = df.corr()
    #print(corr_matrix["median_house_price"].sort_values(ascending=False))

    scaler = StandardScaler()
    X = df.drop(['city', 'median_house_price'], axis=1)
    y = df['median_house_price']
    X = scaler.fit_transform(X)
    X = pd.DataFrame(X, columns=df.drop(['city', 'median_house_price'], axis=1).columns)

    lasso = LassoCV(cv=5, random_state=42).fit(X, y)
    #print('Coefficients: ', lasso.coef_)
    selected_features = X.columns[(lasso.coef_ != 0)]
    #print('Selected features: ', selected_features)
    X = X[selected_features]
    #print(X)

    # Model Tuning
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    ###################################################################################################################
    # Training
    '''
    # Linear Regression
    print("Linear Regression")
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    #y_pred = lin_reg.coef_ * X_train + lin_reg.intercept_
    y_pred = lin_reg.predict(X_train)
    print("Coefficient:", lin_reg.coef_)
    print("R-squared Score:", r2_score(y_train, y_pred))

    # SVR
    print("SVR")
    param_grid = {'C': [0.1, 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000]}
    grid = GridSearchCV(SVR(), param_grid, refit=True, verbose=3)
    grid.fit(X_train, y_train)
    print("Best Parameters:", grid.best_params_)
    print(grid.best_estimator_)
    score = grid.score(X_test, y_test)
    print("R-squared Score:", score)
    print()

    # Kernel SVR
    print("Kernel SVR")
    param_grid = {'C': [0.1, 1, 10, 100, 1000, 10000, 100000, 1000000],
                  'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                  'kernel': ['rbf']}
    grid = GridSearchCV(SVR(), param_grid, refit=True, verbose=3)
    grid.fit(X_train, y_train)
    print("Best Parameters:", grid.best_params_)
    print(grid.best_estimator_)
    score = grid.score(X_test, y_test)
    print("R-squared Score:", score)
    print()

    # Decision Tree
    print("Decision Tree")
    param_grid = {
        'criterion': ['squared_error', 'friedman_mse', 'absolute_error'],
        'splitter': ['best'],
        'max_depth': [None, 40, 50, 60, 70],
        'min_samples_split': [5, 10, 15, 20],
        'min_samples_leaf': [4, 6, 8, 10]
    }
    grid = GridSearchCV(estimator=DecisionTreeRegressor(), param_grid=param_grid, cv=5, verbose=2)
    grid.fit(X_train, y_train)
    print("Best Parameters:", grid.best_params_)
    print(grid.best_estimator_)
    score = grid.score(X_test, y_test)
    print("R-squared Score:", score)
    print()

    # Random Forest
    print("Random Forest")
    param_grid = {
        'n_estimators': [200, 300, 400],
        'min_samples_split': [10, 15, 20, 25, 30],
        'min_samples_leaf': [4, 8, 12]
    }
    grid = GridSearchCV(estimator=RandomForestRegressor(random_state=42), param_grid=param_grid, cv=5, verbose=2)
    grid.fit(X_train, y_train)
    print("Best Parameters:", grid.best_params_)
    print(grid.best_estimator_)
    score = grid.score(X_test, y_test)
    print("R-squared Score:", score)
    print()

    # KNN
    print("KNN")
    param_grid = {
        'n_neighbors': [1, 3, 5, 10, 15, 20],
        'weights': ['uniform', 'distance'],
        'leaf_size': [10, 20, 30, 40, 50]
    }
    grid = GridSearchCV(estimator=KNeighborsRegressor(), param_grid=param_grid, cv=5, verbose=2)
    grid.fit(X_train, y_train)
    print("Best Parameters:", grid.best_params_)
    print(grid.best_estimator_)
    score = grid.score(X_test, y_test)
    print("R-squared Score:", score)
    print()

    # Gradient Boosting
    print("Gradient Boosting")
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.05, 0.1, 0.2],
        'max_depth': [3, 4],
        'subsample': [0.8, 0.9, 1.0]
    }
    grid = GridSearchCV(estimator=GradientBoostingRegressor(), param_grid=param_grid, cv=5, verbose=2)
    grid.fit(X_train, y_train)
    print("Best Parameters:", grid.best_params_)
    print(grid.best_estimator_)
    score = grid.score(X_test, y_test)
    print("R-squared Score:", score)
    print()

    # Elastic-Net Regression
    print("Elastic-Net Regression")
    param_grid = {
        'alpha': [0.001, 0.01, 0.1, 1, 10],
        'l1_ratio': [0.2, 0.4, 0.6, 0.8, 1.0]
    }
    grid = GridSearchCV(estimator=ElasticNet(), param_grid=param_grid, cv=5, verbose=2)
    grid.fit(X_train, y_train)
    print("Best Parameters:", grid.best_params_)
    print(grid.best_estimator_)
    score = grid.score(X_test, y_test)
    print("R-squared Score:", score)
    print()
    '''
    ###################################################################################################################
    # Testing
    # Linear Regression
    print("Linear Regression")
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    evaluate_model(lin_reg, X_test, y_test)

    # SVR
    print("Support Vector Regression (SVR)")
    svr_model = SVR(C=10000000)
    svr_model.fit(X_train, y_train)
    evaluate_model(svr_model, X_test, y_test)

    # Kernel SVR
    print("Kernel SVR")
    kernel_svr_model = SVR(C=1000000, gamma=0.01)
    kernel_svr_model.fit(X_train, y_train)
    evaluate_model(kernel_svr_model, X_test, y_test)

    # Decision Tree
    print("Decision Tree")
    dt_model = DecisionTreeRegressor(criterion='absolute_error', max_depth=40, min_samples_leaf=4, min_samples_split=20)
    dt_model.fit(X_train, y_train)
    evaluate_model(dt_model, X_test, y_test)

    # Random Forest with
    print("Random Forest")
    rf_model = RandomForestRegressor(min_samples_leaf=4, min_samples_split=25, n_estimators=300, random_state=42)
    rf_model.fit(X_train, y_train)
    evaluate_model(rf_model, X_test, y_test)

    # K-Nearest Neighbors
    print("K-Nearest Neighbors")
    knn_model = KNeighborsRegressor(leaf_size=10, n_neighbors=3, weights='distance')
    knn_model.fit(X_train, y_train)
    evaluate_model(knn_model, X_test, y_test)

    # Gradient Boosting
    print("Gradient Boosting")
    gb_model = GradientBoostingRegressor(learning_rate=0.1, max_depth=3, n_estimators=100, subsample=0.8)
    gb_model.fit(X_train, y_train)
    evaluate_model(gb_model, X_test, y_test)

    # Elastic-Net Regression
    print('Elastic-Net Regression:')
    elastic_net = ElasticNet(alpha=0.01, l1_ratio=0.2)
    elastic_net.fit(X_train, y_train)
    evaluate_model(elastic_net, X_test, y_test)


    '''
    df.hist(bins=40, figsize=(20, 20))
    plt.tight_layout()  # Optional, to improve spacing between subplots
    plt.title('Features Histogram', size=18, fontweight="bold")
    plt.savefig('hist.png')
    plt.close()

    plt.figure(figsize=(18, 14))
    sns.heatmap(df.corr(), annot=False, linewidths=0.03, cmap='coolwarm')
    plt.title('Correlation Heatmap', size=18, fontweight="bold")
    plt.savefig('heatmap.png')
    plt.close()


    # Outlier Detection
    numeric_cols = [column for column in df.columns if df[column].dtype in ['int64', 'float64']]

    # Outlier
    num_plots = len(numeric_cols)
    num_cols = 5  
    num_rows = 6
    print(num_rows)
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, num_rows * 5))
    for ind, column in enumerate(numeric_cols):
        i = ind // num_cols
        j = ind % num_cols
        outliers = detect_outliers(df, column)
        print(f'Number of outliers in {column}:', outliers.shape[0])

        # Plot boxplot
        if num_rows == 1:
            if num_cols == 1:
                sns.boxplot(x=df[column], ax=axs)
                axs.set_title(f'Boxplot - {column}')
            else:
                sns.boxplot(x=df[column], ax=axs[j])
                axs[j].set_title(f'Boxplot - {column}')
        else:
            sns.boxplot(x=df[column], ax=axs[i, j])
            axs[i, j].set_title(f'Boxplot - {column}')
    plt.tight_layout()
    plt.savefig('outlier_detection.png')
    plt.close()
    '''




