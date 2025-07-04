#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier, plot_importance
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

    
def XGBoost_for_all(X_resampled, y_resampled):    
    num_boost_round = 100
    
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    
    model = XGBClassifier(n_estimators=num_boost_round)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=100, verbose=False)
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy ========> ", accuracy*100, "%")
    return model 


def draw_learning_curve(model, col_name):
    results = model.evals_result()

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Times New Roman'

    epochs = len(results['validation_0']['logloss'])
    x_axis = range(0, epochs)
    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, results['validation_0']['logloss'], label='Train Log Loss')
    plt.plot(x_axis, results['validation_1']['logloss'], label='Validation Log Loss')
    plt.xlabel('Boosting Rounds', fontsize=18)
    plt.ylabel('Log Loss', fontsize=18)
    plt.title(f'{col_name} Learning Curve', fontsize=18)
    plt.legend(fontsize=15)
    plt.grid(True)
    plt.show()


def feature_importance(model, X, df, meta):
    top_k = 10
    
    feature_importance = model.feature_importances_

    feature_names = X.columns
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    })
    
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    
    top_k_importances = importance_df.head(top_k)
    
    #top_k_features = top_k_importances['Feature'].values
    #top_k_values = top_k_importances['Importance'].values
    
    top_indices = np.argsort(feature_importance)[::-1][:top_k]
    top_indices = X.columns[top_indices]
    print("\nBelow shows the sorted features based on importance\n")
    
    for j in range(len(top_indices)):
        for i in range(len(meta.column_labels)):
            if df.columns[i] == top_indices[j]:
                print(f'{top_indices[j]}, {meta.column_labels[i]}')
        
    xgb.plot_importance(model, importance_type='gain', 
                        max_num_features=10,  # Display only the top 10 features
                        title='Feature Importance', xlabel='Feature Importance', ylabel='Features')
    plt.show()

    return top_k_importances



