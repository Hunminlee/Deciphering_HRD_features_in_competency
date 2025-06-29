#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier, plot_importance
import xgboost as xgb
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def GB(X_train, X_test, y_train, y_test):    
    num_boost_round = 100
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    
    model = GradientBoostingClassifier(n_estimators=num_boost_round, verbose=False)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print("GradientBoostingClassifier Accuracy ========> ", accuracy*100, "%")
    
    return model 
    
    
def XGBoost_for_all(X_resampled, y_resampled):    
    num_boost_round = 100
    
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    
    model = XGBClassifier(n_estimators=num_boost_round)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=100, verbose=False)
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy ========> ", accuracy*100, "%")
    return model 


def XGBoost(X_train, X_test, y_train, y_test, col_name):    
    num_boost_round = 100
    
    model = XGBClassifier(
        n_estimators=num_boost_round,
        #learning_rate=0.1,
        #max_depth=3,
        eval_metric=['logloss', 'error'],  # Monitor both logloss and error
        use_label_encoder=False
    )

    eval_set = [(X_train, y_train), (X_test, y_test)]
    model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("XGBoost Accuracy ========> ", accuracy*100, "%")
    
    draw_learning_curve(model, col_name)
    
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


def LightGBM(X_train, X_test, y_train, y_test):    
    num_boost_round = 100
    
    model = LGBMClassifier(n_estimators=num_boost_round)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=100, verbose=False)
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print("LightGBM Accuracy ========> ", accuracy*100, "%")
    return model 


def CatBoost(X_train, X_test, y_train, y_test):    
    num_boost_round = 100
    
    model = CatBoostClassifier(n_estimators=num_boost_round)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=100, verbose=False)
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print("CatBoost Accuracy ========> ", accuracy*100, "%")
    return model 


def RF(X_train, X_test, y_train, y_test):    
    num_boost_round = 100
    
    model = RandomForestClassifier(n_estimators=num_boost_round,  verbose=False)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print("RF Accuracy ========> ", accuracy*100, "%")
    return model 
    

def Dec_T(X_train, X_test, y_train, y_test):    
    
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print("DT Accuracy ========> ", accuracy*100, "%")
    return model 


def feature_importance(model, X, df, meta):
    top_k = 12
    
    feature_importance = model.feature_importances_

    #feature_names = X.columns
    #importance_df = pd.DataFrame({
    #    'Feature': feature_names,
    #    'Importance': feature_importance
    #})
    
    #importance_df = importance_df.sort_values(by='Importance', ascending=False)
    
    #top_k_importances = importance_df.head(top_k)
    
    #top_k_features = top_k_importances['Feature'].values
    #top_k_values = top_k_importances['Importance'].values

    
    top_indices = np.argsort(feature_importance)[::-1][:top_k]
    top_indices = X.columns[top_indices]
    #print(top_indices)
    print("\nBelow shows the sorted features based on importance\n")
    for j in range(len(top_indices)):
        for i in range(len(meta.column_labels)):
            if df.columns[i] == top_indices[j]:
                print(f'{top_indices[j]}, {meta.column_labels[i]}')
    
    
    
    xgb.plot_importance(model, importance_type='gain', 
                        max_num_features=10,  # Display only the top 10 features
                        title='Feature Importance', xlabel='Feature Importance', ylabel='Features')
    plt.show()






