#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier, plot_importance
import xgboost as xgb
# from lightgbm import LGBMClassifier
# from catboost import CatBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTETomek
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import shap
import visualization


def train_evaluate_model(X, y, description='', learning_graph_show=False):
    print(f"\n🧪 Processing: {description}")
    print(f"Original class distribution:\n{pd.Series(y).value_counts()}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Apply SMOTE + Tomek (oversample + undersample)
    smt = SMOTETomek(random_state=42)
    X_resampled, y_resampled = smt.fit_resample(X_train, y_train)

    print(f"Resampled class distribution:\n{pd.Series(y_resampled).value_counts()}")

    # Train
    #model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

    #model.fit(X_resampled, y_resampled, eval_set=[(X_resampled, y_resampled), (X_test, y_test)],verbose=False)

    model = XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        max_depth=5,
        n_estimators=100,
        learning_rate=0.1
    )

    # Train the model
    model.fit(
        X_resampled, y_resampled,
        eval_set=[(X_resampled, y_resampled), (X_test, y_test)],
        verbose=False
    )

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("XGBoost Accuracy ========> ", accuracy * 100, "%")

    if learning_graph_show:
        visualization.draw_learning_curve(model)

    return accuracy, model

# Function: Train + SHAP analysis
def train_and_analyze(X, Y, label=""):
    y = Y.ravel() if isinstance(Y, np.ndarray) else Y.values.ravel()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = XGBClassifier()
    model.fit(X_train, y_train)

    # SHAP analysis
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': shap_values.values.mean(axis=0)
    }).sort_values(by='importance', ascending=False)

    return model, importance



#############################################################



def GB(X_train, X_test, y_train, y_test):
    num_boost_round = 100
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    model = GradientBoostingClassifier(n_estimators=num_boost_round, verbose=False)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print("GradientBoostingClassifier Accuracy ========> ", accuracy * 100, "%")

    return model


def XGBoost_for_all(X_resampled, y_resampled):
    num_boost_round = 100

    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    model = XGBClassifier(n_estimators=num_boost_round)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=100, verbose=False)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy ========> ", accuracy * 100, "%")
    return model


def XGBoost(X_train, X_test, y_train, y_test, col_name, learning_graph_show):
    num_boost_round = 100

    model = XGBClassifier(
        n_estimators=num_boost_round,
        # learning_rate=0.1,
        # max_depth=3,
        eval_metric=['logloss', 'error'],  # Monitor both logloss and error
        use_label_encoder=False
    )

    eval_set = [(X_train, y_train), (X_test, y_test)]
    model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("XGBoost Accuracy ========> ", accuracy * 100, "%")
    if learning_graph_show:
        visualization.draw_learning_curve(model)

    return model



def LightGBM(X_train, X_test, y_train, y_test):
    num_boost_round = 100

    model = LGBMClassifier(n_estimators=num_boost_round)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=100, verbose=False)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print("LightGBM Accuracy ========> ", accuracy * 100, "%")
    return model


def CatBoost(X_train, X_test, y_train, y_test):
    num_boost_round = 100

    model = CatBoostClassifier(n_estimators=num_boost_round)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=100, verbose=False)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print("CatBoost Accuracy ========> ", accuracy * 100, "%")
    return model


def RF(X_train, X_test, y_train, y_test):
    num_boost_round = 100

    model = RandomForestClassifier(n_estimators=num_boost_round, verbose=False)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print("RF Accuracy ========> ", accuracy * 100, "%")
    return model


def Dec_T(X_train, X_test, y_train, y_test):
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print("DT Accuracy ========> ", accuracy * 100, "%")
    return model


import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb

def feature_importance(model, X, df_original, meta):
    # 특성 중요도 추출
    importances = model.feature_importances_
    top_indices = np.argsort(importances)[::-1][:10]
    top_features = X.columns[top_indices]

    if not hasattr(meta, "column_labels"):
        raise AttributeError("meta 객체에 column_labels 속성이 없습니다.")

    # df_original을 기준으로 X.columns가 어디에 있는지 인덱스를 가져옴
    col_label_map = {}
    for col in X.columns:
        if col in df_original.columns:
            idx = df_original.columns.get_loc(col)
            if idx < len(meta.column_labels):
                col_label_map[col] = meta.column_labels[idx]
            else:
                col_label_map[col] = "Index out of range"
        else:
            col_label_map[col] = "Not in df_original"

    # 상위 10개 중요도 출력
    print("\n📌 Top 10 Feature Importances:")
    for feat in top_features:
        label = col_label_map.get(feat, "Unknown Label")
        print(f"{feat}: {label}")

    # 시각화
    xgb.plot_importance(model, importance_type='gain',
                        max_num_features=10,
                        title='Top 10 Feature Importance (Gain)',
                        xlabel='Gain', ylabel='Features')
    plt.show()



'''def feature_importance(model, X, df, meta):
    top_k = 12

    feature_importance = model.feature_importances_

    # feature_names = X.columns
    # importance_df = pd.DataFrame({
    #    'Feature': feature_names,
    #    'Importance': feature_importance
    # })

    # importance_df = importance_df.sort_values(by='Importance', ascending=False)

    # top_k_importances = importance_df.head(top_k)

    # top_k_features = top_k_importances['Feature'].values
    # top_k_values = top_k_importances['Importance'].values

    top_indices = np.argsort(feature_importance)[::-1][:top_k]
    top_indices = X.columns[top_indices]
    # print(top_indices)
    print("\nBelow shows the sorted features based on importance\n")
    for j in range(len(top_indices)):
        for i in range(len(meta.column_labels)):
            if df.columns[i] == top_indices[j]:
                print(f'{top_indices[j]}, {meta.column_labels[i]}')

    xgb.plot_importance(model, importance_type='gain',
                        max_num_features=10,  # Display only the top 10 features
                        title='Feature Importance', xlabel='Feature Importance', ylabel='Features')
    plt.show()
'''





