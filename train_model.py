import os
import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from datetime import datetime
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV

def train():
    data = pd.read_csv('fianl.csv', sep=';')
    X = data.drop(columns=['Creditability'])
    y = data['Creditability']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = DecisionTreeClassifier(random_state=42)

    param_grid = {
        'max_depth': [3, 5, 7, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy']
    }

    mlflow.set_experiment("Creditability_DecisionTree")

    with mlflow.start_run() as run:
        # 📌 Добавляем описание (тег) к запуску
        mlflow.set_tag("developer", "Danil Moiseenko")
        mlflow.set_tag("task", "Grid search for DecisionTree")
        mlflow.set_tag("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_

        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]

        roc_auc = roc_auc_score(y_test, y_proba)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"Best params: {grid_search.best_params_}")
        print(f"ROC AUC: {roc_auc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | Accuracy: {accuracy:.4f}")

        # ✅ Логируем параметры и метрики
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metrics({
            "roc_auc": roc_auc,
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy
        })

        # ✅ Логируем модель
        mlflow.sklearn.log_model(best_model, "model")

        # ✅ Сохраняем модель как .pkl
        models_dir = 'models'
        os.makedirs(models_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        model_path = os.path.join(models_dir, f"model_{timestamp}.pkl")
        joblib.dump(best_model, model_path)
        print(f"Model saved locally: {model_path}")

        # ✅ Логируем .pkl как артефакт
        mlflow.log_artifact(model_path)

        # ✅ Логируем текущий файл скрипта как артефакт
        current_script = os.path.realpath(__file__)
        mlflow.log_artifact(current_script)

    return best_model
