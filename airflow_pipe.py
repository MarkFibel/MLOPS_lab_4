from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import logging

import pandas as pd
from train_model import train
from preprocess_data import preprocess_and_add_features

def prepare_data():
    logging.info("📦 Загрузка и предобработка данных German Credit...")
    data = pd.read_csv('german.csv', sep=';')
    data = preprocess_and_add_features(data)
    data.to_csv('final.csv', sep=';', index=False)
    logging.info("✅ Данные сохранены как final.csv")

def train_tree():
    logging.info("🚀 Запуск обучения модели на German Credit...")
    model = train()
    logging.info("✅ Обучение завершено. Лучшая модель обучена и сохранена.")

with DAG(
    dag_id="german_training_pipeline",
    start_date=datetime(2025, 4, 26, 8, 5),
    schedule=timedelta(hours=3),  # теперь используется schedule
    catchup=False,
    max_active_runs=1,
    tags=["ml", "german_credit"]
) as dag:

    prepare_task = PythonOperator(
        task_id="prepare_german_data",
        python_callable=prepare_data
    )

    train_task = PythonOperator(
        task_id="train_german_model",
        python_callable=train_tree
    )

    prepare_task >> train_task
