import json
import os
from typing import Any, Dict, Tuple

import requests
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split


def data_acquisition(url: str, folder: str, final_file_name: str) -> None:
    """
    Downloads data from a given URL and saves it to a specified folder with a given file name.

    Parameters:
        url (str): The URL from which to download the data.
        folder (str): The folder in which to save the downloaded data.
        final_file_name (str): The name of the file to save the data as.

    Returns:
        None
    """
    try:
        response = requests.get(url)
        response.raise_for_status()

        conteudo = response.content
        patch_way = os.path.join(folder, final_file_name)

        with open(patch_way, 'wb') as arquivo:
            arquivo.write(conteudo)

        print(f'Dados baixados e salvos com sucesso em {patch_way}')

    except requests.exceptions.RequestException as e:
        print(f"Erro ao baixar os dados: {e}")


def create_train_test_data(data: DataFrame, test_size: float = 0.3, random_state: int = 42) -> tuple:
    """
    Splits the input data into train and test sets.

    Args:
        data (DataFrame): The input data to be split.
        test_size (float, optional): The proportion of the data to be used for testing. Defaults to 0.3.
        random_state (int, optional): The seed used by the random number generator. Defaults to 42.

    Returns:
        tuple: A tuple containing the X_train, X_test, y_train, and y_test sets.
    """
    train, test = train_test_split(
        data, test_size=test_size, random_state=random_state)
    X_train = train.drop(columns=['fraud'])
    X_test = test.drop(columns=['fraud'])
    y_train = train['fraud']
    y_test = test['fraud']

    return X_train, X_test, y_train, y_test


def run_model_training(X_train: Any, y_train: Any, X_test: Any, y_test: Any, random_state: int = 42) -> Tuple[Any, Dict[str, float]]:
    """
    Train a classification model and return the trained model and classification metrics.

    Parameters:
    X_train (Any): Training dataset.
    y_train (Any): Labels of the training dataset.
    X_test (Any): Test dataset.
    y_test (Any): Labels of the test dataset.
    random_state (int, optional): Random seed for ensuring reproducibility. Default is 42.

    Returns:
    Tuple[Any, Dict[str, float]]: A tuple containing the trained model and a dictionary of classification metrics.
    """
    random_forest = RandomForestClassifier(random_state=random_state)

    random_forest.fit(X_train, y_train)

    y_predictions_rf = random_forest.predict(X_test)

    accuracy = accuracy_score(y_test, y_predictions_rf)
    precision = precision_score(y_test, y_predictions_rf)
    recall = recall_score(y_test, y_predictions_rf)
    f1 = f1_score(y_test, y_predictions_rf)

    metrics = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1_score': f1
    }

    return random_forest, metrics


def save_metrics(metrics: Dict[str, float], saving_folder: str, final_file_name: str) -> None:
    """
    Save classification metrics to a JSON file.

    Parameters:
    metrics (Dict[str, float]): A dictionary containing classification metrics.
    saving_folder (str): The directory where the JSON file will be saved.
    final_file_name (str): The name of the JSON file.

    Returns:
    None
    """
    file_path = os.path.join(saving_folder, final_file_name)

    with open(file_path, "w") as file:
        json.dump(metrics, file)
