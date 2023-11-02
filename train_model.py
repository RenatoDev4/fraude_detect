import joblib
import pandas as pd

from functions.func import (create_train_test_data, run_model_training,
                            save_metrics)

path_to_data = "/data/raw_data.csv"
saving_folder = "/data/metrics"

prepared_data = pd.read_csv(path_to_data)

X_train, X_test, y_train, y_test = create_train_test_data(prepared_data)

classifier, metrics = run_model_training(X_train, y_train, X_test, y_test)

joblib.dump(classifier, "/models/fraud_detector_model.pkl")

final_file_name = "metrics.json"

save_metrics(metrics, saving_folder, final_file_name)
