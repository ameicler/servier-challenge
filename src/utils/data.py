import os
import pandas as pd
from sklearn.model_selection import train_test_split

from utils.feature_extractor import fingerprint_features


def load_and_prepare_data(data_dir="data", model_name="2"):
    print("Loading and preparing data")
    df_single = pd.read_csv(os.path.join(data_dir, "dataset_single.csv"))
    # df_multi = pd.read_csv("data/dataset_multi.csv")

    if model_name == "1":
        df_single["fingerprint_features"] = df_single["smiles"].apply(
        lambda x: list(fingerprint_features(x)))
        df_features = pd.DataFrame(df_single["fingerprint_features"].values.tolist())
        X = df_features.values
    elif model_name == "2":
        X = df_single["smiles"]

    y = df_single["P1"]
    X_train, X_test, y_train, y_test = train_test_split(X, y,
        test_size=0.2, stratify=y, random_state=0)
    print("Data loaded and prepared")
    return X_train, X_test, y_train, y_test
