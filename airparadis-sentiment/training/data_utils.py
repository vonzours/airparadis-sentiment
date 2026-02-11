import pandas as pd
from sklearn.model_selection import train_test_split
from training.config import TEXT_COL, LABEL_COL, TEST_SIZE, VAL_SIZE, RANDOM_STATE

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[[TEXT_COL, LABEL_COL]].dropna()
    df[TEXT_COL] = df[TEXT_COL].astype(str)
    df[LABEL_COL] = df[LABEL_COL].astype(int)
    return df

def split_train_val_test(df: pd.DataFrame):
    X = df[TEXT_COL].values
    y = df[LABEL_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # val proportion relative au train restant
    val_rel = VAL_SIZE / (1.0 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_rel, random_state=RANDOM_STATE, stratify=y_train
    )

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
