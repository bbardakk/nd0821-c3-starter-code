import pytest
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
import pandas as pd
from starter.starter.ml.data import process_data
from starter.starter.ml.model import inference


@pytest.fixture()
def data():
    data = pd.read_csv('starter/data/census_cleaned.csv')
    return data


@pytest.fixture()
def X(data):
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    train, _ = train_test_split(data, test_size=0.20)
    X, _, _, _ = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    return X


@pytest.fixture()
def model(X, y):
    dummy = DummyClassifier()
    dummy.fit(X, y)
    return dummy


@pytest.fixture()
def y(data):
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    train, _ = train_test_split(data, test_size=0.20)
    _, y, _, _ = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    return y


def test_inference(model, X):
    pred = inference(model, X)
    assert len(X) == len(pred)


def test_weekly_hour(data):
    assert data["hours-per-week"].between(1, 99).shape[0] == data.shape[0]


def test_mismatch_length(X, y):
    assert len(X) == len(y)
