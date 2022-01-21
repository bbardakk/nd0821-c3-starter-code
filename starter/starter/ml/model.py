from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

from .data import process_data

import pickle
import pandas as pd


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    gb_model = GradientBoostingClassifier()

    parameter_space = {
        "learning_rate": (1e-2, 1e-3),
        "n_estimators": (20, 50),
        "max_depth": [10, 15]
    }

    cs = GridSearchCV(gb_model, parameter_space, n_jobs=-1, verbose=2)
    cs.fit(X_train, y_train)

    print(
        'After hyper paramter tunning => best model: %s Hyperparams: %s best score: %s' %
        (cs.best_estimator_, cs.best_params_, cs.best_score_))

    return cs.best_estimator_


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)


def test_slice(data, feature_col):
    """Slice data based on the different values of a given feature, and test the model for each slice.

    Args:
        model : model to evaluate
        data (df): data to slice
        feature_col : name of the column to slice

    """

    with open('model/gb_model.pkl', 'rb') as f:
        model = pickle.load(f)

    with open('model/encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)

    with open('model/lb.pkl', 'rb') as f:
        lb = pickle.load(f)

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

    with open('model/%s_slice_output.txt' % feature_col, 'a') as f:
        f.write('Model metrics for %s features\n\n\n' % feature_col)

    for feature_val in data[feature_col].unique():
        sliced_data = data[data[feature_col] == feature_val]

        processed_data, y, _, _ = process_data(
            sliced_data, categorical_features=cat_features, label='salary',
            training=False, encoder=encoder, lb=lb
        )

        preds = inference(model, processed_data)
        precision, recall, fbeta = compute_model_metrics(y, preds)

        with open('model/%s_slice_output.txt' % feature_col, 'a') as f:
            f.write('Feature value: %s\n' % feature_val)
            f.write('Precision: %s\n' % precision)
            f.write('Recall: %s\n' % recall)
            f.write('F1 Beta: %s\n' % fbeta)
            f.write('\n\n' % fbeta)
