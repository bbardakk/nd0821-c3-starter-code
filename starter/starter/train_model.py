# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import pickle
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference
import pandas as pd

# Add code to load in the data.
data = pd.read_csv('data/census_cleaned.csv')

# Optional enhancement, use K-fold cross validation instead of a
# train-test split.
train, test = train_test_split(data, test_size=0.20)

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

# Proces the test data with the process_data function.
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False,
    encoder=encoder, lb=lb
)


# Train and save a model.
trained_model = train_model(X_train, y_train)
predictions = inference(trained_model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, predictions)

model_path = 'model/gb_model.pkl'
enc_path = 'model/encoder.pkl'
lb_path = 'model/lb.pkl'
inf_path = 'model/gb_model.info'


with open(inf_path, 'w') as f:
    f.write('Precision: %.3f\nRecall:%.3f\nfbeta:%.3f' %
            (precision, recall, fbeta))

with open(enc_path, 'wb') as f:
    pickle.dump(encoder, f)

with open(model_path, 'wb') as f:
    pickle.dump(trained_model, f)

with open(lb_path, 'wb') as f:
    pickle.dump(lb, f)
