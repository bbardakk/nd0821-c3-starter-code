import pandas as pd
from ml.model import test_slice

data = pd.read_csv('data/census_cleaned.csv')
test_slice(data, feature_col='education')