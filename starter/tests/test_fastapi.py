from fastapi.testclient import TestClient
from main import app


def test_root():

    with TestClient(app) as testapp:
        response = testapp.get("/")

    assert response.status_code == 200
    assert response.json() == 'Welcome to project 3!'



def test_negative_sample():

    data = {
        'age': '37',
        'workclass': 'Private',
        'fnlgt': '280464',
        'education': 'Some-college',
        'education_num': '10',
        'marital_status': 'Married-civ-spouse',
        'occupation': 'Exec-managerial',
        'relationship': 'Husband',
        'race': 'White',
        'sex': 'Male',
        'capital_gain': '40000',
        'capital_loss': '0',
        'hours_per_week': '80',
        'native_country': 'United-States'
    }

    with TestClient(app) as testapp:
        response = testapp.post("/", json=data)

    assert response.status_code == 200
    assert response.json() == '>50k'
    
def test_positive_sample():

    data = {
        'age': '50',
        'workclass': 'Self-emp-not-inc',
        'fnlgt': '83311',
        'education': 'Bachelors',
        'education_num': '13',
        'marital_status': 'Married-civ-spouse',
        'occupation': 'Exec-managerial',
        'relationship': 'Husband',
        'race': 'White',
        'sex': 'Male',
        'capital_gain': '0',
        'capital_loss': '0',
        'hours_per_week': '13',
        'native_country': 'United-States'
    }

    with TestClient(app) as testapp:
        response = testapp.post("/", json=data)

    assert response.status_code == 200
    assert response.json() == '<=50K'
