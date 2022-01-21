# Put the code for your API here.
from fastapi import FastAPI
from pydantic import BaseModel, Field
from starter.starter.ml.model import inference
from starter.starter.ml.data import process_data
import pickle


class Data(BaseModel):
    age: int = Field(None, example=23)
    workclass: str = Field(None, example='Private')
    fnlgt: int = Field(None, example=83311)
    education: str = Field(None, example='Bachelors')
    education_num: int = Field(None, example=9)
    marital_status: str = Field(None, example='Divorced')
    occupation: str = Field(None, example='Adm-clerical')
    relationship: str = Field(None, example='Not-in-family')
    race: str = Field(None, example='White')
    sex: str = Field(None, example='Female')
    capital_gain: int = Field(None, example=0)
    capital_loss: int = Field(None, example=0)
    hours_per_week: int = Field(None, example=0)
    native_country: str = Field(None, example='United-States')


with open('starter/model/gb_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('starter/model/encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

with open('starter/model/lb.pkl', 'rb') as f:
    lb = pickle.load(f)


app = FastAPI()


@app.get("/")
def root():
    return 'Welcome to project 3!'


@app.post("/")
async def infer(data: Data):

    # fix column name issues
    data = {key.replace('_', '-'): [value]
            for key, value in data.__dict__.items()}

    df = pd.DataFrame.from_dict(data)

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

    processed_row, _, _, _ = process_data(
        df, categorical_features=cat_features, label=None,
        training=False, encoder=encoder, lb=lb
    )

    pred = inference(model, processed_row)[0]

    if pred == 0:

        return '<=50K'

    if pred == 1:

        return '>50k'
