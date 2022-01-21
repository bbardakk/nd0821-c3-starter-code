# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
In this project, we use Gradient Boosting based algorithm. Scikit-learn library is selected for implementation. For hyperparamter tunning, we use grid search with several different paramters.

## Intended Use
The main motivation of this ML project is predicting the salary of a person based on their personal information.

## Training Data
The name of data is "census-income". This dataset consist of 8 categorical and 6 numeric features. %80 of the whola dataset is used for training.

## Evaluation Data

%20 of the whole data is used for testing the modelling.

## Metrics

We evaluate our models with three different metrics. 

Precision: 0.832
Recall: 0.443
Fbeta: 0.578

## Ethical Considerations

The data is already anınymized. 

