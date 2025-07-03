# run_pipeline.py

from scripts.feature_engineering import run_feature_engineering
from scripts.train_model import train_regression_model, train_classification_model

run_feature_engineering()

train_regression_model()

train_classification_model()
