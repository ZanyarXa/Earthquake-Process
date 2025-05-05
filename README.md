# Earthquake Data Analysis Pipeline

This project is a complete pipeline for importing, preprocessing, and analyzing earthquake data. It uses a SQLite database and Python scripts to perform data cleaning, feature engineering, and analytical queries.


project/
|
|── database/
|   |__ earthquakes.db
|
|── data/
|   |__ earthquakes.csv
|   |__ earthquakes_cleaned.csv
|   |__ engineered_features.pkl
|
|── scripts/ #creat database and preparing data
|   |__ models.py
|   |__ db_connection.py
|   |__ preprocess.py
|   |__ import_data.py
|   |__ feature_engineering.py
|
|── analysis/
|   |__ queries_filtering.py
|   |__ queries_highlighting.py
|
|__ pipline.py
|__ requirement.txt
|__ README.md
