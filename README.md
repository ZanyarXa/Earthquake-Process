# Earthquake Data Analysis Pipeline

This project is a complete pipeline for importing, preprocessing, and analyzing earthquake data. It uses a SQLite database and Python scripts to perform data cleaning, feature engineering, and analytical queries.

To run the code, first install the required libraries using the following command.
`pip install -r requirements.txt`

Then, by running the pipeline, you can access the cleaned and engineered data in the database.
`python pipeline.py`

if you using python3:
`python3 pipeline.py`

## File Structure

project/ <br>
| <br>
|── database/ <br>
|   |__ earthquakes.db <br>
| <br>
|── data/ <br>
|   |__ earthquakes.csv <br>
|   |__ earthquakes_cleaned.csv <br>
|   |__ engineered_features.pkl <br>
| <br>
|── scripts/ #creat database and preparing data <br>
|   |__ models.py <br>
|   |__ db_connection.py <br>
|   |__ preprocess.py <br>
|   |__ import_data.py<br>
|   |__ feature_engineering.py<br>
|<br>
|── analysis/<br>
|   |__ queries_filtering.py<br>
|   |__ queries_highlighting.py<br>
|<br>
|__ pipline.py<br>
|__ requirement.txt<br>
|__ README.md<br>
