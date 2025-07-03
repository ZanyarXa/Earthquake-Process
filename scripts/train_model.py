import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBRegressor, XGBClassifier

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping


def print_regression_metrics(model_name, y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\n=== {model_name} ===")
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE    : {rmse:.4f}")
    print(f"MAE     : {mae:.4f}")


def predict_proba(model, X_test):
    if hasattr(model, "predict_proba"):
        if len(model.classes_) == 2:
            return model.predict_proba(X_test)[:, 1]
        else:
            if model.classes_[0] == 1:
                return np.ones(len(X_test))
            else:
                return np.zeros(len(X_test))
    elif hasattr(model, "predict"):
        return model.predict(X_test).flatten()
    else:
        return model.predict(X_test).flatten()


import matplotlib.pyplot as plt
import seaborn as sns

def print_classification_metrics(name, y_true, y_pred_proba, label_encoder):
    y_pred = np.argmax(y_pred_proba, axis=1)
    print(f"\n--- {name} ---")
    print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))

    cm = confusion_matrix(y_true, y_pred)
    
    # === توضیح متنی
    print("\nConfusion Matrix (detailed explanation):")
    for i, label in enumerate(label_encoder.classes_):
        row = cm[i]
        total = sum(row)
        print(f"\nTrue class: {label} (Total: {total})")
        for j, val in enumerate(row):
            pred_label = label_encoder.classes_[j]
            if i == j:
                print(f"  → Predicted as {pred_label}: {val} ✅ Correct")
            else:
                print(f"  → Predicted as {pred_label}: {val} ❌ Incorrect")
    
    # === رسم گرافیکی بلافاصله بعدش
#    print("RUNNING CONFUSION MATRIX")
#    plt.figure(figsize=(6, 5))
#    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                xticklabels=label_encoder.classes_,
#                yticklabels=label_encoder.classes_)
#    plt.title(f"Confusion Matrix - {name}")
#    plt.xlabel("Predicted")
#    plt.ylabel("True")
#    plt.tight_layout()
#    plt.save_fig("scripts/confusion.png")
#    plt.close("all")
#    plt.show()


def load_and_prepare_data():
    df = pd.read_csv("engineered_dataset_combined.csv")

    # اصلاح نام ستون‌ها به lowercase
    df.columns = [col.lower() for col in df.columns]

    # اگر time وجود داشت، ویژگی‌های زمانی را استخراج کن
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
        df['year'] = df['time'].dt.year
        df['month'] = df['time'].dt.month
        df['quarter'] = df['time'].dt.quarter
        df['day_of_week'] = df['time'].dt.dayofweek

    # میانگین سالیانه بزرگی زلزله
    if 'year' in df.columns and 'magnitude' in df.columns:
        df['yearly_avg_magnitude'] = df.groupby('year')['magnitude'].transform('mean')

    # عددی‌سازی depth_category
    if 'depth_category' in df.columns:
        df['depth_category'] = df['depth_category'].astype(str)
        depth_mapping = {'Shallow': 0, 'Intermediate': 1, 'Deep': 2}
        df['depth_category_num'] = df['depth_category'].map(depth_mapping)

    return df

def train_regression_model():
    print("=== Training Regression Models ===")

    try:
        df = pd.read_csv("engineered_dataset_combined.csv")
        required_columns = [
            "distance_to_fault_km", "depth", "moho_depth", "crustal_thickness", "num_neighbors_50km",
            "year", "month", "day_of_week", "quarter",
            "yearly_avg_magnitude",
            "depth_to_magnitude", "magnitude_depth_interaction"
        ]
        if not set(required_columns).issubset(df.columns):
            raise KeyError("Missing expected columns in combined dataset")
        source = "combined"
    except (FileNotFoundError, KeyError):
        print("Falling back to engineered_dataset.csv...")
        df = pd.read_csv("engineered_dataset.csv")
        source = "basic"

    # تعریف فیچرها بر اساس نوع فایل
    if source == "combined":
        numeric_features = [
            "distance_to_fault_km", "depth", "moho_depth", "crustal_thickness", "num_neighbors_50km",
            "year", "month", "day_of_week", "quarter",
            "yearly_avg_magnitude",
            "depth_to_magnitude", "magnitude_depth_interaction"
        ]
        if 'depth_category_num' in df.columns:
            numeric_features.append('depth_category_num')
        onehot_features = [col for col in df.columns if col.startswith("type_") or col.startswith("magnitude_type_")]
    else:
        numeric_features = [
            "distance_to_fault_km", "Depth", "moho_depth", "crustal_thickness", "num_neighbors_50km"
        ]
        onehot_features = []

    features = numeric_features + onehot_features

    # حالا ادامه‌ی پردازش با df و features...

    features = numeric_features + onehot_features

    target_column = "magnitude" if source == "combined" else "Magnitude"
    df_model = df.dropna(subset=features + [target_column])
    X = df_model[features].values
    y = df_model[target_column].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    print_regression_metrics("Random Forest", y_test, y_pred_rf)

    # XGBoost
    xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    print_regression_metrics("XGBoost", y_test, y_pred_xgb)

    # Neural Network
    tf.random.set_seed(42)
    nn_model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    nn_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    nn_model.fit(
        X_train_scaled, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=0
    )

    y_pred_nn = nn_model.predict(X_test_scaled).flatten()
    print_regression_metrics("Neural Network", y_test, y_pred_nn)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

def predict_proba(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    else:  # TensorFlow model
        return model.predict(X)

def print_classification_metrics(model_name, y_true, y_pred_proba, label_encoder):
    y_pred = np.argmax(y_pred_proba, axis=1)
    print(f"\n--- {model_name} ---")
    print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

def train_classification_model():
    print("=== Training Classification Models ===")

    # 1. Load Data
    df_combined = pd.read_csv("engineered_dataset_combined.csv")
    df_basic = pd.read_csv("engineered_dataset.csv")

    combined_cols = [
        "magnitude_category", "depth_to_magnitude", "magnitude_depth_interaction"
    ]
    basic_cols = [
        "distance_to_fault_km", "Depth", "moho_depth", "crustal_thickness",
        "num_neighbors_50km", "Year", "Month", "Day"
    ]

    for col in combined_cols:
        if col not in df_combined.columns:
            raise ValueError(f"Column '{col}' not found in engineered_dataset_combined.csv")
    for col in basic_cols:
        if col not in df_basic.columns:
            raise ValueError(f"Column '{col}' not found in engineered_dataset.csv")

    # Convert month to number if necessary
    month_map = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4,
        'May': 5, 'June': 6, 'July': 7, 'August': 8,
        'September': 9, 'October': 10, 'November': 11, 'December': 12
    }
    if df_basic['Month'].dtype == object:
        df_basic['Month'] = df_basic['Month'].map(month_map)

    # 2. Combine Data
    df_combined_selected = df_combined[combined_cols]
    df_basic_selected = df_basic[basic_cols]
    df_model = pd.concat([df_combined_selected, df_basic_selected], axis=1)

    # 3. Prepare Input and Target
    features = basic_cols + combined_cols[1:]  # excluding magnitude_category
    target_column = "magnitude_category"

    df_model = df_model.dropna(subset=features + [target_column])

    # Drop classes with too few samples
    class_counts = df_model[target_column].value_counts()
    too_few = class_counts[class_counts < 2]
    if not too_few.empty:
        df_model = df_model[~df_model[target_column].isin(too_few.index)]
        print("Dropped classes with very few samples:", too_few.to_dict())

    X = df_model[features].values
    y = df_model[target_column].values

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # 4. Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
    )

    # 5. Normalization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    def explain_confusion_matrix(cm, labels):
        print("\nConfusion Matrix (detailed explanation):")
        for i, label in enumerate(labels):
            row = cm[i]
            total = sum(row)
            print(f"\nTrue class: {label} (Total: {total})")
            for j, val in enumerate(row):
                pred_label = labels[j]
                if i == j:
                    print(f"  → Predicted as {pred_label}: {val} ✅ Correct")
                else:
                    print(f"  → Predicted as {pred_label}: {val} ❌ Incorrect")

    def print_classification_metrics(name, y_true, y_pred_proba, label_encoder):
        y_pred = np.argmax(y_pred_proba, axis=1)
        print(f"\n--- {name} ---")
        print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))
        cm = confusion_matrix(y_true, y_pred)
        explain_confusion_matrix(cm, label_encoder.classes_)

    # === Models ===

    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_proba_rf = predict_proba(rf_model, X_test)
    print_classification_metrics("Random Forest", y_test, y_pred_proba_rf, le)

    # XGBoost
    xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5,
                              use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb_model.fit(X_train, y_train)
    y_pred_proba_xgb = predict_proba(xgb_model, X_test)
    print_classification_metrics("XGBoost", y_test, y_pred_proba_xgb, le)

    # Logistic Regression
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)
    y_pred_proba_lr = predict_proba(lr_model, X_test_scaled)
    print_classification_metrics("Logistic Regression", y_test, y_pred_proba_lr, le)

    # Neural Network
    tf.random.set_seed(42)
    nn_model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        Dense(32, activation='relu'),
        Dense(len(le.classes_), activation='softmax')
    ])
    nn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    nn_model.fit(
        X_train_scaled, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=0
    )

    y_pred_proba_nn = predict_proba(nn_model, X_test_scaled)
    print_classification_metrics("Neural Network", y_test, y_pred_proba_nn, le)

# اجرای مدل‌ها

if __name__ == "__main__":
    train_regression_model()
    train_classification_model()

