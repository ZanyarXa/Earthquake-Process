# src/model_utils.py

import numpy as np

def predict_proba(model, X_test):
    if hasattr(model, "predict_proba"):
        if len(model.classes_) == 2:
            return model.predict_proba(X_test)[:, 1]
        else:
            if model.classes_[0] == 1:
                return np.ones(len(X_test))
            else:
                return np.zeros(len(X_test))
    else:
        return model.predict(X_test).flatten()
