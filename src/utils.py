import sys
import os
import pickle
from src.exception import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    report = {}
    fitted_models = {}
    try:
      for name, model in models.items():
        params = param.get(name, {})
        
        # Use GridSearchCV for hyperparameter tuning if parameters are provided
        if params:
            gs = GridSearchCV(model, params, cv=3, scoring="r2", n_jobs=-1)
            gs.fit(X_train, y_train)
            best_model = gs.best_estimator_
        else:
            # Directly fit the model if no params are provided
            model.fit(X_train, y_train)
            best_model = model

        # Store the fitted model
        fitted_models[name] = best_model

        # Evaluate on test data
        y_pred = best_model.predict(X_test)
        report[name] = r2_score(y_test, y_pred)

        return report, fitted_models
    except Exception as e:
      raise CustomException(e, sys)