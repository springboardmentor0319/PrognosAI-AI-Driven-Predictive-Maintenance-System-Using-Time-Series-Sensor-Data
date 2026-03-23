import os
import joblib
import xgboost as xgb

def load_all_models():
    base_path = os.path.dirname(os.path.dirname(__file__))
    model_dir = os.path.join(base_path, "models")

    models = {}

    for file in os.listdir(model_dir):
        file_path = os.path.join(model_dir, file)

       
        if file.endswith(".json"):
            name = file.replace(".json", "")
            model = xgb.Booster()
            model.load_model(file_path)
            models[name] = model

        
        elif file.endswith(".pkl"):
            name = file.replace(".pkl", "")
            obj = joblib.load(file_path)
            models[name] = obj

    return models