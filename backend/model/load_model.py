import os
import joblib
import xgboost as xgb

def load_all_models():
    base_path = os.path.dirname(os.path.dirname(__file__))
    model_dir = os.path.join(base_path, "models")

    models = {}

    for file in os.listdir(model_dir):
        file_path = os.path.join(model_dir, file)

       
        dataset = file.lower().split("_")[-1].replace(".pkl", "").replace(".json", "")

        # ----------- KMEANS -----------
        if "kmeans" in file.lower():
            kmeans = joblib.load(file_path)
            print(f"{file} → {type(kmeans)}")
            models[f"kmeans_{dataset}"] = kmeans

        # ----------- SCALER -----------
        elif "scaler" in file.lower():
            scaler = joblib.load(file_path)
            print(f"{file} → {type(scaler)}")
            models[f"scaler_{dataset}"] = scaler

        # ----------- XGBOOST JSON -----------
        elif file.endswith(".json"):
            model = xgb.Booster()
            model.load_model(file_path)
            print(f"{file} → Booster loaded")
            models[f"model_{dataset}"] = model

        # ----------- PKL MODEL -----------
        elif file.endswith(".pkl"):
            obj = joblib.load(file_path)
            print(f"{file} → {type(obj)}")

            if "scaler" not in file.lower():
                models[f"model_{dataset}"] = obj

    return models