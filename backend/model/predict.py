import numpy as np
import xgboost as xgb
from utils.feature_engineering import expand_features

def classify_rul(rul):
    if rul > 80:
        return "Healthy"
    elif rul > 30:
        return "At Risk"
    else:
        return "Critical"


def predict_rul(models, dataset_type, input_data):

    print("Input length:", len(input_data))

    model = models[f"model_{dataset_type}"]
    scaler = models[f"scaler_{dataset_type}"]

    X = expand_features(input_data)

    print("Before scaling:", X)

    

    try:
        
        if dataset_type == "fd004" and isinstance(scaler, dict):
            print("Using condition-based scaling for FD004")

            kmeans = models.get("kmeans_fd004")

            if kmeans is None:
                print("KMeans model missing, skipping scaling")
                X_scaled = X
            else:
                settings = np.array(input_data[:3]).reshape(1, -1)

                cond = kmeans.predict(settings)[0]
                print("Predicted condition:", cond)

                if cond in scaler:
                    selected_scaler = scaler[cond]
                    X_scaled = selected_scaler.transform(X)
                else:
                    print("Condition not found, skipping scaling")
                    X_scaled = X

     
        elif hasattr(scaler, "transform"):
            X_scaled = scaler.transform(X)

        
        else:
            print("Invalid scaler, skipping scaling")
            X_scaled = X

        print("After scaling:", X_scaled)

    except Exception as e:
        print("Scaler ERROR:", e)
        return {"error": f"Scaler error: {str(e)}"}

    try:
        if isinstance(model, xgb.Booster):
            dmatrix = xgb.DMatrix(X_scaled)
            prediction = model.predict(dmatrix)[0]
        else:
            prediction = model.predict(X_scaled)[0]

        print("Prediction:", prediction)

    except Exception as e:
        print("Model ERROR:", e)
        return {"error": f"Model error: {str(e)}"}

    status = classify_rul(prediction)

    return {
        "predicted_rul": float(prediction),
        "status": status
    }