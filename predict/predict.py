import os
import pickle
import xgboost
import pandas as pd

def load_model(model_path):
    try:
        with open(model_path, 'rb') as file:
            model_pipeline = pickle.load(file)

        print("Pipeline successfully loaded.")
    
    except FileNotFoundError:
        print(f"Error: The file '{model_path}' was not found.")
    except Exception as e:
        print(f"An error occurred while loading the pipeline: {e}")
    return model_pipeline

def prepare_feature(raw_data):
    categorical = ['date', 'dayofweek', 'away_league', 'home_league', 'park_id']
    numerical = ['home_OPS_blend', 'home_FIP_blend', 'home_FPCT_blend', \
        'away_OPS_blend', 'away_FIP_blend', 'away_FPCT_blend']
    df_features = raw_data[categorical + numerical]
    dicts = df_features.to_dict(orient='records')
    return dicts

def predict_single(raw_data):
    model_path = 'xgb_pipeline.pkl'
    print("cwd:", os.getcwd())
    print("exists?", os.path.exists(model_path))
    model_pipeline = load_model(model_path)
    dicts = prepare_feature(raw_data)
    prediction = model_pipeline.predict_proba(dicts)[:,1]
    pred_win = prediction > 0.7
    pred_lose = prediction < 0.3
    result = {
        'win_prob': prediction,
        'predicted_win': pred_win,
        'predicted_lose': pred_lose
    }

    return result

if __name__ == "__main__":
    new_data = pd.DataFrame([
        {
            'date': 20250320,
            'dayofweek': 'Thu',
            'away_league': 'NL',
            'home_league': 'NL',
            'park_id': 'CHI12',
            'home_OPS_blend': 0.85,
            'home_FIP_blend': 3.9,
            'home_FPCT_blend': 0.976,
            'away_OPS_blend': 0.78,
            'away_FIP_blend': 4.9,
            'away_FPCT_blend': 0.985
        }
    ])
    prediction = predict_single(new_data)
    print('data: ', new_data, 'prediction: ', prediction)