import os
import pickle
import numpy as np
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

data_path = '../data/vector/'
X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

search_space = {
        'max_depth': scope.int(hp.quniform('max_depth', 3, 10, 1)),
        'n_estimators': scope.int(hp.quniform('n_estimators', 50, 500, 50)),
        'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.15)),
        'reg_alpha': hp.loguniform('reg_alpha', np.log(1e-4), np.log(1.0)),
        'reg_lambda': hp.loguniform('reg_lambda', np.log(0.1), np.log(5)),
        'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
        'colsample_bytree': hp.quniform('colsample_bytree', 0.6, 0.9, 0.05),
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'random_state': 42
    }

score = []
def objective(params):
    xgb_model = xgb.XGBClassifier(early_stopping_rounds=50, **params)
    xgb_model.fit(X_train, y_train,
                 eval_set=[(X_val, y_val)],
                 verbose=False)
    y_pred_proba = xgb_model.predict_proba(X_val)[:, 1]
    y_pred = xgb_model.predict(X_val)
    auc = roc_auc_score(y_val, y_pred_proba)
    score.append({'params': params, 'auc': auc})
    
    return {'loss': -auc, 'status': STATUS_OK}

rstate = np.random.default_rng(42)
num_trials = 50
best_result = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=num_trials,
        trials=Trials(),
        rstate=rstate
    )

best_result_clean = {
    'max_depth': int(best_result['max_depth']),
    'n_estimators': int(best_result['n_estimators']),
    'min_child_weight': float(best_result['min_child_weight']),
    'colsample_bytree': float(best_result['colsample_bytree']),
    'learning_rate': float(best_result['learning_rate']),
    'reg_alpha': float(best_result['reg_alpha']),
    'reg_lambda': float(best_result['reg_lambda'])
}

vec = load_pickle(os.path.join(data_path, "dv.pkl"))
xgb_model = xgb.XGBClassifier(early_stopping_rounds=50, **best_result_clean)
xgb_model.fit(X_train, y_train, \
            eval_set=[(X_val, y_val)], \
            verbose=False)

pipeline = Pipeline([
    ('vectorizer', vec),
    ('xgb', xgb_model)
])

filename = 'xgb_pipeline.pkl'
with open(filename, 'wb') as file:
    pickle.dump(pipeline, file)

