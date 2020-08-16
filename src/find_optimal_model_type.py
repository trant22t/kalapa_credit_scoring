"""
This script experiments with different model types (Random Forest, XGBoost, LightGBM). For each model type, we train
the model using both the default parameters (of scikit-learn) and the best parameters found through cross-validation.
(Small note: for LightGBM cross-validation, we use the auto hyperparameter tuning feature of Optuna). In addition,
we output the final binary predictions based on both 0.5 threshold (default of scikit-learn) and Youden's index
threshold. In summary, there are 5 models trained and 10 sets of predictions.

Result:
- LightGBM (cv) is the best model, followed by XGBoost (cv).
- All models are saved in `model/`
- All test predictions are saved in `result/`
- Log file is saved in `logs/`
"""

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from collections import defaultdict

import pandas as pd
import numpy as np
import xgboost as xgb
import optuna.integration.lightgbm as lgb
import copy
import pickle
import time

from src.util import log, get_youden_thres, get_gini, output_submission

# 1. Prepare modeling data
train = pd.read_pickle('data/intermediate/train2.pkl')
X_train = train.drop('label', axis=1)
y_train = train['label'].astype('category')

X_test = pd.read_pickle('data/intermediate/test1.pkl')
test_ids = pd.read_csv('data/raw/test.csv')['id']

train_x, val_x, train_y, val_y = train_test_split(X_train, y_train.astype(int), test_size=0.20)
d_train = lgb.Dataset(train_x, label=train_y)
d_val = lgb.Dataset(val_x, label=val_y)

# 2. Fit model
# 2.1. Candidate models
models = dict()
models['rf_default'] = RandomForestClassifier(n_estimators=500, max_depth=20, random_state=42)
rf_grid = {'n_estimators': [500, 1000],
           'max_depth': [2, 4, 6],
           'min_samples_split': [10, 20, 40],
           'min_samples_leaf': [10, 20, 40]}
models['rf_cv'] = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=rf_grid,
                               cv=3, verbose=2, n_jobs=2, scoring='roc_auc')

models['xb_default'] = xgb.XGBClassifier(random_state=42, n_estimators=250, max_depth=20)
xgb_grid = {'learning_rate': [0.001, 0.01, 0.1],
            'n_estimators': [100, 500],
            'max_depth': [2, 4, 6],
            'subsample': [0.8, 1]}
models['xb_cv'] = GridSearchCV(estimator=xgb.XGBClassifier(random_state=42), param_grid=xgb_grid,
                               cv=3, verbose=2, n_jobs=2, scoring='roc_auc')

lgbm_params = {'objective': 'binary',
               'metric': 'binary_logloss',
               'verbosity': -1,
               'boosting_type': 'gbdt'
               }


# 2.2. Loop through model types to train
model_direc = 'model/'
# model_types = ['rf_default', 'rf_cv', 'xb_default', 'xb_cv', 'lgbm_cv']
model_types = ['xb_cv', 'lgbm_cv']

my_logger = log(path='logs/', file='find_optimal_model_type.logs')
my_logger.info('Start model training process....')

for model_type in model_types:
    my_logger.info('Train {}'.format(model_type))
    start = time.time()

    fn = model_direc + '{}.pkl'.format(model_type)

    if model_type != 'lgbm_cv':
        md = copy.deepcopy(models[model_type])
        md.fit(X_train, y_train)
        with open(fn, 'wb') as f:
            pickle.dump(md, f, pickle.HIGHEST_PROTOCOL)
    else:
        models[model_type] = lgb.train(lgbm_params, d_train, valid_sets=[d_train, d_val], verbose_eval=100,
                                       early_stopping_rounds=100)
        with open(fn, 'wb') as f:
            pickle.dump(models[model_type], f, pickle.HIGHEST_PROTOCOL)

    my_logger.info('Save {}'.format(model_type))
    my_logger.info('Elapsed time: {} seconds'.format(round(time.time() - start), 2))
    my_logger.info('--------------------------------------------')


# 3. Analyze results
res_direc = 'result/'
ypred_types = ['sklearn_binary', 'sklearn_prob', 'youden_binary']
train_preds = defaultdict()
test_preds = defaultdict()
train_scores = defaultdict()

my_logger.info('Start model analyzing process....')

for model_type in model_types:
    fn = model_direc + '{}.pkl'.format(model_type)
    with open(fn, 'rb') as f:
        model = pickle.load(f)

    train_preds[model_type] = dict()
    test_preds[model_type] = dict()
    train_scores[model_type] = dict()

    # 3.1. For train set
    if model_type != 'lgbm_cv':
        train_skl_prob = model.predict_proba(X_train)[:, 1]
        train_preds[model_type]['sklearn_prob'] = train_skl_prob
        train_preds[model_type]['sklearn_binary'] = model.predict(X_train)
    else:
        model = copy.deepcopy(models[model_type])
        train_skl_prob = model.predict(X_train, num_iteration=model.best_iteration)
        train_preds[model_type]['sklearn_prob'] = train_skl_prob
        train_preds[model_type]['sklearn_binary'] = np.rint(train_skl_prob)

    train_youden_thres = get_youden_thres(y_train, train_skl_prob)
    train_preds[model_type]['youden_binary'] = np.where(train_skl_prob >= train_youden_thres, 1, 0)

    for ypred_type in ypred_types:
        pred = train_preds[model_type][ypred_type]
        auc = roc_auc_score(y_train, pred)
        gini = get_gini(y_train, pred)

        my_logger.info('AUC of {}_{} on train: {}'.format(model_type, ypred_type, auc))
        my_logger.info('Gini of {}_{} on train: {}'.format(model_type, ypred_type, gini))
        my_logger.info('--------------------------------------------')

        train_scores[model_type][ypred_type] = [auc, gini]

    # 3.2. For test set
    if model_type != 'lgbm_cv':
        test_skl_prob = model.predict_proba(X_test)[:, 1]
        test_skl_binary = model.predict(X_test)
    else:
        model = copy.deepcopy(models[model_type])
        test_skl_prob = model.predict(X_test, num_iteration=model.best_iteration)
        test_skl_binary = np.rint(test_skl_prob)

    output_submission(test_ids, res_direc, model_type, test_skl_binary, test_skl_prob, train_youden_thres)

train_res = pd.DataFrame.from_records(
    [(k, sub_k, sub_v[0], sub_v[1]) for k, v in train_scores.items() for sub_k, sub_v in v.items()],
    columns=['model', 'pred_type', 'auc', 'gini']
)
