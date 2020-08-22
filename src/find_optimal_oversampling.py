"""
This script attempts to find the best combination of model type and oversampling strategy. We start with 2 best model
types found in `src/find_optimal_model_type.py` (XGBoost and LightGBM - whose parameters are found through cross-validation)
and a range of values for the number of neighbors `k` (a parameter used in SMOTE oversampling technique).

Result:
- LightGBM (cv) performs better than XGBoost (cv) for all values of `k`.
- Within each model type, `k` doesn't affect the model performance too much, although the higher the `k`, the better
the model's AUC. Again, the model improvement by `k` is very small.
"""

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from src.util import log, get_youden_thres, output_submission

import pandas as pd
import numpy as np
import xgboost as xgb
import pickle

# 1. Set up
# 1.1. Prepare training data
train = pd.read_pickle('data/intermediate/train2.pkl')
X_train = train.drop('label', axis=1)
y_train = train['label'].astype('category')

X_test = pd.read_pickle('data/intermediate/test1.pkl')
test_ids = pd.read_csv('data/raw/test.csv')['id']

# 1.2. Prepare pre-trained model (from `src/find_optimal_model_type.py`)
with open('model/xb_cv.pkl', 'rb') as f:
    xb_cv = pickle.load(f)
xb_params = xb_cv.best_params_
xb_model = xgb.XGBClassifier(n_estimators=xb_params['n_estimators'], max_depth=xb_params['max_depth'],
                             learning_rate=xb_params['learning_rate'], subsample=xb_params['subsample'],
                             random_state=42)

with open('model/lgbm_cv.pkl', 'rb') as f:
    lgbm_cv = pickle.load(f)
lgbm_params = lgbm_cv.params
lgbm_model = LGBMClassifier(objective=lgbm_params['objective'], metric=lgbm_params['metric'],
                            boosting_type=lgbm_params['boosting_type'], reg_alpha=lgbm_params['lambda_l1'],
                            reg_lambda=lgbm_params['lambda_l2'], num_leaves=lgbm_params['num_leaves'],
                            subsample=lgbm_params['bagging_fraction'], subsample_freq=lgbm_params['bagging_freq'],
                            colsample_bytree=lgbm_params['feature_fraction'], random_state=42,
                            min_child_samples=lgbm_params['min_child_samples'])

# 2. Loop through different model types and SMOTE hyperparameter values
my_logger = log(path='logs/', file='find_optimal_oversampling.logs')
my_logger.info('Start cross-validating process....')

for classifier in [xb_model, lgbm_model]:
    my_logger.info('Classifier: {}'.format(str(classifier).split('(')[0]))

    for k in [1, 2, 3, 5, 7]:
        my_logger.info('k: {}'.format(k))

        pipe = Pipeline(steps=[
            ('scale', StandardScaler()),
            ('over', SMOTE(sampling_strategy='auto', k_neighbors=k, random_state=42)),
            ('under', RandomUnderSampler(sampling_strategy='auto', random_state=42)),
            ('model', classifier)
        ])
        cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=3, random_state=42)
        scores = cross_val_score(pipe, X_train, y_train, scoring='roc_auc', cv=cv, n_jobs=2)
        score = np.mean(scores)

        my_logger.info('Mean AUC: {}'.format(score))
        my_logger.info('--------------------------------------------')

# 3. Define final modeling pipeline (with the best model type and k value found in #2), fit and output results
final_pipe = Pipeline(steps=[
            ('scale', StandardScaler()),
            ('over', SMOTE(sampling_strategy='auto', k_neighbors=7, random_state=42)),
            ('under', RandomUnderSampler(sampling_strategy='auto', random_state=42)),
            ('model', lgbm_model)
        ])
final_pipe.fit(X_train, y_train)

train_prob = final_pipe.predict_proba(X_train)[:, 1]
train_youden_thres = get_youden_thres(y_train, train_prob)

test_binary = final_pipe.predict(X_test)
test_prob = final_pipe.predict_proba(X_test)[:, 1]

output_submission(test_ids, 'result/', 'lgbm_cv_smote', test_binary, test_prob, train_youden_thres)



