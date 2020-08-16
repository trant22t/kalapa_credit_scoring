"""
This script experiments with different imputation methods to select the best technique for the training set at hand.

Ref1: https://scikit-learn.org/stable/auto_examples/impute/plot_iterative_imputer_variants_comparison.html#sphx-glr-auto-examples-impute-plot-iterative-imputer-variants-comparison-py
Ref2: https://scikit-learn.org/stable/auto_examples/impute/plot_missing_values.html#sphx-glr-auto-examples-impute-plot-missing-values-py

Final result: IterativeImputer with ExtraTreeRegressor is the best.

"""

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# 1. Set up
# 1.1. Helper function
def get_scores_for_imputer(imputer, X_missing, y_missing):
    """Estimate the cross validation score for each imputer"""
    estimator = make_pipeline(imputer, RandomForestRegressor(random_state=42))
    impute_scores = cross_val_score(estimator, X_missing, y_missing, scoring='neg_mean_squared_error', cv=5)
    return impute_scores


# 1.2. Prepare data
train1 = pd.read_pickle('KalapaCreditScoring/data/intermediate/before_impute.pkl')
numeric_df = train1.select_dtypes('number')
random_samp = numeric_df.sample(n=10000, random_state=42)
X = random_samp.drop('label', axis=1)
y = random_samp['label']


# 2. Fit various imputers
# 2.1 SingleImputer (constant and mean strategies)
score_simple_imputer = pd.DataFrame()
for strategy in ('constant', 'mean'):
    simple_imputer = SimpleImputer(missing_values=np.nan, strategy=strategy)
    score_simple_imputer[strategy] = get_scores_for_imputer(simple_imputer, X, y)

# 2.2. KNNImputer
knn_imputer = KNNImputer(missing_values=np.nan)
score_knn_imputer = pd.DataFrame(get_scores_for_imputer(knn_imputer, X, y), columns=['KNNImputer'])

# 2.3. IterativeImputer
iterative_estimators = (
    BayesianRidge(),
    ExtraTreesRegressor(n_estimators=10, random_state=42)
)
score_iterative_imputer = pd.DataFrame()
for impute_estimator in iterative_estimators:
    iterative_imputer = IterativeImputer(random_state=42, estimator=impute_estimator)
    score_iterative_imputer[impute_estimator.__class__.__name__] = get_scores_for_imputer(iterative_imputer, X, y)


# 3. Analyze and plot results
scores = pd.concat([score_simple_imputer, score_knn_imputer, score_iterative_imputer],
                   keys=['SimpleImputer', 'KNNImputer', 'IterativeImputer'], axis=1)
scores.to_pickle('KalapaCreditScoring/data/intermediate/impute_scores.pkl')

fig, ax = plt.subplots(figsize=(13, 6))
means = -scores.mean()
errors = scores.std()
means.plot.barh(xerr=errors, ax=ax)
ax.set_title('Performance of Different Imputation Methods')
ax.set_xlabel('MSE (smaller is better)')
ax.set_yticks(np.arange(means.shape[0]))
ax.set_yticklabels([' w/ '.join(key) for key in means.index.tolist()])
plt.tight_layout(pad=1)
plt.show()

