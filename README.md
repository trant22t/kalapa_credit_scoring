# Kalapa Credit Scoring Challenge
## Background and Data
As part of their Credit Scoring Challenge, Kalapa provides us with de-identified information of 53,030 consumers along 
with their binary credit worthiness labels (good/bad). Using this dataset, we need to train a model to predict credit scores
of the remaining 20,381 consumers provided with no labels. 

## Folder structure
```
├── README.md                           <- The top-level README for developers using this project.
├── data                                <- Folder that includes original data as well as intermediate and processed data.
│   ├── intermediate
│   │   ├── before_impute.pkl
│   │   ├── impute_scores.pkl
│   │   ├── test1.pkl
│   │   ├── train1.pkl
│   │   ├── train2.pkl
│   │   └── tree_imputer.pkl
│   └── raw
│       ├── test.csv
│       └── train.csv
├── requirements.txt                    <- Text file that lists libraries required to run notebook and Python scripts.
└── src
    ├── __init__.py
    ├── build_neural_net.ipynb          <- Notebook that builds out the neural network model (originally a Google Colab)
    ├── explore_imputation.py           <- Script that explores various imputation methods
    ├── find_optimal_model_type.py      <- Script that experiments with different model types (RF, XGBoost, LightGBM, etc.)
    ├── find_optimal_oversampling.py    <- Script that attempts to find the best combination of model type and oversampling strategy
    ├── process_data.py                 <- Entire data processing procedure
    └── util.py                         <- Utility functions used in various modeling scripts
```

## Approaches and Results
### Data Preparation
The provided raw data includes 195 features of various mixed types and ambiguous column names that seem to come from 
heterogenous sources. Considering the fact that high-quality data leads to better models and predictions, we take some 
fundamental steps to process and clean data. 

#### Integration
- Feature engineering: create new attributes based on the given set of attributes (e.g.: age, gender, location, 
differences between dates, etc.)
- Concept hierarchy generation for categorical data: values for categorical attributes are generalized to higher-order 
concepts (e.g.: job category)

#### Cleaning
Our imputation approach to deal with missing data is as follows: 
- For categorical attributes:
    - If they have only one unique level, create missing indicators accordingly and remove the raw attributes
    - If they have more than one unique levels, replace missing values with a new level, named `missing`
- For numerical attributes: 
    - We experiment with different imputation methods, such as filling in with 0, filling in with mean of entire column, 
    filling in with mean of k-nearest neighbors or using algorithms like Bayesian ridge regression and decision trees.
    - The best strategy selected is to model each feature with missing values as a function of other features in a round-robin
    fashion using an extra-trees regressor. 

#### Reduction
Since data is collected from multiple sources, which may lead to redundant and inconsistent information, in order to 
have a condensed representation of the data set which is smaller in volume, while maintaining the integrity of original, 
we take the following steps:
- Drop records that have 90% missing values 
- Drop all categorical attributes that only have one unique level
- Drop attributes whose correlation coefficient with another is higher than 0.90 
- Drop raw attributes whose information has been encoded in other newly created attributes during feature engineering process
- Drop meta attributes such as id, longitudes, latitudes, etc. and attributes with cryptic meanings and more than 10k levels.

#### Transformation
To ensure that our attributes are in appropriate forms for modeling, we: 
- Dummy encode all remaining categorical attributes
- Expand date attributes (yyyy-mm-dd) into separate year and month columns. 

#### Feature selection
After all aforementioned processes, we end up with 616 columns, to select a subset of relevant features for use in model 
construction, we employ two-sample Kolmogorov-Smirnov Test (K-S Test). This test is used to determine whether the distribution
of a feature is the same across two samples (i.e. positively-labeled samples and negatively-labeled samples). Features with 
the larger differences between two samples have higher K-S scores, thus higher ranks and lower p-values. In the end, we only 
keep features whose p-values are smaller than 0.001, which results to a set of 186 attributes. 

### Modeling

#### Classifiers, Hyperparameter Tuning, and Oversampling Techniques
The main model is formulated as a binary classification problem, in which the target variable can take the value of either 
0 (good) or 1 (bad). We experiment with random forest, XGBoost, Light GBM and multilayer perceptron. For all model types, we
both use the Python built-in default parameters and search through customized grid spaces for optimal hyperparameters. 
LightGBM with cross-validated hyperparameters achieves the best performance, followed by XGBoost.

We also attempt to find the best combination of model type and oversampling strategy to address the unbalanced data issue.
We experiment with 2 best classifiers, i.e. XGBoost and LightGBM with their optimal hyperparameters found through cross-validation
and a range of values for the number of neighbors `k` used in SMOTE oversampling technique. LightGBM (cv) performs better 
than XGBoost (cv) for all values of `k`. However, within each model type, `k` doesn't affect the model performance too much. 
On top of that, test performance of model with SMOTE is actually worse than that without SMOTE. Our theory for this is 
that even though our data is unbalanced, the ratio between minority class and majority class is 1:2, which means the unbalanced
problem is not severe and the small difference between number of instances among two classes can be safely ignored. Thus, 
our final model is trained without employing SMOTE. Instead, to alleviate the imbalance, we derive an optimal cut-off point
between positive and negative labels, rather than using the naive 0.5 threshold - this will be further discussed in the 
next section.

#### Evaluation Metrics
- Youden index threshold: Since the data set is not balanced, using a naive threshold of 0.5 to separate between positive 
and negative classes might not be preferable and accurate. Instead, we derive the optimal cut-off point from Youden's index.
The index is based on the idea of maximizing the true positive rate (TPR) while minimizing the false positive rate (FPR).
The threshold is found when TPR - (1 - FPR) = 0. If both the TP and FP lines are shown in the same chart, Youden's index 
is the intersection between these 2 lines. 
- AUC ROC: This metric represents the area under the ROC curve. The curve evaluates the false positive rate against the 
true positive rate. The higher the value, the better the model is at distinguishing between two classes.
- Gini score: In the consumer finance industry, Gini score tends to be used more often than AUC. We can calculate this 
metric based on AUC, i.e.: Gini score = AUC score * 2 - 1.

 
