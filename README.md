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

## Approach and Conclusion
(more to come)