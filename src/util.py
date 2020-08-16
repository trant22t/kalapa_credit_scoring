"""
Utility functions used in various modeling scripts.
"""

from sklearn.metrics import roc_auc_score, roc_curve
import pandas as pd
import numpy as np
import logging
import os


def log(path, file):
    """
    Create a log file to record the experiment's logs
    :param path: path to the directory where log file is saved
    :param file: file name
    :return: logger object that record logs
    """
    log_file = os.path.join(path, file)
    if not os.path.isfile(log_file):
        open(log_file, 'w+').close()

    console_logging_format = '%(levelname)s %(message)s'
    file_logging_format = '%(levelname)s: %(asctime)s: %(message)s'

    logging.basicConfig(level=logging.INFO, format=console_logging_format)
    logger = logging.getLogger()
    formatter = logging.Formatter(file_logging_format)
    handler = logging.FileHandler(log_file)

    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def get_gini(y_true, y_pred):
    """
    Compute Gini score of model from predictions
    :param y_true: true labels
    :param y_pred: target scores, can be either probability estimates or binary decision values
    :return: Gini score
    """
    return roc_auc_score(y_true, y_pred)*2 - 1


def get_youden_thres(y_true, y_prob):
    """
    Compute the threshold that separates postive and negative classes based on Youden's J statistic.
    J = sensitivity + specificity - 1 = TPR + (1 - FPR) - 1 = TPR - FPR.
    :param y_true: true labels
    :param y_prob: probability estimates of positive class
    :return: separating threshold based on Youden's index
    """
    fpr, tpr, thres = roc_curve(y_true, y_prob)
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'fpr': pd.Series(fpr, index=i),
                        'tpr': pd.Series(tpr, index=i),
                        '1-fpr': pd.Series(1 - fpr, index=i),
                        'youden': pd.Series(tpr - fpr, index=i),
                        'threshold': pd.Series(thres, index=i)})
    opt_pt = roc.iloc[roc['youden'].idxmax(), :]
    return opt_pt['threshold']


def output_submission(id_list, res_direc, model_type, sklearn_binary, sklearn_prob, youden_thres):
    """
    Output the test predictions as a csv file in the right format to be submitted
    :param id_list: list of ids in the test set
    :param res_direc: path to directory to save file
    :param model_type: name of model type that outputs the predictions
    :param sklearn_binary: predictions based on 0.5 threshold
    :param sklearn_prob: probability estimates of predictions
    :param youden_thres: threshold found from Youden index
    :return: None, but output files are saved in result directory
    """
    test_skl_df = pd.concat([id_list, pd.DataFrame(sklearn_binary, columns=['label'])], axis=1)
    fn1 = res_direc + '{}_{}.csv'.format(model_type, 'sklearn_binary')
    test_skl_df.to_csv(fn1, index=False)

    test_youden_binary = np.where(sklearn_prob >= youden_thres, 1, 0)
    test_youden_df = pd.concat([id_list, pd.DataFrame(test_youden_binary, columns=['label'])], axis=1)
    fn2 = res_direc + '{}_{}.csv'.format(model_type, 'youden_binary')
    test_youden_df.to_csv(fn2, index=False)

    return None
