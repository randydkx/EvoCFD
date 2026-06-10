import os
import numpy as np
import pandas as pd
# from .treemodel.LGBM import *
# from .treemodel.CatB import *
# from .treemodel.XGB import *
from model.dlmodel.model.utils import get_method

THIS_PATH = os.path.dirname(__file__)

models = ['EvoCFD']
indices_models = ['EvoCFD']
tabr_ohe_models = []

def log_result(message, dataset, stage):
    filename = f"{dataset}_{stage}.txt"
    print(message.strip())
    with open(filename, 'a') as f:
        f.write(message)


def deep_learning(args, stage_data, stage_index, logger):
    summary = stage_data.summary
    dataset_task = summary.get('task')
    if dataset_task is None:
        raise ValueError("Summary metadata missing 'task'; cannot determine training objective.")
    if dataset_task != 'binary':
        raise ValueError(f"Only binary classification is supported now, got task={dataset_task}.")
    method = get_method(args.model_type)(args, stage_data=stage_data)
    # argparse.Namespace stores attributes, so use dot notation instead of dict-style lookup
    _, best_recall = method.fit(stage_index, train=args.train, logger=logger, sys_args=args)
    method.calculate_and_log_gain_rate_etc()
    return best_recall
    
