import os

from evaluation.metrics import compute_metrics
from layout_placement.config.config import Config
from layout_placement.placement_utils.utils import *

if __name__ == "__main__":
    config = Config()
    set_seed(config.args.seed)
    predictions = read_json(os.path.join(config.args.out_dir, 'prediction.json'))
    pred_list = [pred['prediction'] for pred in predictions]
    golds = read_json(os.path.join(config.args.data_dir, config.args.test_ground_truth_file))
    training_golds = read_json(os.path.join(config.args.data_dir, config.args.train_source_file))
    _, metrics = compute_metrics(pred_list, golds, training_golds, config.args.dataset_name)
    write_json(metrics, os.path.join(config.args.out_dir, 'metrics.json'))