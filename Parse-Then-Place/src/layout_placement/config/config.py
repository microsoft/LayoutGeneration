import argparse
import os


def is_true(b):
    return str(b).lower() in ('true', '1', 'yes')

class Config:

    def __init__(self):
        self.args = self.build_config()

    def build_config(self):
        parser = argparse.ArgumentParser()

        # ----------training args------------
        parser.add_argument("--epochs", default=50, type=int)
        parser.add_argument("--batch_size", default=8, type=int)
        parser.add_argument("--lr", default=1e-4, type=float)
        parser.add_argument("--weight_decay", default=0, type=float)
        parser.add_argument("--seed", default=42, type=int)
        parser.add_argument("--warmup_ratio", default=0.1, type=float)
        parser.add_argument("--max_grad_norm", default=1.0, type=float)
        parser.add_argument("--distribute", default="true", type=is_true, help="whether or not to use distributed training")
        parser.add_argument("--local_rank", default=-1, type=int, help="rank")
        parser.add_argument("--evaluation_interval", default=1, type=int)
        parser.add_argument("--num_train", type=int)
        parser.add_argument("--is_two_stage", default="true", type=is_true)
        parser.add_argument("--add_complete_token", default="true", type=is_true)
        parser.add_argument("--use_wandb", default="true", type=is_true)
        parser.add_argument("--scheduler", default="warmuplinear", type=str)
        parser.add_argument("--run_description", type=str)

        # -----------model args---------------
        parser.add_argument("--model_name_or_path", default="t5-base",
                            type=str, help="the backbone for training and inference")
        # parser.add_argument("--output_dir", default="save_model", type=str)
        parser.add_argument("--tokenizer_path", default="t5-base",
                            type=str, help="tokenizer directory")
        parser.add_argument("--config", type=str, default="new_config",
                            help="config for model to train from scratch")

        # -----------data args----------------
        parser.add_argument("--shuffle", default="true", type=is_true, help="shuffle or not when training")
        parser.add_argument("--num_workers", default=8, type=int)
        parser.add_argument("--dataset_name", default='web', type=str)

        parser.add_argument("--train_source_file", default="train_source.txt", type=str)
        parser.add_argument("--train_target_file", default="train_target.txt", type=str)
        parser.add_argument("--val_source_file", default="val_source.txt", type=str)
        parser.add_argument("--val_target_file", default="val_target.txt", type=str)

        parser.add_argument("--test_ground_truth_file", default="test.json", type=str)
        parser.add_argument("--stage_one_prediction_file", type=str, default="prediction.json")
        parser.add_argument("--test_prediction_ir", default="true", type=is_true)

        # -----------IO args---------------
        parser.add_argument("--data_dir", type=str, default=os.getenv('AMLT_DATA_DIR', './datasets/'), help="data dir")
        parser.add_argument("--out_dir", type=str, default=os.getenv('AMLT_OUTPUT_DIR', './output/'), help="output directory")

        # -----------sampling---------------
        parser.add_argument("--temperature", type=float, default=0.7)
        parser.add_argument("--num_return_sequences", type=int, default=5)

        args = parser.parse_args()
        return args