from trainer.basic_trainer import Trainer
from trainer.multitask_trainer import MultiTaskTrainer, DSMultiTaskTrainer
from trainer.utils import CheckpointMeasurement
from trainer.generator import Generator


def get_trainer(args) -> Trainer:
    if args.trainer == 'basic':
        return MultiTaskTrainer
    elif args.trainer == 'deepspeed':
        return DSMultiTaskTrainer
    raise NotImplementedError(f"No Trainer: {args.trainer}")
