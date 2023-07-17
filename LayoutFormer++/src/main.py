import argparse
import os

from torch.optim import Adam
from deepspeed.runtime.lr_schedules import WarmupLR

from utils import utils, config
from tasks.task_utils import T5LayoutSequence, TASK_CONFIG, \
    create_tokenizer, build_model, create_dataset, \
    create_fid_model, \
    TrainFn, EvaluateFn
from tasks.gen_r import RelationTypes
from trainer import get_trainer, CheckpointMeasurement, Generator
from model.layout_transformer import constrained_decoding


def train(args):
    task_config = {
        'tasks': args.tasks,
        'task_sample_weights': args.task_weights,
        'task_loss_weights': args.task_loss_weights,
        'gaussian_noise_mean': args.gaussian_noise_mean,
        'gaussian_noise_std': args.gaussian_noise_std,
        'bernoulli_beta': args.train_bernoulli_beta,
        'add_sep_token': args.add_sep_token,
        'add_task_prompt': args.add_task_prompt,
        'num_layers': args.num_layers,
        'num_head': args.nhead,
        'd_model': args.d_model,
        'dropout': args.dropout,
        'share_embedding': args.share_embedding,
        'model': 'transformer',
        'sort_by_dict': args.sort_by_dict,
        'partition_training_data': args.partition_training_data,
        'partition_training_data_task_buckets': args.partition_training_data_task_buckets,
        'single_task_per_batch': args.single_task_per_batch,
        'add_task_embedding': args.add_task_embedding,
        'add_task_prompt_token_in_model': args.add_task_prompt_token_in_model
    }
    print(f"Training on tasks: {args.tasks}")
    print(task_config)
    tasks = args.tasks.split(",")
    tn = 'multitask' if len(tasks) > 1 else args.tasks

    tokenizer = create_tokenizer(tasks, args.dataset, args.discrete_x_grid,
                                 add_sep_token=args.add_sep_token,
                                 sep_token=T5LayoutSequence.SEP_TOKEN,
                                 add_task_prompt=args.add_task_prompt)
    model = build_model(args, tokenizer)
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = WarmupLR(optimizer, warmup_max_lr=args.lr,
                         warmup_num_steps=args.warmup_num_steps)

    train_dataset = create_dataset(args, tokenizer=tokenizer, split='train',
                                   task_config=TASK_CONFIG, sort_by_pos=not args.sort_by_dict)
    val_dataset = create_dataset(args, tokenizer=tokenizer, split='val',
                                 task_config=TASK_CONFIG, sort_by_pos=not args.sort_by_dict)

    print(f'train data: {len(train_dataset)}')
    print(f'valid data: {len(val_dataset)}')

    trainer_class = get_trainer(args)
    trainer = trainer_class(tn, args, tokenizer, model, val_dataset.seq_processor,
                            train_dataset=train_dataset, val_dataset=val_dataset,
                            optimizer=optimizer, scheduler=scheduler, is_label_condition=False,
                            checkpoint_measure=CheckpointMeasurement.EVAL_LOSS,
                            task_config=task_config, save_vocab=True)

    task_loss_weights = None
    if args.task_loss_weights:
        task_loss_weights = dict()
        weights = list(map(float, args.task_loss_weights.split(",")))
        assert len(weights) == len(tasks)
        task_loss_weights = {tn: w for tn, w in zip(tasks, weights)}
    print("Task Loss Weights: ", task_loss_weights)
    train_fn = TrainFn(task_loss_weights)
    evaluate_fn = EvaluateFn(args.max_num_elements, enable_task_measure=args.enable_task_measure,
                             decode_max_length=args.decode_max_length, topk=args.topk,
                             temperature=args.temperature)

    trainer(train_fn, evaluate_fn, tasks=train_dataset.tasks,
            eval_interval=args.eval_interval)
    trainer.clean_up()


def inference(args):
    utils.set_seed(args.eval_seed)
    tasks = args.tasks.split(",")
    tokenizer = create_tokenizer(tasks, args.dataset, max(args.discrete_x_grid, args.discrete_y_grid),
                                 add_sep_token=args.add_sep_token,
                                 sep_token=T5LayoutSequence.SEP_TOKEN,
                                 add_task_prompt=args.add_task_prompt)
    if args.load_vocab:
        print("Loading Vocabulary from vocab.json")
        tokenizer.from_vocab(os.path.join(args.out_dir, "vocab.json"))
    model = build_model(args, tokenizer)

    test_dataset = create_dataset(args, tokenizer=tokenizer, split='test',
                                  task_config=TASK_CONFIG, sort_by_pos=not args.sort_by_dict)
    fid_model = create_fid_model(args)

    print(f"Evaluate Checkpoint: {args.eval_ckpt_tag}")
    generator = Generator(args, tokenizer, model, None, test_dataset, fid_model,
                          ckpt_path=os.path.join(args.out_dir, f"{args.eval_ckpt_tag}_checkpoint.pth.tar"), ds_ckpt_tag=args.eval_ckpt_tag,
                          is_label_condition=False, saved_layouts=['input', 'gold', 'pred'],
                          save_entries=['input_str', 'gold_str', 'pred_str', 'pred_str_raw', 'relations', 'violate_num'])

    test_dataset.switch_task(tasks[0])
    index2label = test_dataset.seq_processor.index2label
    label_set = list(index2label.values())
    gen_t_constraint_fn = constrained_decoding.TransformerSortByDictLabelConstraint(
        tokenizer, discrete_degree=max(
            args.discrete_y_grid, args.discrete_x_grid),
        label_set=label_set, index2label=index2label,
        add_sep_token=args.add_sep_token
    )
    gen_ts_constraint_fn = constrained_decoding.TransformerSortByDictLabelSizeConstraint(
        tokenizer, discrete_degree=max(
            args.discrete_y_grid, args.discrete_x_grid),
        label_set=label_set, index2label=index2label,
        add_sep_token=args.add_sep_token
    )
    gen_r_constraint_fn = constrained_decoding.TransformerSortByDictRelationConstraint(
        tokenizer, discrete_degree=max(
            args.discrete_y_grid, args.discrete_x_grid),
        label_set=label_set, index2label=index2label,
        add_sep_token=args.add_sep_token,
        rel_index2type=RelationTypes.index2type()
    )

    eval_tasks = tasks if args.eval_tasks is None else args.eval_tasks.split(
        ",")
    for task in eval_tasks:
        print(
            f"Inference on Task: {task}, Use constrained Decoding: {TASK_CONFIG[task]['use_constrained_decoding']}")
        generator.switch_task(
            task, saved_layouts=TASK_CONFIG[task]['saved_layouts'])
        constraint_fn = None
        if TASK_CONFIG[task]['use_constrained_decoding']:
            if task == 'gen_ts':
                print("Using label & size constraint fn")
                constraint_fn = gen_ts_constraint_fn
            elif task == 'gen_r':
                print("Using relation constraint fn")
                constraint_fn = gen_r_constraint_fn
            else:
                constraint_fn = gen_t_constraint_fn
        generator(TASK_CONFIG[task]['inference'], draw_colors=test_dataset.colors,
                  constraint_fn=constraint_fn)
    generator.clean_up()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    config.add_arguments(parser)
    config.add_trainer_arguments(parser)
    config.add_task_arguments(parser)

    args = parser.parse_args()

    if not args.test:
        train(args)
    else:
        inference(args)
