from typing import Dict
import numpy as np
import random
from collections import defaultdict

from data import RicoDataset, PubLayNetDataset


class T5MultiTaskDataset:
    def __init__(self, cargs, tokenizer, task_config,
                 sort_by_pos: bool = True, *args, **kwargs) -> None:
        self.task_config = task_config
        self.tasks = cargs.tasks.split(",")
        self.task_dict = dict()
        for task in self.tasks:
            if task in self.task_config:
                self.task_dict[task] = dict()

                seq_processor_params = dict()
                for param, default_value in self.task_config[task].get("seq_processor_params", list()):
                    seq_processor_params[param] = getattr(cargs, param, default_value)
                seq_processor = self.task_config[task]['seq_processor'](
                    cargs.dataset, tokenizer, add_sep_token=cargs.add_sep_token, **seq_processor_params)

                task_sort_by_pos = getattr(
                    cargs, f'{task}_sort_by_pos', sort_by_pos)
                dataset = self.task_config[task]['dataset'](
                    cargs, tokenizer, seq_processor, sort_by_pos=task_sort_by_pos,
                    shuffle_before_sort_by_label=getattr(cargs, f'{task}_shuffle_before_sort_by_label', False),
                    sort_by_pos_before_sort_by_label=getattr(cargs, f'{task}_sort_by_pos_before_sort_by_label', False))
                self.task_dict[task]['seq_processor'] = seq_processor
                self.task_dict[task]['dataset'] = dataset
        self.add_task_prompt = cargs.add_task_prompt
        super().__init__(*args, **kwargs)

    @property
    def num_tasks(self):
        return len(self.tasks)

    @property
    def seq_processor(self):
        return {task: v['seq_processor'] for task, v in self.task_dict.items()}


class T5MultiTaskSamplingDataset(T5MultiTaskDataset):

    def __init__(self, cargs, tokenizer, task_config,
                 sort_by_pos: bool = True, *args, **kwargs) -> None:
        super().__init__(cargs, tokenizer, task_config, sort_by_pos, *args, **kwargs)
        # Allow fine-grained control
        if cargs.task_weights:
            weights = np.array([float(w) for w in cargs.task_weights.split(',')])
            self.task_probs = weights / weights.sum()
        else:
            self.task_probs = np.ones(self.num_tasks) * (1 / self.num_tasks)
        print("Task Probs: ", self.task_probs)

    def _sample_task(self) -> int:
        task_id = np.random.choice(self.num_tasks, 1, p=self.task_probs)[0]
        return task_id

    def process(self, data) -> Dict:
        sampled_task_id = self._sample_task()
        sampled_task = self.tasks[sampled_task_id]
        result = self.task_dict[sampled_task]['dataset'].process(data)
        if self.add_task_prompt:
            in_str = "{} {}".format(self.task_config[sampled_task]['prompt'], result['in_str'])
        else:
            in_str = result['in_str']
        return {
            'in_str': in_str,
            'out_str': result['out_str'],
            'gold_labels': result['gold_labels'],
            'gold_bboxes': result['gold_bboxes'],
            'name': result['name'],
            'task_name': sampled_task,
            'task_id': self.task_config[sampled_task]['task_id']
        }


class T5MultiTaskConcatDataset(T5MultiTaskDataset):

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        task_id, data_idx = idx // len(self.data), idx % len(self.data)
        task, data = self.tasks[task_id], self.data[data_idx]
        result = self.task_dict[task]['dataset'].process(data)
        if self.add_task_prompt:
            in_str = "{} {}".format(self.task_config[task]['prompt'], result['in_str'])
        else:
            in_str = result['in_str']
        return {
            'in_str': in_str,
            'out_str': result['out_str'],
            'gold_labels': result['gold_labels'],
            'gold_bboxes': result['gold_bboxes'],
            'name': result['name'],
            'task_name': task,
            'task_id': self.task_config[task]['task_id']
        }


class T5MultiTaskRotationDataset(T5MultiTaskDataset):

    def __init__(self, cargs, tokenizer, task_config, sort_by_pos, *args, **kwargs) -> None:
        super().__init__(cargs, tokenizer, task_config, sort_by_pos, *args, **kwargs)
        self._curr_task = None

    @property
    def num_tasks(self):
        return len(self.tasks)

    @property
    def curr_task(self):
        return self._curr_task

    @property
    def seq_processor(self):
        if self.curr_task is None:
            return None
        return self.task_dict[self.curr_task]['seq_processor']

    def switch_task(self, task: str):
        if task not in self.tasks:
            raise Exception(f"No such task: {task}")
        self._curr_task = task

    def __getitem__(self, idx):
        data = self.data[idx]
        result = self.task_dict[self.curr_task]['dataset'].process(data)
        if self.add_task_prompt:
            in_str = "{} {}".format(self.task_config[self.curr_task]['prompt'], result['in_str'])
        else:
            in_str = result['in_str']
        result.update({
            'in_str': in_str,
            'task_name': self.curr_task,
            'task_id': self.task_config[self.curr_task]['task_id']
        })
        return result


class T5MultiTaskPartitionDataset(T5MultiTaskDataset):

    def __init__(self, cargs, tokenizer, task_config, sort_by_pos: bool = True,
                 task_buckets: str = None, permutation_seed: int = 100, *args, **kwargs) -> None:
        super().__init__(cargs, tokenizer, task_config, sort_by_pos, *args, **kwargs)

        if task_buckets is None:
            buckets = list(range(self.num_tasks))
        else:
            buckets = list(map(int, task_buckets.replace("\\", "").split(',')))
        if self.num_tasks != len(buckets):
            raise ValueError("Task bucket specification should be equivalent to the number of tasks")

        bucket2tasks = defaultdict(list)
        for task, bucket in zip(self.tasks, buckets):
            bucket2tasks[bucket].append(task)
        self.bucket2tasks, bucket_map = list(), dict()
        for b, tasks in bucket2tasks.items():
            self.bucket2tasks.append(tasks)
            bucket_map[b] = len(self.bucket2tasks) - 1
        self.num_buckets = len(bucket2tasks)
        print("bucket2tasks: ", self.bucket2tasks)

        # partition
        data_idx = np.random.RandomState(
            seed=permutation_seed).permutation(np.arange(len(self)))
        bucket_size = [int(np.ceil(len(self) * float(len(tasks) / self.num_tasks)))
                       for tasks in self.bucket2tasks]
        print(bucket_size)
        data_idx_splits = np.split(data_idx, np.cumsum(bucket_size)[:-1])
        self.data2bucket = dict()
        for bucket_id, indices in enumerate(data_idx_splits):
            print(f"bucket_id: {bucket_id}, num items: {len(indices)}")
            for idx in indices:
                self.data2bucket[idx] = bucket_id

        # add those tasks in buckets with negative id to other buckets
        for bid, tasks in bucket2tasks.items():
            if bid >= 0:
                continue
            # add
            for _bid, _ in bucket2tasks.items():
                if _bid >= 0:
                    self.bucket2tasks[bucket_map[_bid]].extend(tasks)
        print("utimate bucket2tasks: ", self.bucket2tasks)

    def _get_task_id(self, idx):
        bucket_id = self.data2bucket[idx]
        tasks = self.bucket2tasks[bucket_id]
        return random.choice(tasks)

    def __getitem__(self, idx):
        task = self._get_task_id(idx)
        data = self.data[idx]
        result = self.task_dict[task]['dataset'].process(data)
        if self.add_task_prompt:
            in_str = "{} {}".format(self.task_config[task]['prompt'], result['in_str'])
        else:
            in_str = result['in_str']
        return {
            'in_str': in_str,
            'out_str': result['out_str'],
            'gold_labels': result['gold_labels'],
            'gold_bboxes': result['gold_bboxes'],
            'name': result['name'],
            'task_name': task,
            'task_id': self.task_config[task]['task_id']
        }


class T5MultiTaskFineGrainedPartitionDataset(T5MultiTaskDataset):

    def __init__(self, cargs, tokenizer, task_config, sort_by_pos: bool = True,
                 task_data_size: str = None, permutation_seed: int = 100, task_weights: str = None, *args, **kwargs) -> None:
        super().__init__(cargs, tokenizer, task_config, sort_by_pos, *args, **kwargs)

        self.task_data_size = list(map(int, task_data_size.split(",")))
        if self.num_tasks != len(self.task_data_size):
            raise ValueError("Task data size specification should be equivalent to the number of tasks")
        if sum(self.task_data_size) > len(self):
            raise ValueError("The sume of task data size shoule not be larger than the size of training data")

        # num_buckets = len(self.task_data_size) + 1
        bucket_size = self.task_data_size + [len(self) - sum(self.task_data_size)]
        self.bucket2tasks = [[task] for task in self.tasks]
        self.bucket2tasks.append(self.tasks)

        print("Buckets: ", self.bucket2tasks)
        print("Bucket size: ", bucket_size)
        data_idx = np.random.RandomState(seed=permutation_seed).permutation(np.arange(len(self)))
        data_idx_splits = np.split(data_idx, np.cumsum(bucket_size)[:-1])
        self.data2bucket = dict()
        for bucket_id, indices in enumerate(data_idx_splits):
            print(f"bucket_id: {bucket_id}, num items: {len(indices)}")
            for idx in indices:
                self.data2bucket[idx] = bucket_id

        # task sampling weights in the task-shared buckets
        if task_weights is not None:
            sampling_weights = np.array([float(w) for w in task_weights.split(',')])
            self.task_probs = sampling_weights / sampling_weights.sum()
        else:
            self.task_probs = np.ones(self.num_tasks) * (1 / self.num_tasks)
        print("Task sampling Probs in the shared bucket: ", self.task_probs)

    def _get_task_id(self, idx):
        bucket_id = self.data2bucket[idx]
        tasks = self.bucket2tasks[bucket_id]
        # the last bucket is shared
        if bucket_id == len(self.bucket2tasks) - 1:
            return random.choices(tasks, self.task_probs, k=1)[0]
        else:
            return tasks[0]

    def __getitem__(self, idx):
        task = self._get_task_id(idx)
        data = self.data[idx]
        result = self.task_dict[task]['dataset'].process(data)
        if self.add_task_prompt:
            in_str = "{} {}".format(self.task_config[task]['prompt'], result['in_str'])
        else:
            in_str = result['in_str']
        return {
            'in_str': in_str,
            'out_str': result['out_str'],
            'gold_labels': result['gold_labels'],
            'gold_bboxes': result['gold_bboxes'],
            'name': result['name'],
            'task_name': task,
            'task_id': self.task_config[task]['task_id']
        }


class T5RicoMultiTaskSamplingDataset(T5MultiTaskSamplingDataset, RicoDataset):
    def __init__(self, args, tokenizer, task_config, split, online_process=True,
                 remove_too_long_layout: bool = False, sort_by_pos: bool = True) -> None:
        super().__init__(args, tokenizer, task_config=task_config,
                         sort_by_pos=sort_by_pos,
                         root=args.data_dir, split=split,
                         max_num_elements=args.max_num_elements,
                         online_process=online_process)


class T5RicoMultiTaskConcatDataset(T5MultiTaskConcatDataset, RicoDataset):
    def __init__(self, args, tokenizer, task_config, split, online_process=True,
                 remove_too_long_layout: bool = False, sort_by_pos: bool = True) -> None:
        super().__init__(args, tokenizer, task_config=task_config,
                         sort_by_pos=sort_by_pos,
                         root=args.data_dir, split=split,
                         max_num_elements=args.max_num_elements,
                         online_process=online_process)


class T5RicoMultiTaskRotationDataset(T5MultiTaskRotationDataset, RicoDataset):
    def __init__(self, args, tokenizer, task_config, split, online_process=True,
                 remove_too_long_layout: bool = False, sort_by_pos: bool = True) -> None:
        super().__init__(args, tokenizer, task_config=task_config,
                         sort_by_pos=sort_by_pos,
                         root=args.data_dir, split=split,
                         max_num_elements=args.max_num_elements,
                         online_process=online_process)


class T5RicoMultiTaskPartitionDataset(T5MultiTaskPartitionDataset, RicoDataset):
    def __init__(self, args, tokenizer, task_config, split, online_process=True,
                 remove_too_long_layout: bool = False, sort_by_pos: bool = True,
                 task_buckets: str = None) -> None:
        super().__init__(args, tokenizer, task_config=task_config,
                         sort_by_pos=sort_by_pos, task_buckets=task_buckets,
                         root=args.data_dir, split=split,
                         max_num_elements=args.max_num_elements,
                         online_process=online_process)


class T5RicoMultiTaskFineGrainedPartitionDataset(T5MultiTaskFineGrainedPartitionDataset, RicoDataset):
    def __init__(self, args, tokenizer, task_config, split, online_process=True,
                 remove_too_long_layout: bool = False, sort_by_pos: bool = True,
                 task_data_size: str = None, task_weights: str = None) -> None:
        super().__init__(args, tokenizer, task_config=task_config, sort_by_pos=sort_by_pos,
                         task_data_size=task_data_size, task_weights=task_weights,
                         root=args.data_dir, split=split,
                         max_num_elements=args.max_num_elements,
                         online_process=online_process)


class T5PubLayNetMultiTaskSamplingDataset(T5MultiTaskSamplingDataset, PubLayNetDataset):
    def __init__(self, args, tokenizer, task_config, split, online_process=True,
                 remove_too_long_layout: bool = False, sort_by_pos: bool = True) -> None:
        super().__init__(args, tokenizer, task_config=task_config,
                         sort_by_pos=sort_by_pos,
                         root=args.data_dir, split=split,
                         max_num_elements=args.max_num_elements,
                         online_process=online_process)


class T5PubLayNetMultiTaskConcatDataset(T5MultiTaskConcatDataset, PubLayNetDataset):
    def __init__(self, args, tokenizer, task_config, split, online_process=True,
                 remove_too_long_layout: bool = False, sort_by_pos: bool = True) -> None:
        super().__init__(args, tokenizer, task_config=task_config,
                         sort_by_pos=sort_by_pos,
                         root=args.data_dir, split=split,
                         max_num_elements=args.max_num_elements,
                         online_process=online_process)


class T5PubLayNetMultiTaskRotationDataset(T5MultiTaskRotationDataset, PubLayNetDataset):
    def __init__(self, args, tokenizer, task_config, split, online_process=True,
                 remove_too_long_layout: bool = False, sort_by_pos: bool = True) -> None:
        super().__init__(args, tokenizer, task_config=task_config,
                         sort_by_pos=sort_by_pos,
                         root=args.data_dir, split=split,
                         max_num_elements=args.max_num_elements,
                         online_process=online_process)


class T5PubLayNetMultiTaskPartitionDataset(T5MultiTaskPartitionDataset, PubLayNetDataset):
    def __init__(self, args, tokenizer, task_config, split, online_process=True,
                 remove_too_long_layout: bool = False, sort_by_pos: bool = True,
                 task_buckets: str = None) -> None:
        super().__init__(args, tokenizer, task_config=task_config,
                         sort_by_pos=sort_by_pos, task_buckets=task_buckets,
                         root=args.data_dir, split=split,
                         max_num_elements=args.max_num_elements,
                         online_process=online_process)


class T5PubLayNetMultiTaskFineGrainedPartitionDataset(T5MultiTaskFineGrainedPartitionDataset, PubLayNetDataset):
    def __init__(self, args, tokenizer, task_config, split, online_process=True,
                 remove_too_long_layout: bool = False, sort_by_pos: bool = True,
                 task_data_size: str = None, task_weights: str = None) -> None:
        super().__init__(args, tokenizer, task_config=task_config, sort_by_pos=sort_by_pos,
                         task_data_size=task_data_size, task_weights=task_weights,
                         root=args.data_dir, split=split,
                         max_num_elements=args.max_num_elements,
                         online_process=online_process)
