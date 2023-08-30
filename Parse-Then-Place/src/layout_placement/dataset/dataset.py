import os

from torch.utils.data import Dataset

from ir.executor_rico import Executor as Rico_Executor
from ir.executor_web import Executor as Web_Executor
from layout_placement.placement_utils.utils import read_json, read_txt


class PlacementDataset(Dataset):

    def __init__(self, config, split) -> None:
        super().__init__()
        assert split in ["train", "val", "test", "prediction"]
        self.config = config
        self.split = split
        self.source_path = os.path.join(self.config.args.data_dir, self.data_dict[split]['source'])
        self.target_path = os.path.join(self.config.args.data_dir, self.data_dict[split]['target'])
        self.num = self.config.args.num_train if split == "train" else None
        if self.source_path.endswith('txt'):
            self._build_txt_data()
        elif self.source_path.endswith('json'):
            self._build_json_data()

    @property
    def data_dict(self):
        return {
            'train': {'source': self.config.args.train_source_file, 'target': self.config.args.train_target_file},
            'val': {'source': self.config.args.val_source_file, 'target': self.config.args.val_target_file},
            'test': {'source': self.config.args.test_ground_truth_file, 'target': ''},
            'prediction': {'source': self.config.args.stage_one_prediction_file, 'target': ''}
        }

    def _build_txt_data(self):
        self.source_data = read_txt(self.source_path)
        self.target_data = read_txt(self.target_path)
        len_source, len_target = len(self.source_data), len(self.target_data)
        assert len_source == len_target, "lines in source and target file of dataset should have same length."
        if self.num is not None:
            self.source_data = self.source_data[:self.num]
            self.target_data = self.target_data[:self.num]

    def _build_json_data(self):
        _data = read_json(self.source_path)
        if self.config.args.is_two_stage:
            if self.split == 'prediction':
                self._process_stage_one_prediction(_data)
            else:
                self.source_data = [item['execution'] for item in _data]
                if self.config.args.add_complete_token:
                    self.target_data = [item['layout_seq_with_completion'] for item in _data]
                else:
                    self.target_data = [item['layout_seq_without_completion'] for item in _data]
        else:
            self.source_data = [item['text'] for item in _data]
            if self.config.args.add_complete_token:
                self.target_data = [item['layout_seq_with_completion'] for item in _data]
            else:
                self.target_data = [item['layout_seq_without_completion'] for item in _data]
        if self.num is not None:
            self.source_data = self.source_data[:self.num]
            self.target_data = self.target_data[:self.num]

    def _process_stage_one_prediction(self, predictions):
        if self.config.args.dataset_name == 'web':
            executor = Web_Executor(os.path.join(os.getcwd(), './ir/grammar_web.lark'))
        elif self.config.args.dataset_name == 'rico':
            executor = Rico_Executor(os.path.join(os.getcwd(), './ir/grammar_rico.lark'))
        self.source_data = []
        self.target_data = []
        for prediction in predictions:
            lf = prediction['pred_lf'] if self.config.args.test_prediction_ir else prediction['gold_lf']
            try:
                execution = executor(lf)[0].input
                self.source_data.append(execution)
            except:
                self.source_data.append('')
            self.target_data.append('')

    def __len__(self):
        return len(self.source_data)

    def __getitem__(self, index):
        src_text = self.source_data[index]
        tgt_text = self.target_data[index]

        return {
            'source_text': src_text,
            'target_text': tgt_text,
        }

