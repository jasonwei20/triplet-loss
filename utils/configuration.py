import json
from typing import NamedTuple

class triplet_ap_config(NamedTuple):

    exp_id: str = None
    seed_num: int = 0

    #dataset params
    train_path: str = None
    test_path: str = None
    num_output_classes: int = None

    # model params
    max_length: int = 64
    encoding_size: int = 768 #bert embedding size
    embedding_size: int = 30 #output embedding size
    triplet_margin: float = 0.4
    learning_rate: float = 2e-5

    # training params
    train_nc: int = None
    val_subset: int = 1000
    total_updates: int = 10000
    eval_interval: int = 100

    @classmethod
    def from_json(cls, file_path):
        return cls(**json.load(open(file_path, 'r')))