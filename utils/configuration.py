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
    hard_negative_mining: str = None

    # augmentation params
    n_aug: int = None
    aug_type: str = None #sr, eda
    alpha: int = 0.1

    @classmethod
    def from_json(cls, file_path):
        return cls(**json.load(open(file_path, 'r')))

class triplet_ap_cl_config(NamedTuple):

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
    hard_negative_mining: str = None

    # augmentation params
    n_aug: int = None
    aug_type: str = None #sr, eda
    alpha: int = 0.1

    # cl params
    first_stage_updates: int = None

    @classmethod
    def from_json(cls, file_path):
        return cls(**json.load(open(file_path, 'r')))

class triplet_ap_cl_gradual_config(NamedTuple):

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

    # augmentation params
    n_aug: int = None
    aug_type: str = None #sr, eda
    second_stage_alpha: int = None
    third_stage_alpha: int = None
    fourth_stage_alpha: int = None
    fifth_stage_alpha: int = None
    sixth_stage_alpha: int = None

    # cl params
    first_stage_updates: int = None
    second_stage_updates: int = None
    third_stage_updates: int = None
    fourth_stage_updates: int = None
    fifth_stage_updates: int = None
    hard_negative_mining: str = None

    @classmethod
    def from_json(cls, file_path):
        return cls(**json.load(open(file_path, 'r')))

class mlp_ap_config(NamedTuple):

    exp_id: str = None
    seed_num: int = 0

    #dataset params
    train_path: str = None
    test_path: str = None
    num_output_classes: int = None

    # model params
    max_length: int = 64
    encoding_size: int = 768 #bert embedding size
    learning_rate: float = 1e-3
    weight_decay: float = 0.05
    decay_gamma: float = 0.95
    model: str = None #"LR"

    # training params
    train_nc: int = None
    val_subset: int = 1000
    num_epochs: int = 100
    eval_interval: int = 100
    minibatch_size: int = 20

    # augmentation params
    n_aug: int = None
    aug_type: str = None #sr, eda
    alpha: int = 0.1

    @classmethod
    def from_json(cls, file_path):
        return cls(**json.load(open(file_path, 'r')))