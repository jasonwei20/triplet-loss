# Triplet loss text classification with BERT

## Data
```
unzip data.zip
```

## Usage

### Triplet Loss Experiments (no hard negative mining)

For experiments, you should comment out the config files for the experiments you want to run:

1. No augmentation
```
python multi_seed_triplet_ap_vanilla.py
```

2. Standard EDA augmentation
```
python multi_seed_triplet_ap_eda_alpha.py
```

3. Curriculum two-stage augmentation
```
python multi_seed_triplet_ap_eda_twostep.py
```

4. Curriculum gradual augmentation
```
python multi_seed_triplet_ap_eda_gradual.py
```

### Triplet Loss Experiments (*with* hard negative mining)

1. No augmentation
```
python multi_seed_triplet_ap_vanilla_mine.py
```

2. Standard EDA augmentation
```
python multi_seed_triplet_ap_eda_mine_alpha.py
```

3. Curriculum two-stage augmentation
```
python multi_seed_triplet_ap_eda_mine_twostep.py
```

4. Curriculum gradual augmentation
```
python multi_seed_triplet_ap_eda_mine_gradual.py
```

### Other augmentation methods in standard vs two-stage curriculum

Token Substitution
```
python triplet_ap_sr_alpha.py
python triplet_ap_sr_twostep.py
```

Word Dropout
```
python triplet_ap_rd_alpha.py
python triplet_ap_rd_twostep.py
```

SwitchOut
```
python triplet_ap_so_alpha.py
python triplet_ap_so_twostep.py
```

Back-translation
```
python triplet_ap_bt_alpha.py
python triplet_ap_bt_twostep.py
```

### Very Simple Baselines

Run LR/MLP baselines for classification of BERT-avgpool encodings:
```
python multi_seed_mlp.py
```

Run k-NN baseline for classification of BERT-avgpool encodings (not used in paper):
```
python knn_ap_vanilla.py
```