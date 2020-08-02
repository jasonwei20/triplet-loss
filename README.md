# Triplet loss text classification with BERT

## Usage

### Very Simple Baselines
Run k-NN baseline for text classification of BERT-avgpool encodings:
```
python knn_ap_vanilla.py
```

Run MLP/LR baselines for text classification of BERT-avgpool encodings:
```
python mlp_ap_vanilla.py
```

### Triplet Loss Experiments
Run vanilla triplet loss classification of BERT-avgpool encodings:
```
python triplet_ap_vanilla.py
```

Add in simple synonym replacement data augmentation with 10% of the words replaced (alpha = 0.1):
```
python triplet_ap_sr.py
```

