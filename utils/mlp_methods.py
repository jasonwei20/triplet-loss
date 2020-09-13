import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

from utils import dataloader, mlp_dataloader, bert_avgpool, triplet_models, visualization

class LR(nn.Module):

    def __init__(self, num_classes):
        super(LR, self).__init__()
        self.fc1 = nn.Linear(768, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        output = torch.softmax(x, dim=1)
        return output

class MLP(nn.Module):

    def __init__(self, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(768, 100)
        self.relu1 = nn.Tanh()
        self.fc2 = nn.Linear(100, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        output = torch.softmax(x, dim=1)
        return output

def train_mlp(
    cfg
):

    #load data
    train_sentence_to_label, _, train_label_to_sentences, test_sentence_to_label, train_sentence_to_encoding, test_sentence_to_encoding = dataloader.load_ap_data(cfg)
    train_x, train_y = mlp_dataloader.get_mlp_train_x_y(cfg, train_label_to_sentences, train_sentence_to_encoding)
    test_x, test_y = mlp_dataloader.get_mlp_test_x_y(cfg, test_sentence_to_label, test_sentence_to_encoding)

    if cfg.model == "LR":
        model = LR(num_classes=cfg.num_output_classes)
    else:
        model = MLP(num_classes=cfg.num_output_classes)
    
    optimizer = optim.Adam(params=model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay) #wow, works for even large learning rates
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=cfg.decay_gamma)

    num_minibatches_train = int(train_x.shape[0] / cfg.minibatch_size)
    train_loss_list = []; val_acc_list = []

    ######## training loop ########
    for epoch in range(1, cfg.num_epochs + 1):

        ######## training ########
        model.train(mode=True)

        train_x, train_y = shuffle(train_x, train_y, random_state = cfg.seed_num)

        for minibatch_num in range(num_minibatches_train):

            start_idx = minibatch_num * cfg.minibatch_size
            end_idx = start_idx + cfg.minibatch_size
            train_inputs = torch.from_numpy(train_x[start_idx:end_idx].astype(np.float32))
            train_labels = torch.from_numpy(train_y[start_idx:end_idx].astype(np.long))

            # Forward and backpropagation.
            with torch.set_grad_enabled(mode=True):

                train_outputs = model(train_inputs)
                train_conf, train_preds = torch.max(train_outputs, dim=1)
                train_loss = nn.CrossEntropyLoss()(input=train_outputs, target=train_labels)
                train_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        train_loss_list.append(train_loss)

        ######## validation ########
        model.train(mode=False)

        val_inputs = torch.from_numpy(test_x.astype(np.float32))
        val_labels = torch.from_numpy(test_y.astype(np.long))

        # Feed forward.
        with torch.set_grad_enabled(mode=False):
            val_outputs = model(val_inputs)
            val_confs, val_preds = torch.max(val_outputs, dim=1)
            val_loss = nn.CrossEntropyLoss()(input=val_outputs, target=val_labels)
            val_loss_print = val_loss / val_inputs.shape[0]
            val_acc = accuracy_score(test_y, val_preds)
            val_acc_list.append(val_acc)

    Path(f"plots/{cfg.exp_id}").mkdir(parents=True, exist_ok=True)
    visualization.plot_jasons_lineplot(None, train_loss_list, 'updates', 'training loss', f"{cfg.exp_id} n_train_c={cfg.train_nc} max_val_acc={max(val_acc_list):.3f}", f"plots/{cfg.exp_id}/train_loss.png")    
    visualization.plot_jasons_lineplot(None, val_acc_list, 'updates', 'validation accuracy', f"{cfg.exp_id} n_train_c={cfg.train_nc} max_val_acc={max(val_acc_list):.3f}", f"plots/{cfg.exp_id}/val_acc{max(val_acc_list):.3f}.png")    
    # torch.save(model.state_dict(), 'triplet/models/baseline_covid_weights.pt')
