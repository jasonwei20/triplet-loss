from utils import dataloader, bert_avgpool, triplet_models, visualization, triplet_methods
import torch
import torch.optim as optim
from tqdm import tqdm
from scipy.spatial import distance
from pathlib import Path


def train_eval_cl_model(
    cfg
):

    #load data
    train_sentence_to_label, train_label_to_sentences, train_label_to_sentences_aug, test_sentence_to_label, train_sentence_to_encoding, test_sentence_to_encoding = dataloader.load_ap_cl_data(cfg)

    # initialize model
    model, loss_fn, optimizer, device = triplet_methods.initialize_model(cfg)

    # train the model
    iter_bar = tqdm(range(cfg.total_updates))
    update_num_list = []; train_loss_list = []; val_acc_list = []

    for update_num in iter_bar:

        #sample differently based on which stage of curriculum learning you're in
        if update_num < cfg.first_stage_updates:
            anchor, pos, neg = dataloader.generate_triplet_batch(train_label_to_sentences, train_sentence_to_encoding, device)
        else:
            anchor, pos, neg = dataloader.generate_triplet_batch(train_label_to_sentences_aug, train_sentence_to_encoding, device)

        model.train()
        model.zero_grad()

        logits = model(anchor, pos, neg)
        train_loss = loss_fn(*logits)

        train_loss.backward()
        optimizer.step()

        if update_num % cfg.eval_interval == 0:

            val_acc = triplet_methods.eval_model(
                model, 
                device, 
                train_sentence_to_label, 
                train_label_to_sentences, 
                train_sentence_to_encoding, 
                test_sentence_to_label, 
                test_sentence_to_encoding,
            )

            iter_bar_str =  ( f"update {update_num}/{cfg.total_updates}: " 
                            + f"mb_train_loss={float(train_loss):.4f}, "
                            + f"val_acc={float(val_acc):.4f}, "
                            )
            iter_bar.set_description(iter_bar_str)
            update_num_list.append(update_num); val_acc_list.append(val_acc); train_loss_list.append(train_loss)

    Path(f"plots/{cfg.exp_id}").mkdir(parents=True, exist_ok=True)
    visualization.plot_jasons_lineplot(update_num_list, train_loss_list, 'updates', 'training loss', f"{cfg.exp_id} n_train_c={cfg.train_nc} max_val_acc={max(val_acc_list):.3f}", f"plots/{cfg.exp_id}/train_loss.png")    
    visualization.plot_jasons_lineplot(update_num_list, val_acc_list, 'updates', 'validation accuracy', f"{cfg.exp_id} n_train_c={cfg.train_nc} max_val_acc={max(val_acc_list):.3f}", f"plots/{cfg.exp_id}/val_acc.png")    
    # # torch.save(model.state_dict(), 'triplet/models/baseline_covid_weights.pt')