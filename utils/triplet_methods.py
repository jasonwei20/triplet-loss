from utils import dataloader, bert_avgpool, triplet_models, visualization
import torch
import torch.optim as optim
from tqdm import tqdm
from scipy.spatial import distance
from pathlib import Path

def initialize_model(
    cfg,
):
    
    embedding_net = triplet_models.EmbeddingNet(cfg.encoding_size, cfg.embedding_size)
    model = triplet_models.TripletNet(embedding_net)
    loss_fn = triplet_models.TripletLoss(margin=0.4, distance_type='C')
    optimizer = optim.Adam(model.parameters(), lr=2e-5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model, loss_fn, optimizer, device

def eval_model(
    model,
    device,
    train_sentence_to_label,
    train_label_to_sentences,
    train_sentence_to_encoding,
    test_sentence_to_label,
    test_sentence_to_encoding,
):
    model.eval()

    def get_sentence_embedding(model, sentence_to_encoding, sentence):
        return model.get_embedding(torch.tensor(sentence_to_encoding[sentence]).to(device)).detach().cpu().numpy()

    def get_train_sentence_to_embedding(model, train_label_to_sentences, train_sentence_to_encoding):
        train_sentence_to_embedding = {}
        for train_sentences in train_label_to_sentences.values():
            for train_sentence in train_sentences:
                embedding = get_sentence_embedding(model, train_sentence_to_encoding, train_sentence)
                train_sentence_to_embedding[train_sentence] = embedding
        return train_sentence_to_embedding
        
    def get_closest_train_sentence(test_sentence_embedding, train_sentence_to_embedding):
        train_sentence_to_dist_list = [ (train_sentence, distance.cosine(test_sentence_embedding, train_sentence_embedding)) for train_sentence, train_sentence_embedding in train_sentence_to_embedding.items()]
        sorted_train_sentence_dist_list = list(sorted(train_sentence_to_dist_list, key=lambda tup: tup[1]))
        return sorted_train_sentence_dist_list[0][0]

    train_sentence_to_embedding = get_train_sentence_to_embedding(model, train_label_to_sentences, train_sentence_to_encoding)

    num_correct = 0 #probably should be refactored
    for test_sentence, label in test_sentence_to_label.items():
        test_sentence_embedding = get_sentence_embedding(model, test_sentence_to_encoding, test_sentence)
        closest_train_sentence = get_closest_train_sentence(test_sentence_embedding, train_sentence_to_embedding)
        predicted_label = train_sentence_to_label[closest_train_sentence]
        if predicted_label == label:
            num_correct += 1
    
    acc = num_correct / len(test_sentence_to_label)
    return acc

def train_eval_model(
    cfg
):

    #load data
    train_sentence_to_label, train_label_to_sentences, train_label_to_sentences_aug, test_sentence_to_label, train_sentence_to_encoding, test_sentence_to_encoding = dataloader.load_ap_data(cfg)
    
    # initialize model
    model, loss_fn, optimizer, device = initialize_model(cfg)

    # train the model
    iter_bar = tqdm(range(cfg.total_updates))
    update_num_list = []; train_loss_list = []; val_acc_list = []

    Path(f"plots{cfg.seed_num}/{cfg.exp_id}").mkdir(parents=True, exist_ok=True)
    writer = open(f"plots{cfg.seed_num}/{cfg.exp_id}/logs.csv", "w")
    mb_size = 64
    target_activated_examples = 64
    avg_percent_activated = 1.0
    percent_activated_list = []

    for update_num in iter_bar:

        anchor, pos, neg = dataloader.generate_triplet_batch(train_label_to_sentences_aug, train_sentence_to_encoding, device, mb_size=mb_size)

        model.train()
        model.zero_grad()

        logits = model(anchor, pos, neg)
        train_loss, percent_activated = loss_fn(*logits)

        train_loss.backward()
        optimizer.step()
        percent_activated_list.append(percent_activated)

        if update_num % cfg.eval_interval == 0:

            val_acc = eval_model(
                model, 
                device, 
                train_sentence_to_label, 
                train_label_to_sentences, 
                train_sentence_to_encoding, 
                test_sentence_to_label, 
                test_sentence_to_encoding,
            )
            avg_percent_activated = sum(percent_activated_list) / len(percent_activated_list)

            iter_bar_str =  ( f"update {update_num}/{cfg.total_updates}: " 
                            + f"mb_train_loss={float(train_loss):.4f}, "
                            + f"val_acc={float(val_acc):.4f}, "
                            + f"percent_activated={float(avg_percent_activated):.3f}, "
                            + f"mb_size={mb_size}"
                            )
            iter_bar.set_description(iter_bar_str)
            update_num_list.append(update_num); val_acc_list.append(val_acc); train_loss_list.append(train_loss)
            writer.write(f"{update_num},{val_acc:.4f},{train_loss:.4f}\n")
            percent_activated_list = []
            if cfg.hard_negative_mining == 'semi-hard':
                mb_size = min(int(target_activated_examples / avg_percent_activated), 2000)

    visualization.plot_jasons_lineplot(update_num_list, train_loss_list, 'updates', 'training loss', f"{cfg.exp_id} n_train_c={cfg.train_nc} max_val_acc={max(val_acc_list):.3f}", f"plots{cfg.seed_num}/{cfg.exp_id}/train_loss.png")    
    visualization.plot_jasons_lineplot(update_num_list, val_acc_list, 'updates', 'validation accuracy', f"{cfg.exp_id} n_train_c={cfg.train_nc} max_val_acc={max(val_acc_list):.3f}", f"plots{cfg.seed_num}/{cfg.exp_id}/val_acc{max(val_acc_list):.3f}.png")    
    # torch.save(model.state_dict(), 'triplet/models/baseline_covid_weights.pt')
