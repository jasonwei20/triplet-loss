import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin=0.4, distance_type="C", account_for_nonzeros=False):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.distance_type=distance_type.lower().strip()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.account_for_nonzeros = account_for_nonzeros

    def forward(self, anchor, positive, negative):
        
        if self.distance_type == "c":
            ## cosine distance 
            distance_positive = -self.cos(anchor, positive)
            distance_negative= -self.cos(anchor, negative)
            losses = F.relu(distance_positive - distance_negative + self.margin)
            
        elif self.distance_type == "e":
            
            ## this is using Euclidean distance 
            distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
            distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
            losses = F.relu(distance_positive - distance_negative + self.margin)
        else:
            raise Exception('please specify distance_type as C or E')

        semi_hard_indexes = [i for i in range(len(losses)) if losses[i] > 0]
        percent_activated = len(semi_hard_indexes) / len(losses)
        if self.account_for_nonzeros:
            loss = losses.sum() / len(semi_hard_indexes)
        else:
            loss = losses.mean()

        return loss, percent_activated


class EmbeddingNet(nn.Module):
    def __init__(self,hidden_size,output_size,dropout_p=0.0,layernorm=False,batchnorm=False):
        super(EmbeddingNet, self).__init__()
        self.dropout1 = nn.Dropout(p=dropout_p)
        self.dropout2 = nn.Dropout(p=0.4)
        self.Tanh = nn.Tanh()  ##nn.Tanh() / nn.ReLU() etc        
        self.fc1 = nn.Linear(hidden_size, 200)
        if batchnorm:
            self.bn1=nn.BatchNorm1d(num_features=200)
        else:
            self.bn1 = None
        if layernorm:
            self.ln1 = nn.LayerNorm(200)
        else:
            self.ln1 = None
        self.fc2 = nn.Linear(200, output_size)
        
    def forward(self, x):
        output = self.dropout1(x)
        output = self.fc1(output)
        if not self.bn1 is None:
            output = self.bn1(output)
        if not self.ln1 is None:
            output = self.ln1(output)
        output = self.Tanh(output)
        output = self.dropout2(output)
        output = self.fc2(output)
        
        return output

    def get_embedding(self, x):
        return self.forward(x)


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)