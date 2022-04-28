import Data
import Networks

import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralNet(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        n_hidden_units,
        n_classes,
        n_words):

        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size + 1, # vocab_size + 1 due to the padding and out of vocabulary tokens
            embedding_dim,
            padding_idx=0) # all zeros will be embedded to the origin, all zeros

        self.activation = nn.ReLU()

        # Here we add n_words so that the onehot encoding later will fit
        self.dense = nn.Linear(embedding_dim + n_words, n_hidden_units)

        self.out = nn.Linear(n_hidden_units, n_classes)


    def forward(self, X, w):

        emb = self.embedding(X)
        emb = emb.sum(dim=1)

        # Adding the onehot encoded word class to the results from the word embedding
        onehot = F.one_hot(w, num_classes=30).type(torch.float32).to('cuda')
        concatenated = torch.cat((onehot, emb), dim=1)

        hidden = self.activation(self.dense(concatenated))
        logits = self.out(hidden)

        return logits.squeeze(dim=1)


if __name__ == "__main__":
    # Model hyper parameters
    vocab_size = 10000
    padding_size = 10
    embedding_dim = 512
    n_hidden_units = 512
    n_classes = 222

    # Training config
    n_epochs = 50
    val_size = 0.2
    batch_size = 512
    random_state = 42
    n_words = 30
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_dataloader, val_dataloader, test_dataloader, label_encoder = Data.get_dataloaders(
        vocab_size,
        padding_size,
        val_size,
        batch_size,
        random_state)


    net = NeuralNet(
        vocab_size,
        embedding_dim,
        n_hidden_units,
        n_classes,
        n_words)

    net.to(device)

    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=1e-3,
        weight_decay=1e-3)

    # Using categorical cross entropy los
    loss_f = torch.nn.CrossEntropyLoss()

    best_model = Networks.train(net, train_dataloader, val_dataloader, optimizer, loss_f, n_epochs, device)
    predictions = Networks.predict(best_model, test_dataloader, device)

    # Writing the results from the best performing model to the disk
    with open('predictions_perc.txt', 'w') as f:
        for label in label_encoder.inverse_transform(predictions):
            f.writelines(f'{label}\n')


