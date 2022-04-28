import Data
import Networks

import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):

    def __init__(
        self,
        vocab_size,
        padding_size,
        embedding_dim,
        n_hidden_units,
        n_layers,
        n_classes,
        n_words):

        super().__init__()
        self.padding_size = padding_size

        # Embedding
        self.embedding = nn.Embedding(
            vocab_size + 1,
            embedding_dim,
            padding_idx=0)

        self.activation = nn.ReLU()

        # LSTM
        self.lstm1 = nn.LSTM(
            embedding_dim,
            n_hidden_units,
            batch_first=True,
            num_layers=n_layers)
        self.lstm2 = nn.LSTM(
            embedding_dim,
            n_hidden_units,
            batch_first=True,
            num_layers=n_layers)

        # Dense layers
        self.dense = nn.Linear(n_hidden_units * 2 + n_words, n_hidden_units)
        self.out = nn.Linear(n_hidden_units, n_classes)


    def forward(self, X, w):

        emb = self.embedding(X)

        emb1 = emb[:, :self.padding_size+1, :]
        _, (h1, _) = self.lstm1(emb1)
        h1 = torch.squeeze(h1, dim=0)

        emb2 = torch.flip(emb[:, self.padding_size:, :], dims=(1,2))
        _, (h2, _) = self.lstm2(emb2)
        h2 = torch.squeeze(h2, dim=0)

        # adding the onehot encoded word classes
        onehot = F.one_hot(w, num_classes=30).to('cuda')
        concatenated = torch.cat((onehot, h1, h2), dim=1)

        hidden = self.activation(self.dense(concatenated))
        logits = self.out(hidden)

        return logits.squeeze(dim=1)



if __name__ == "__main__":
    # model hyper parameters
    vocab_size = 10000
    padding_size = 10
    embedding_dim = 512
    n_layers = 1
    n_hidden_units = 512
    n_classes = 222

    # Training config
    n_epochs = 50
    val_size = 0.1
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

    net = LSTM(
        vocab_size,
        padding_size,
        embedding_dim,
        n_hidden_units,
        n_layers,
        n_classes,
        n_words)

    net.to(device)

    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=1e-3,
        weight_decay=1e-3)

    # Using categorical cross entropy loss
    loss_f = torch.nn.CrossEntropyLoss()

    best_model = Networks.train(net, train_dataloader, val_dataloader, optimizer, loss_f, n_epochs, device)
    predictions = Networks.predict(best_model, test_dataloader, device)

    # Writing the results from the best performing model to the disk
    with open('predictions_lstm.txt', 'w') as f:
        for label in label_encoder.inverse_transform(predictions):
            f.writelines(f'{label}\n')

