import Data
import Networks

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):

    def __init__(self,
                vocab_size,
                padding_size,
                embedding_dim,
                n_channels,
                kernel_size,
                n_hidden_units,
                n_classes,
                n_words):

        super().__init__()

        # Embedding
        self.embedding = nn.Embedding(
                vocab_size + 1,
                embedding_dim,
                padding_idx=0) 

        # Convolutions
        self.convolutional1 = nn.Conv1d(
            in_channels=embedding_dim, 
            out_channels=n_channels[0], 
            kernel_size=kernel_size[0])
        self.convolutional2 = nn.Conv1d(
            in_channels=n_channels[0], 
            out_channels=n_channels[1], 
            kernel_size=kernel_size[1])
        self.pooling = nn.MaxPool2d(2)

        self.activation = nn.ReLU()

        # Here we need to compute the input size for the dense layer
        final_emb_dim = (padding_size * 2 + 1) - (sum(kernel_size) - len(kernel_size)) 
        padding_output_size = (final_emb_dim // 2) * (n_channels[1] // 2)

        # Dense layers
        self.dense = nn.Linear(padding_output_size + n_words, n_hidden_units)
        self.out = nn.Linear(n_hidden_units, n_classes)


    def forward(self, X, w):

        emb = self.embedding(X).transpose(1,2)

        feature_map1 = self.activation(self.convolutional1(emb))
        feature_map2 = self.activation(self.convolutional2(feature_map1))

        pooled = self.pooling(feature_map2)
        flattened = torch.flatten(pooled, 1, 2)

        # Adding the onehot encoded word classes to the flattened feature maps
        # from the convolutional layers
        onehot = F.one_hot(w, num_classes=30).to('cuda')
        concatenated = torch.cat((onehot, flattened), dim=1)

        hidden = self.activation(self.dense(concatenated))
        logits = self.out(hidden)

        return logits.squeeze(dim=1)



if __name__ == "__main__":
    # model hyper parameters
    vocab_size = 10000
    padding_size = 10
    embedding_dim = 512
    n_channels = [32, 64]
    kernel_size = [3, 3]
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

    net = ConvNet(
        vocab_size,
        padding_size,
        embedding_dim,
        n_channels,
        kernel_size,
        n_hidden_units,
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
    with open('predictions_conv.txt', 'w') as f:
        for label in label_encoder.inverse_transform(predictions):
            f.writelines(f'{label}\n')


    
    




