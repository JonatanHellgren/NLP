import numpy as np
import copy
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import accuracy_score

import torch
from tqdm import tqdm

def evaluate(loss_f, net, dataloader, device):
    """
    This function evaluates a network on a given dataloader using GPU is possible.
    It returns the mean loss and the accuracy.
    """
    net.eval()
    all_losses = []
    y_true = []
    y_pred = []

    with torch.no_grad():
        for X_batch, y_batch, w_batch in dataloader:
            # Saving actual labels
            y_true.append(np.array(y_batch))

            # Moving to GPU if possible
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # Predicting the model
            logits = net(X_batch, w_batch)
            guesses = logits.argmax(dim=1)

            # Computing loss and accuracy
            y_pred.append(np.array(guesses.cpu()))
            all_losses.append(loss_f(logits, y_batch).item())

    y_true = np.hstack(y_true)
    y_pred = np.hstack(y_pred)
    acc = accuracy_score(y_true, y_pred)

    return np.mean(all_losses), acc

def predict(net, dataloader, device):
    """
    This function is used to get predictions from a network on the given dataloader.
    It returns them onecold encoded, i.e. as integers.
    """
    net.eval()
    y_pred = []
    with torch.no_grad():
        for X_batch, _, w_batch in dataloader:
            X_batch = X_batch.to(device)
            w_batch = w_batch.to(device)
            logits = net(X_batch, w_batch)
            guesses = logits.argmax(dim=1)
            y_pred.append(np.array(guesses.cpu()))

    return np.hstack(y_pred)

def train(net, train_dataloader, val_dataloader, optimizer, loss_f, n_epochs, device):
    """
    This is the function used to train the networks. It trains using the GPU if possible.
    It trains the network for n_epochs and saves the model every time a new minimum loss has
    been reached. When training is done a .png image with the training history is saved to the disk.
    """
    history = defaultdict(list)

    # For early stopping
    lowest_loss = 1e6
    best_model = None

    # Epochs
    for e in range(1, n_epochs + 1):
        net.train()

        for X_batch, y_batch, w_batch in tqdm(train_dataloader):

            # Move to GPU if possible
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # predict and compute loss with output
            probs = net(X_batch, w_batch)
            loss = loss_f(probs, y_batch)

            # Update the model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluating the model
        train_loss, train_acc = evaluate(loss_f, net, train_dataloader, device)
        val_loss, val_acc = evaluate(loss_f, net, val_dataloader, device)

        # Saving the evaluation metrics 
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch: {e}, Train loss:{round(train_loss,3)}, Train acc:{round(train_acc,3)}\n Validation loss: {round(val_loss,3)}, Validation acc:{round(val_acc,3)}")

        # Earlt stopping
        if val_loss < lowest_loss:
            lowest_loss = val_loss
            best_model = copy.deepcopy(net)

    # For the figure
    plt.style.use('seaborn')
    x = range(len(history['train_loss']))
    fig, ax = plt.subplots(1, 2, figsize=(8,3))
    plt.tight_layout()
    # lines
    ax[0].plot(x, history['train_loss'], x, history['val_loss']);
    ax[0].legend(['train loss', 'val loss']);
    ax[1].plot(x, history['train_acc'], x, history['val_acc']);
    ax[1].legend(['train acc', 'val acc']);
    # text
    fig.suptitle('tmp')
    ax[0].set_ylabel('loss')
    ax[0].set_xlabel('epoch')
    ax[1].set_ylabel('accuracy')
    ax[1].set_xlabel('epoch')
    # save
    plt.savefig('fig.png')

    return best_model
