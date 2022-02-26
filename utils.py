import matplotlib.pyplot as plt
import torch


def label2onehot(labels):
    uni_labels = labels.unique(sorted=True)
    k = 0
    dic = {}
    for l in uni_labels:
        dic[str(l.item())] = k
        k += 1
    for (i, l) in enumerate(labels):
        labels[i] = dic[str(l.item())]
    return labels


def CrossEntropy_uniform(pred, batch_size, n_class):
    # logsoftmax = nn.LogSoftmax(dim=1)
    unif = torch.full((batch_size, n_class), 1/n_class)
    left_sum = torch.sum(unif * torch.log(pred), 1)
    return torch.mean(-1 * left_sum)


def make_Loss_Graph(G_total_loss, D_total_loss):

    fig = plt.figure(figsize=(10,5))

    plt.title("Generator and Discriminator Loss")
    plt.plot(G_total_loss, label="G_losses")
    plt.plot(D_total_loss, label="D_losses")
    plt.xlabel("epochs")
    plt.ylabel("Loss")
    plt.legend()

    fig.savefig("training_loss.png")