import os

import progressbar
import torch
import torch.nn as nn
import torch.optim as optim

from fashion_data import testloader, trainloader
from net1 import net

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


def model_fname(name: str) -> str:
    return f"results/{name}_model.pth"


def save(network, name):
    os.makedirs(os.path.dirname(model_fname(name)), exist_ok=True)
    torch.save(network.state_dict(), model_fname(name))


def load(network, name):
    if os.path.exists(model_fname("net1")):
        return network.load_state_dict(torch.load(model_fname(name)))
    return None


def train(epochs=2):
    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, (inputs, labels) in progressbar.progressbar(enumerate(trainloader)):
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs.cuda())
            loss = criterion(outputs, labels.cuda())
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
        save(net, "net1")


def test(network):
    correct = 0
    total = 0
    for inputs, labels in progressbar.progressbar(testloader):
        with torch.no_grad():
            outputs = network(inputs.cuda())
            output_labels = outputs.data.max(1, keepdim=True)[1].T
            results = output_labels == labels.cuda()
        correct += results.sum()
        total += len(inputs)
        # [(int(a),int(b)) for a,b in torch.stack((output_labels[(output_labels == labels.cuda()) == False], labels[((output_labels == labels.cuda()) == False)[0]].cuda()), dim=1)]
        # TODO make a confusion matrix (sparse, hopefully)

        # TODO visualize exactly which images are confusing
    return int(correct) / total


if __name__ == "__main__":
    if load(net, "net1"):
        print("Loaded model from", model_fname("net1"))
    train(5)
