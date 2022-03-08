import torch
import matplotlib.pyplot as plt

import evaluate
import fashion_data
import visualize
from net1 import net
from train import load, test, train

TRAIN_NETWORK = False


def main():
    if TRAIN_NETWORK:
        print("Training neural network")
        train(5)
    else:
        print("Loading pre-trained neural network")
        load(net, "net1")

    print("Testing neural network")
    test(net)

    print("Showing a sample of images and predictions")
    vis(0, show_predictions=True)
    plt.show()


def vis(start_ix, show_predictions=False):
    end_ix = start_ix + 16

    targets = fashion_data.fashion.targets[start_ix:end_ix]
    titles = visualize.targets_to_classes(fashion_data.fashion, targets)
    if show_predictions:
        predictions = evaluate.evaluate(
            torch.cat(
                [fashion_data.fashion[i][0] for i in range(start_ix, end_ix + 1)]
            ),
            net,
        )
        titles = [
            title + "\np:" + class_
            for title, class_ in zip(
                titles, visualize.targets_to_classes(fashion_data.fashion, predictions)
            )
        ]

    visualize.show_images(fashion_data.fashion.data[start_ix:end_ix], titles=titles)


if __name__ == "__main__":
    main()
