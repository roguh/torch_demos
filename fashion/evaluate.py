import torch
from typing import List


def evaluate(elements, network) -> List[int]:
    """Takes a 1x28x28 tensor and returns the classification."""
    with torch.no_grad():
        if len(elements.size()) == 3:
            elements = elements.view(
                (elements.size()[0], 1, elements.size()[1], elements.size()[2])
            )
        output = network(elements.cuda())
        classes = output.argmax(dim=1, keepdim=True)
        return list(int(c) for c in classes.T[0])
