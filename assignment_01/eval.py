import torch
from torch import nn
from torch.utils.data import DataLoader



class Evaluator:

    def __init__(self, model: nn.Module, test_loader: DataLoader):
        self._model = model
        self._test_loader = test_loader
        
    def eval(self, model: nn.Module) -> float:
        """ Model evaluation, return the model accuracy over test set """
        print("Start testing...")
        self._model = model
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self._test_loader:
                images, labels = data
                outputs = self._model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))
 ##       self._model.eval()
        return 100 * correct / total

