import os
import time

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch import optim
import torch.nn.functional as F

from utils import AverageMeter
from eval import Evaluator

class Trainer:
    """ Trainer for MNIST classification """

    def __init__(self, model: nn.Module):
        self._model = model

    def train(
            self,
            train_loader: DataLoader,
            test_loader: DataLoader,
            epochs: int,
            lr: float,
            save_dir: str,
    ) -> None:
        """ Model training, TODO: consider adding model evaluation into the training loop """

        optimizer = optim.SGD(params=self._model.parameters(), lr=lr)
        evaluator = Evaluator(model=self._model, test_loader=test_loader)
        loss_track = AverageMeter()
        self._model.train()
        best_acc = 0.0

        print("Start training...")
        for i in range(epochs):
            tik = time.time()
            loss_track.reset()
            for data, target in train_loader:
                optimizer.zero_grad()
                output = self._model(data)

                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()

                loss_track.update(loss.item(), n=data.size(0))

            elapse = time.time() - tik
            print("Epoch: [%d/%d]; Time: %.2f; Loss: %.5f" % (i + 1, epochs, elapse, loss_track.avg))

            if i % 2 == 1:
                acc = evaluator.eval(model=self._model)
                if best_acc < acc and acc > 97.0:
                    best_acc = acc
                    print("Best model found, saving to %s" % save_dir)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    torch.save(self._model.state_dict(), os.path.join(save_dir, "mnist.pth"))
                


        print("Training completed, saving model to %s" % save_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(self._model.state_dict(), os.path.join(save_dir, "mnist.pth"))

        return

    def eval(self, test_loader: DataLoader) -> float:
        """ Model evaluation, return the model accuracy over test set """
        print("Start testing...")
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                outputs = self._model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))
        return

    def infer(self, sample: Tensor) -> int:
        """ Model inference: input an image, return its class index """
        return

    def load_model(self, path: str) -> None:
        """ load model from a .pth file """
        return

