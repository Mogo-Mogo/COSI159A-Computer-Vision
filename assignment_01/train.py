import os
import time

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch import optim
import torch.nn.functional as F
import math

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

        optimizer = optim.SGD(params=self._model.parameters(), lr=lr)
        evaluator = Evaluator(model=self._model, test_loader=test_loader)
        loss_track = AverageMeter()
        self._model.train()
        best_acc = 0.0
        best_model = None

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

            #evaluation every other epoch

            if i % 2 == 1:
                acc = evaluator.eval(model=self._model)
                print('Accuracy of the network on the 10000 test images: ' + str(math.floor(acc)) + "%")
                if best_acc < acc and acc > 97.0:
                    best_acc = acc
                    best_model = self._model
                    print("Best model found, saving to %s" % save_dir)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    torch.save(self._model.state_dict(), os.path.join(save_dir, "mnist.pth"))

        #saving the best model both as a .pth file and as a runtime object allows for quicker callback later

        print("Testing best model...")
        acc = evaluator.eval(model=best_model)
        print('Accuracy of the network on the 10000 test images: ' + str(acc) + "%")
        print("Training completed, model is saved at %s" % save_dir)

        return

