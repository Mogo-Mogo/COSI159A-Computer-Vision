import argparse

import torch
import torchvision

from model import Net
from train import Trainer

'''default values are the best performance that I was able to find.'''
def parse_args():
    parser = argparse.ArgumentParser(description='mnist classification')
    parser.add_argument('--epochs', type=int, default=64, help="training epochs")
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--bs', type=int, default=32, help="batch size")
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    # model
    model = Net()

    # datasets
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(root='./data/', train=True, download=True, transform=transform),
        batch_size=args.bs,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(root='./data/', train=False, download=True, transform=transform),
        batch_size=args.bs,
        shuffle=False,
    )

    # trainer
    trainer = Trainer(model=model)

    # model training
    trainer.train(train_loader=train_loader, test_loader=test_loader, epochs=args.epochs, lr=args.lr, save_dir="save/")

    return


if __name__ == "__main__":
    main()
