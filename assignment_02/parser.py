# @Author  : Peizhao Li
# @Contact : peizhaoli05@gmail.com

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="SphereFace")

    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--device', type=str, default="cuda:0")

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=7e-4)
    parser.add_argument('--eval_interval', type=int, default=4)

    parser.add_argument('--train_file', type=str, default="assignment_02/data/pairsDevTrain.txt")
    parser.add_argument('--eval_file', type=str, default="assignment_02/data/pairsDevTest.txt")
    parser.add_argument('--img_folder', type=str, default="assignment_02/data/lfw")

    return parser.parse_args()
