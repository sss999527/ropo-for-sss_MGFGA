import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network, loss
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def tsne_visualization(target_features, target_labels, epoch):
    learning_rates = [0.001, 0.01, 0.1, 1, 10, 50, 100, 200, 500, 1000]
    for lr in learning_rates:
        tsne = TSNE(n_components=2, perplexity=10, learning_rate=lr, random_state=42)
        target_tsne = tsne.fit_transform(target_features)

        plt.figure(figsize=(5, 4))
        for label in np.unique(target_labels):
            indices = np.where(target_labels == label)
            plt.scatter(target_tsne[indices, 0], target_tsne[indices, 1], s=15,
                        alpha=0.7, edgecolors='none')
        plt.axis('off')
        plt.grid(False)
        plt.grid(True, linestyle='--', alpha=0.5)
        # plt.tight_layout()
        plt.savefig('tsne_epoch_{}_lr_{}.png'.format(epoch, lr), dpi=300)
        plt.close()