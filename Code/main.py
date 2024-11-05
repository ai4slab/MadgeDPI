import random

import numpy as np
import torch

from dataset import Dataset
from model import Model
from utils import plot_auc

if __name__ == '__main__':
    seed = 2023
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device('cuda:0')

    data = Dataset()
    model = Model(data, num_layers=6, hide_dim=256).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)

    for epoch in range(1, 301):
        model.train()
        optimizer.zero_grad()
        _, loss = model(data.train_samples, data.train_labels_th)
        loss.backward()
        optimizer.step()
        print('Epoch: ', epoch, ' --- ', loss.item())

        model.eval()
        output, _ = model(data.test_samples, data.test_labels_th)

        if epoch % 50 == 0:
            plot_auc(data.test_labels, output.cpu().detach().numpy(), epoch)
