import copy

import numpy as np
import torch
import torch_geometric.utils as pyg_utils


class Dataset(object):
    def __init__(self, device=torch.device('cuda:0')):
        drug_attr = np.load(f'../Data/features/drug_mf.npy')
        protein_attr = np.load(f'../Data/features/prot_doc2vec.npy')
        num_drugs = len(drug_attr)
        num_proteins = len(protein_attr)
        self.num_drugs = num_drugs
        self.num_proteins = num_proteins

        drug_kg = np.load(f'../Data/DistMult/drug_kg_feat_1.npy')
        protein_kg = np.load(f'../Data/DistMult/protein_kg_feat_1.npy')

        train_pos = np.load(f'../Data/splits/train_pos_1.npy')
        train_neg = np.load(f'../Data/splits/train_neg_1.npy')
        test_pos = np.load(f'../Data/splits/test_pos_1.npy')
        test_neg = np.load(f'../Data/splits/test_neg_1.npy')

        DTI_mat = np.zeros((num_drugs, num_proteins))
        for u, v in train_pos:
            DTI_mat[u, v] = 1
        TDI_mat = DTI_mat.T
        self.DTI_mat = torch.tensor(DTI_mat, dtype=torch.float).to(device)
        self.TDI_mat = torch.tensor(TDI_mat, dtype=torch.float).to(device)

        edge_index = copy.deepcopy(train_pos)
        edge_index[:, 1] += num_drugs
        edge_index = torch.tensor(edge_index.T, dtype=torch.long)
        self.edge_index = pyg_utils.to_undirected(edge_index).to(device)

        train_samples = np.concatenate([train_pos, train_neg])
        train_labels = np.concatenate([np.ones(len(train_pos)), np.zeros(len(train_neg))])
        test_samples = np.concatenate([test_pos, test_neg])
        test_labels = np.concatenate([np.ones(len(test_pos)), np.zeros(len(test_neg))])

        self.train_samples = train_samples
        self.test_samples = test_samples
        self.train_labels = train_labels
        self.test_labels = test_labels

        self.train_labels_th = torch.tensor(train_labels, dtype=torch.long).to(device)
        self.test_labels_th = torch.tensor(test_labels, dtype=torch.long).to(device)

        self.drug_attr = torch.tensor(drug_attr, dtype=torch.float).to(device)
        self.drug_kg = torch.tensor(drug_kg, dtype=torch.float).to(device)
        self.protein_attr = torch.tensor(protein_attr, dtype=torch.float).to(device)
        self.protein_kg = torch.tensor(protein_kg, dtype=torch.float).to(device)

        print('GA-ENs dataset is loaded.')


if __name__ == '__main__':
    Dataset()
