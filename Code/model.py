import torch
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn

device = torch.device('cuda:0')


class MLP(torch.nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim, output_dim,
                 feature_pre=True, layer_num=2, dropout=True, ):
        super(MLP, self).__init__()
        self.feature_pre = feature_pre
        self.layer_num = layer_num
        self.dropout = dropout
        if feature_pre:
            self.linear_pre = nn.Linear(input_dim, feature_dim, bias=True)
        else:
            self.linear_first = nn.Linear(input_dim, hidden_dim)
        self.linear_hidden = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for i in range(layer_num - 2)])
        self.linear_out = nn.Linear(feature_dim, output_dim, bias=True)

    def forward(self, data):
        x = data
        if self.feature_pre:
            x = self.linear_pre(x)
        prelu = nn.PReLU().to(device)
        x = prelu(x)
        for i in range(self.layer_num - 2):
            x = self.linear_hidden[i](x)
            x = F.tanh(x)
            if self.dropout:
                x = F.dropout(x, training=self.training)
        x = self.linear_out(x)
        x = F.normalize(x, p=2, dim=-1)
        return x


class Model(nn.Module):
    def __init__(self, data, num_layers=4, hide_dim=256):
        super().__init__()
        self.num_layers = num_layers
        self.data = data
        self.drug_attr = data.drug_attr
        self.drug_kg = data.drug_kg
        self.protein_attr = data.protein_attr
        self.protein_kg = data.protein_kg

        alpha = 1. / (num_layers + 1)
        self.alpha = torch.tensor([alpha] * (num_layers + 1))
        self.lgc = nn.ModuleList([pyg_nn.LGConv() for _ in range(num_layers)])

        self.hide_dim = hide_dim

        self.drug_attr_proj = nn.Linear(64, hide_dim, bias=True)
        self.protein_attr_proj = nn.Linear(64, hide_dim, bias=True)

        self.drug_kg_proj = nn.Linear(200, hide_dim, bias=True)
        self.protein_kg_proj = nn.Linear(200, hide_dim, bias=True)

        nn.init.xavier_normal_(self.drug_attr_proj.weight, )
        nn.init.xavier_normal_(self.protein_attr_proj.weight, )
        nn.init.xavier_normal_(self.drug_kg_proj.weight, )
        nn.init.xavier_normal_(self.protein_kg_proj.weight, )

        self.mlp = MLP(hide_dim, hide_dim * 3, hide_dim // 2, hide_dim * 3)
        self.mlp1 = MLP(hide_dim, hide_dim * 3, hide_dim // 2, hide_dim * 3)
        self.mlp2 = MLP(hide_dim, hide_dim * 3, hide_dim // 2, hide_dim * 3)
        self.mlp3 = MLP(hide_dim, hide_dim * 3, hide_dim // 2, hide_dim * 3)
        self.meta_netu = nn.Linear(hide_dim * 3, hide_dim, bias=True)
        self.meta_neti = nn.Linear(hide_dim * 3, hide_dim, bias=True)

        embedding_size = hide_dim
        self.pred = nn.Sequential(
            # nn.BatchNorm1d(embedding_size * 2),
            nn.Linear(embedding_size * 2, embedding_size),
            nn.Dropout(0.2),
            nn.ReLU(),
            # nn.BatchNorm1d(embedding_size),
            nn.Linear(embedding_size, 1),
            nn.Sigmoid(),
        )
        self.criterion = torch.nn.BCELoss(reduction='mean')
        self.beta = 0.5

    def metafortansform(self, auxiembedu, targetembedu, auxiembedi, targetembedi):
        uneighbor = t.matmul(self.data.DTI_mat, self.g_protein)
        ineighbor = t.matmul(self.data.TDI_mat, self.g_drug)
        tembedu = (self.meta_netu(t.cat((auxiembedu, targetembedu, uneighbor), dim=1).detach()))
        tembedi = (self.meta_neti(t.cat((auxiembedi, targetembedi, ineighbor), dim=1).detach()))
        metau1 = self.mlp(tembedu).reshape(-1, self.hide_dim, 3)
        metau2 = self.mlp1(tembedu).reshape(-1, 3, self.hide_dim)
        metai1 = self.mlp2(tembedi).reshape(-1, self.hide_dim, 3)
        metai2 = self.mlp3(tembedi).reshape(-1, 3, self.hide_dim)
        meta_biasu = (torch.mean(metau1, dim=0))
        meta_biasu1 = (torch.mean(metau2, dim=0))
        meta_biasi = (torch.mean(metai1, dim=0))
        meta_biasi1 = (torch.mean(metai2, dim=0))
        low_weightu1 = F.softmax(metau1 + meta_biasu, dim=1)
        low_weightu2 = F.softmax(metau2 + meta_biasu1, dim=1)
        low_weighti1 = F.softmax(metai1 + meta_biasi, dim=1)
        low_weighti2 = F.softmax(metai2 + meta_biasi1, dim=1)

        tembedus = (t.sum(t.multiply((auxiembedu).unsqueeze(-1), low_weightu1),
                          dim=1))
        tembedus = t.sum(t.multiply((tembedus).unsqueeze(-1), low_weightu2), dim=1)
        tembedis = (t.sum(t.multiply((auxiembedi).unsqueeze(-1), low_weighti1), dim=1))
        tembedis = t.sum(t.multiply((tembedis).unsqueeze(-1), low_weighti2), dim=1)
        transfuEmbed = tembedus
        transfiEmbed = tembedis
        return transfuEmbed, transfiEmbed

    def forward(self, samples, labels):
        drug_attr = self.drug_attr_proj(self.drug_attr)
        protein_attr = self.protein_attr_proj(self.protein_attr)

        x = torch.concat([drug_attr, protein_attr], dim=0)
        out = x * self.alpha[0]

        for i in range(self.num_layers):
            x = self.lgc[i](x, self.data.edge_index)
            out = out + x * self.alpha[i + 1]

        g_drug, g_protein = torch.split(out, [self.data.num_drugs, self.data.num_proteins])
        self.g_drug, self.g_protein = g_drug, g_protein
        kg_drug = self.drug_kg_proj(self.drug_kg)
        kg_protein = self.protein_kg_proj(self.protein_kg)
        metatsuembed, metatsiembed = self.metafortansform(kg_drug, g_drug, kg_protein, g_protein)
        kg_drug = kg_drug + metatsuembed
        kg_protein = kg_protein + metatsiembed
        dr = self.beta * g_drug + (1 - self.beta) * kg_drug
        pr = self.beta * g_protein + (1 - self.beta) * kg_protein

        u_emb = dr[samples[:, 0]]
        v_emb = pr[samples[:, 1]]

        uv_feature = torch.cat((u_emb, v_emb), dim=1)
        out = self.pred(uv_feature)
        out = torch.squeeze(out)

        loss_train = self.criterion(out, labels.float())

        return out, loss_train
