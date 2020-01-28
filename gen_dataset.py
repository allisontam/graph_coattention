import json
import numpy as np
import torch.utils.data

from random import shuffle

# [(ppair_drug, ppair_cmpd), etc.]
def train_collate_fn(batch):
    ret_pos = []
    ret_neg = []
    for item in batch: # item is output of __getitem__
        ret_pos.extend(item[0])
        ret_neg.extend(item[1])
    pos_batch = ddi_collate_batch(ret_pos)
    neg_batch = ddi_collate_batch(ret_neg)
    return pos_batch, neg_batch

def test_collate_fn(batch):
    ret_all, truth = list(zip(*batch))
    return (ddi_collate_batch(ret_all), truth)

def ddi_collate_batch(batch):
    """
    Black-box that does all the stuff she emailed about
    """
    drug, cmpd = list(zip(*batch))

    ddi_idxs1, ddi_idxs2 = collate_drug_pairs(drug, cmpd)
    drug = (*collate_drugs(drug), *ddi_idxs1)
    cmpd = (*collate_drugs(cmpd), *ddi_idxs2)

    return (*drug, *cmpd)


def collate_drug_pairs(drugs1, drugs2):
    n_atom1 = [d['n_atom'] for d in drugs1]
    n_atom2 = [d['n_atom'] for d in drugs2]
    c_atom1 = [sum(n_atom1[:k]) for k in range(len(n_atom1))]
    c_atom2 = [sum(n_atom2[:k]) for k in range(len(n_atom2))]

    ddi_seg_i1, ddi_seg_i2, ddi_idx_j1, ddi_idx_j2 = zip(*[
        (i1 + c1, i2 + c2, i2, i1)
        for l1, l2, c1, c2 in zip(n_atom1, n_atom2, c_atom1, c_atom2)
        for i1 in range(l1) for i2 in range(l2)])

    ddi_seg_i1 = torch.LongTensor(ddi_seg_i1)
    ddi_idx_j1 = torch.LongTensor(ddi_idx_j1)

    ddi_seg_i2 = torch.LongTensor(ddi_seg_i2)
    ddi_idx_j2 = torch.LongTensor(ddi_idx_j2)

    return (ddi_seg_i1, ddi_idx_j1), (ddi_seg_i2, ddi_idx_j2)


def collate_drugs(drugs):
    c_atoms = [sum(d['n_atom'] for d in drugs[:k]) for k in range(len(drugs))]

    atom_feat = torch.FloatTensor(np.vstack([d['atom_feat'] for d in drugs]))
    atom_type = torch.LongTensor(np.hstack([d['atom_type'] for d in drugs]))
    bond_type = torch.LongTensor(np.hstack([d['bond_type'] for d in drugs]))
    bond_seg_i = torch.LongTensor(np.hstack([
        np.array(d['bond_seg_i']) + c for d, c in zip(drugs, c_atoms)]))
    bond_idx_j = torch.LongTensor(np.hstack([
        np.array(d['bond_idx_j']) + c for d, c in zip(drugs, c_atoms)]))
    batch_seg_m = torch.LongTensor(np.hstack([
        [k] * d['n_atom'] for k, d in enumerate(drugs)]))

    return batch_seg_m, atom_type, atom_feat, bond_type, bond_seg_i, bond_idx_j

class GenericTestDataset(torch.utils.data.Dataset):
    """
    Only for predicting single properties from pairs

    data: pandas for the intended split
    """
    def __init__(self, data, mol_desc_path):
        if len(data) == 0:
            raise ValueError("Empty dataset")

        self.pairs = list(map(lambda x: tuple(x), data.iterrows()))
        self.drug_struct = {}
        with open(mol_desc_path) as f:
            for l in f:
                idx, l = l.strip().split('\t')
                self.drug_struct[idx] = json.loads(l)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair1, pair2, label = self.pairs[idx]
        cmpds = self.lookup( [(pair1, pair2)] )

        return (cmpds, label)

    def lookup(self, data):
        ret = []
        for pair in data:
            ret.append( (self.drug_struct[pair[0]], self.drug_struct[pair[1]]) )
        return ret

class GenericTrainDataset(GenericTestDataset):
    """
    Only for predicting single properties from pairs

    data: pandas for the intended split
    """
    def __init__(self, data, mol_desc_path, ratioCap=150):
        super().__init__(data, mol_desc_path)
        self.ratioCap = ratioCap
        self.pos_pairs = list(filter(lambda x: x[-1] == 1, self.pairs))
        self.neg_pairs = list(filter(lambda x: x[-1] == 0, self.pairs))
        del self.pairs

        if self.neg2pos_ratio() > self.ratioCap:
            self.pos_pairs *= int(len(self.neg_pairs)/self.ratioCap)
        self.shuffle()

    def shuffle(self):
        shuffle(self.pos_pairs)
        shuffle(self.neg_pairs)

    def neg2pos_ratio(self):
        return 0 if len(self.pos_pairs) == 0 else len(self.neg_pairs)/len(self.pos_pairs)

    def __len__(self):
        return len(self.pos_pairs)

    def __getitem__(self, idx):
        ratio = int(self.neg2pos_ratio())
        ret_pos = self.lookup( [self.pos_pairs[idx]] )

        end_ind = (idx+1)*ratio
        if len(self.neg_pairs) - end_ind < ratio:  # make sure to include all data
            end_ind = len(self.neg_pairs)
        ret_neg = self.lookup( self.neg_pairs[idx*ratio:end_ind] )
        return (ret_pos, ret_neg)
