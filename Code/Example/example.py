import math
import pandas as pd
from rdkit import Chem
from collections import defaultdict
from sklearn.metrics import pairwise_distances
from example_model import DeepEnzyme
import torch
import pickle
import numpy as np
import scipy.sparse as sparse

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def split_sequence(sequence, ngram, word_dict):
    sequence = '--' + sequence + '='

    words = list()
    for i in range(len(sequence) - ngram + 1):
        try:
            words.append(word_dict[sequence[i:i + ngram]])
        except:
            word_dict[sequence[i:i + ngram]] = 0
            words.append(word_dict[sequence[i:i + ngram]])

    return np.array(words)


def create_atoms(mol, atom_dict):
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    for a in mol.GetAromaticAtoms():
        i = a.GetIdx()
        atoms[i] = (atoms[i], 'aromatic')
    atoms = [atom_dict[a] for a in atoms]
    return np.array(atoms)


def create_ijbonddict(mol, bond_dict):
    i_jbond_dict = defaultdict(lambda: [])
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bond = bond_dict[str(b.GetBondType())]
        i_jbond_dict[i].append((j, bond))
        i_jbond_dict[j].append((i, bond))
    return i_jbond_dict


def extract_fingerprints(atoms, i_jbond_dict, radius, fingerprint_dict, edge_dict):
    if (len(atoms) == 1) or (radius == 0):
        fingerprints = [fingerprint_dict[a] for a in atoms]
    else:
        nodes = atoms
        i_jedge_dict = i_jbond_dict

        for _ in range(radius):
            fingerprints = []
            for i, j_edge in i_jedge_dict.items():
                neighbors = [(nodes[j], edge) for j, edge in j_edge]
                fingerprint = (nodes[i], tuple(sorted(neighbors)))
                try:
                    fingerprints.append(fingerprint_dict[fingerprint])
                except:
                    fingerprint_dict[fingerprint] = 0
                    fingerprints.append(fingerprint_dict[fingerprint])
            nodes = fingerprints

            _i_jedge_dict = defaultdict(lambda: [])
            for i, j_edge in i_jedge_dict.items():
                for j, edge in j_edge:
                    both_side = tuple(sorted((nodes[i], nodes[j])))
                    try:
                        edge = edge_dict[(both_side, edge)]
                    except:
                        edge_dict[(both_side, edge)] = 0
                        edge = edge_dict[(both_side, edge)]

                    _i_jedge_dict[i].append((j, edge))
            i_jedge_dict = _i_jedge_dict

    return np.array(fingerprints)


def create_adjacency(mol):
    adjacency = Chem.GetAdjacencyMatrix(mol)
    return np.array(adjacency)


def get_ca_coords(pdb):
    with open(pdb, 'r') as file:
        lines = file.readlines()
        file.close()

    out = []

    for line in lines:
        if line.startswith('ATOM ') and line.split()[4] == 'A' and line.split()[2] == 'CA':
            res_num = line.split()[5]
            res_name = line.split()[3]
            x = line.split()[6]
            y = line.split()[7]
            z = line.split()[8]
            if len(x) > int(8):
                x = line.split()[6][:-8]
                y = line.split()[6][-8:]
                z = line.split()[7]
            elif len(y) > int(8):
                x = line.split()[6]
                y = line.split()[7][:-8]
                z = line.split()[7][-8:]
            elif len(res_num) > int(4):
                x = line.split()[5][-8:]
                y = line.split()[6]
                z = line.split()[7]
                res_num = line.split()[5][:-8]

            out.append([res_num, res_name, x, y, z])

    df = pd.DataFrame(out, columns=['res_num', 'res_name', 'x', 'y', 'z'])

    return df


def luciferase_contact_map(pdb,seq):
    ca_coords = get_ca_coords(pdb)
    dist_arr = pairwise_distances(ca_coords[['x', 'y', 'z']].values)  # distance
    dist_tensor = torch.from_numpy(dist_arr)
    dist_thres = 10
    cont_arr = (dist_arr < dist_thres).astype(int)
    cont_tensor = torch.from_numpy(cont_arr)
    if cont_arr.shape[0] == len(seq):
        proteinadjacency = sparse.csr_matrix(cont_arr)
    else:
        a = np.zeros((cont_arr.shape[0], len(seq) - cont_arr.shape[0]))
        cont_arr = np.column_stack((cont_arr, a))
        b = np.zeros((len(seq) - cont_arr.shape[0], len(seq)))
        cont_arr = np.row_stack((cont_arr, b))
        row, col = np.diag_indices_from(cont_arr)
        cont_arr[row, col] = 1
        proteinadjacency = sparse.csr_matrix(cont_arr)
    return proteinadjacency


def main():
    dim = 64
    layer_output = 3
    hidden_dim1 = 64
    hidden_dim2 = 128
    dropout = 0
    nhead = 4
    hid_size = 64
    layers_trans = 3
    radius = 2
    ngram = 4

    dir_input = '../../../DeepEnzyme/Data/Input/'
    sequence = '"MPFGNTHNKFKLNYKPEEEYPDLSKHNNHMAKVLTLELYKKLRDKETPSGFTVDDVIQTGVDNPGHPFIMTVGCVAGDEESYEVFKELFDPIISDRHGGYKPTDKHKTDLNHENLKGGDDLDPNYVLSSRVRTGRSIKGYTLPPHCSRGERRAVEKLSVEALNSLTGEFKGKYYPLKSMTEKEQQQLIDDHFLFDKPVSPLLLASGMARDWPDARGIWHNDNKSFLVWVNEEDHLRVISMEKGGNMKEVFRRFCVGLQKIEEIFKKAGHPFMWNQHLGYVLTCPSNLGTGLRGGVHVKLAHLSTHPKFEEILTRLRLQKRGTGGVDTAAVGSVFDVSNADRLGSSEVEQVQLVVDGVKLMVEMEKKLEKGQSIDDMIPAQK'
    smiles = 'CN(CC(=O)O)C(=N)N'
    structure = '../../../DeepEnzyme/Data/Input/2.7.3.2_Homo_sapiens_Seq_5476_relaxed_rank_1_model_3.pdb'
    atom_dict = load_pickle(dir_input + 'atom_dict_0612.pickle')
    bond_dict = load_pickle(dir_input + 'bond_dict_0612.pickle')
    edge_dict = load_pickle(dir_input + 'edge_dict_0612.pickle')
    fingerprint_dict = load_pickle(dir_input + 'fingerprint_dict_0612.pickle')
    word_dict = load_pickle(dir_input + 'sequence_dict_0612.pickle')
    n_fingerprint = len(fingerprint_dict)
    n_word = len(word_dict)

    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    atoms = create_atoms(mol, atom_dict)
    i_jbond_dict = create_ijbonddict(mol, bond_dict)
    fingerprints = extract_fingerprints(atoms, i_jbond_dict, radius, fingerprint_dict, edge_dict)
    fingerprints = torch.LongTensor(fingerprints).to(device)
    smilesadjacency = create_adjacency(mol)
    smilesadjacency = torch.FloatTensor(smilesadjacency).to(device)
    words = split_sequence(sequence, ngram, word_dict)
    words = torch.LongTensor(words).to(device)
    proteinadjacency = luciferase_contact_map(structure, sequence)

    inputs = [fingerprints, smilesadjacency, words, proteinadjacency]

    model = DeepEnzyme(n_fingerprint, dim, n_word, layer_output, hidden_dim1, hidden_dim2, dropout, nhead, hid_size,
                     layers_trans).to(device)
    #example can be downloaded from https://figshare.com/articles/dataset/DeepEnzyme/25771062
    model.load_state_dict(torch.load('../../../DeepEnzyme/example'),
                          strict=False)
    model.train(False)

    prediction = model.forward(inputs, layer_output, dropout)
    Kcat_log2_value = prediction.item()
    Kcat_value = '%.4f' % math.pow(2, Kcat_log2_value)

    print('Predict Kcat value is: ' + Kcat_value)


if __name__ == '__main__':
    main()
