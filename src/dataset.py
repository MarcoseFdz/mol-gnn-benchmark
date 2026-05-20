import os
import urllib.request
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdmolops


import tensorflow as tf

def tf_random_augment(x, adj, mask_rate=0.1, drop_rate=0.1):
    """
    On-the-fly augmentation using TensorFlow primitives.
    """
    # Node attribute masking
    n_nodes = tf.shape(x)[0]
    n_feats = tf.shape(x)[1]
    node_mask = tf.cast(tf.random.uniform((n_nodes, 1)) > mask_rate, tf.float32)
    x_aug = x * node_mask

    # Edge dropout (perturbation)
    # We only drop existing edges. 
    edge_mask = tf.cast(tf.random.uniform(tf.shape(adj)) > drop_rate, tf.float32)
    adj_aug = adj * edge_mask
    
    # Re-normalize adjacency (Simplified GCN-style normalization)
    # Adding self-loops and degree normalization
    adj_aug = adj_aug + tf.eye(n_nodes)
    deg = tf.reduce_sum(adj_aug, axis=-1)
    deg_inv_sqrt = tf.where(deg > 0, tf.math.pow(deg, -0.5), 0.0)
    D_inv_sqrt = tf.linalg.diag(deg_inv_sqrt)
    adj_norm = D_inv_sqrt @ adj_aug @ D_inv_sqrt

    return x_aug, adj_norm

def download_bace():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(base_dir, "data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    file_path = os.path.join(raw_dir, "bace.csv")
    if not os.path.exists(file_path):
        url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv"
        urllib.request.urlretrieve(url, file_path)
    return file_path


def one_hot(val, choices):
    vec = [0.0] * (len(choices) + 1)
    if val in choices:
        vec[choices.index(val)] = 1.0
    else:
        vec[-1] = 1.0
    return vec


def atom_features(atom):
    atom_types  = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'Si', 'B']
    degrees     = [0, 1, 2, 3, 4, 5]
    total_hs    = [0, 1, 2, 3, 4]
    valences    = [0, 1, 2, 3, 4, 5, 6]
    hybrids     = [
        Chem.HybridizationType.SP,
        Chem.HybridizationType.SP2,
        Chem.HybridizationType.SP3,
        Chem.HybridizationType.SP3D,
        Chem.HybridizationType.SP3D2,
    ]
    chiral_tags = [
        Chem.ChiralType.CHI_UNSPECIFIED,
        Chem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
    ]
    feats = []
    feats += one_hot(atom.GetSymbol(), atom_types)
    feats += one_hot(atom.GetDegree(), degrees)
    feats += one_hot(atom.GetTotalNumHs(), total_hs)
    feats += one_hot(atom.GetValence(Chem.ValenceType.IMPLICIT), valences)
    feats += one_hot(atom.GetHybridization(), hybrids)
    feats += one_hot(atom.GetChiralTag(), chiral_tags)
    feats.append(float(atom.GetIsAromatic()))
    feats.append(float(atom.IsInRing()))
    feats.append(float(atom.IsInRingSize(3)))
    feats.append(float(atom.IsInRingSize(4)))
    feats.append(float(atom.IsInRingSize(5)))
    feats.append(float(atom.IsInRingSize(6)))
    feats.append(float(atom.GetFormalCharge()))
    feats.append(float(atom.GetNumRadicalElectrons()))
    return feats


def bond_features(bond):
    bond_types = [
        Chem.BondType.SINGLE,
        Chem.BondType.DOUBLE,
        Chem.BondType.TRIPLE,
        Chem.BondType.AROMATIC,
    ]
    stereo = [
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
        Chem.rdchem.BondStereo.STEREOANY,
    ]
    feats = []
    feats += one_hot(bond.GetBondType(), bond_types)
    feats += one_hot(bond.GetStereo(), stereo)
    feats.append(float(bond.GetIsConjugated()))
    feats.append(float(bond.IsInRing()))
    return feats


ATOM_FEAT_DIM = len(atom_features(Chem.MolFromSmiles("C").GetAtomWithIdx(0)))
BOND_FEAT_DIM = len(bond_features(Chem.MolFromSmiles("CC").GetBondWithIdx(0)))


def mol_to_graph(mol, max_atoms=50):
    if mol is None or mol.GetNumAtoms() > max_atoms:
        return None, None, None

    num_atoms = mol.GetNumAtoms()
    x         = np.zeros((max_atoms, ATOM_FEAT_DIM), dtype=np.float32)
    adj       = np.zeros((max_atoms, max_atoms),      dtype=np.float32)
    edge_feat = np.zeros((max_atoms, max_atoms, BOND_FEAT_DIM), dtype=np.float32)

    for atom in mol.GetAtoms():
        idx    = atom.GetIdx()
        x[idx] = atom_features(atom)

    for bond in mol.GetBonds():
        i  = bond.GetBeginAtomIdx()
        j  = bond.GetEndAtomIdx()
        bf = bond_features(bond)
        adj[i, j] = adj[j, i] = 1.0
        edge_feat[i, j] = edge_feat[j, i] = bf

    for i in range(num_atoms):
        adj[i, i] = 1.0

    d     = np.sum(adj, axis=1)
    d_inv = np.where(d > 0, np.power(d, -0.5), 0.0)
    D_inv = np.diag(d_inv)
    adj   = D_inv @ adj @ D_inv

    return x, adj, edge_feat


def smiles_to_graph(smiles, max_atoms=50):
    return mol_to_graph(Chem.MolFromSmiles(smiles), max_atoms)


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def enumerate_smiles(smiles, n=5, max_atoms=50, rng=None):
    """
    SMILES enumeration: generate up to `n` non-canonical SMILES for the same
    molecule. Each produces a different atom ordering -> different graph, same
    label. This is the strongest valid augmentation for molecular GNNs.
    """
    if rng is None:
        rng = np.random.default_rng()

    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() > max_atoms:
        return []

    num_atoms = mol.GetNumAtoms()
    seen      = set()
    graphs    = []

    attempts = 0
    while len(graphs) < n and attempts < n * 8:
        attempts += 1
        new_order = rng.permutation(num_atoms).tolist()
        new_mol   = rdmolops.RenumberAtoms(mol, new_order)
        smi       = Chem.MolToSmiles(new_mol, canonical=False)
        if smi in seen:
            continue
        seen.add(smi)
        result = mol_to_graph(new_mol, max_atoms)
        if result[0] is not None:
            graphs.append(result)

    return graphs


def node_feature_mask(x, adj, edge_feat, mask_rate=0.1, rng=None):
    """
    Randomly zero out atom features (not the padding rows — only real atoms).
    Leaves graph topology intact. Analogous to feature dropout at data level.
    """
    if rng is None:
        rng = np.random.default_rng()

    x         = x.copy()
    real_mask = np.any(x != 0, axis=-1)
    real_idx  = np.where(real_mask)[0]

    n_mask = max(1, int(len(real_idx) * mask_rate))
    to_mask = rng.choice(real_idx, size=n_mask, replace=False)
    x[to_mask] = 0.0
    return x, adj, edge_feat


def drop_non_ring_edges(x, adj, edge_feat, drop_rate=0.1, rng=None):
    """
    Randomly drop non-ring, non-aromatic edges (safe: these are the only bonds
    whose removal cannot disconnect a ring system or break aromaticity).
    The adjacency is re-normalised after dropping.
    """
    if rng is None:
        rng = np.random.default_rng()

    adj       = adj.copy()
    edge_feat = edge_feat.copy()

    real_mask = np.any(x != 0, axis=-1)
    n         = np.sum(real_mask)

    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] <= 0:
                continue
            in_ring    = bool(edge_feat[i, j, -1])
            is_arom    = bool(edge_feat[i, j, 3])
            if in_ring or is_arom:
                continue
            if rng.random() < drop_rate:
                adj[i, j] = adj[j, i] = 0.0
                edge_feat[i, j] = edge_feat[j, i] = 0.0

    d     = np.sum(adj, axis=1)
    d_inv = np.where(d > 0, np.power(d, -0.5), 0.0)
    D_inv = np.diag(d_inv)
    adj   = D_inv @ adj @ D_inv
    return x, adj, edge_feat


def augment_graph(x, adj, edge_feat, rng, mask_rate=0.1, drop_rate=0.1):
    """Apply node masking and edge dropout together."""
    x, adj, edge_feat = node_feature_mask(x, adj, edge_feat, mask_rate, rng)
    x, adj, edge_feat = drop_non_ring_edges(x, adj, edge_feat, drop_rate, rng)
    return x, adj, edge_feat


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_graphs(
    max_atoms=50,
    num_train=1000, num_val=250, num_test=250,
    augment=True,
    enum_copies=6,
    mask_rate=0.10,
    drop_rate=0.10,
):
    """
    Load BACE, apply stratified split, then augment only the training set via:
      1. SMILES enumeration   (enum_copies extra graphs per molecule)
      2. Node feature masking (mask_rate fraction of real atoms zeroed)
      3. Non-ring edge dropout (drop_rate fraction of acyclic bonds removed)

    Val and test sets are never augmented.
    """
    file_path = download_bace()
    df        = pd.read_csv(file_path)[["mol", "Class"]].dropna()

    rng = np.random.default_rng(42)

    smiles_list, xs, adjs, edges, ys = [], [], [], [], []
    for _, row in df.iterrows():
        x, adj, ef = smiles_to_graph(row["mol"], max_atoms)
        if x is not None:
            smiles_list.append(row["mol"])
            xs.append(x)
            adjs.append(adj)
            edges.append(ef)
            ys.append(int(row["Class"]))

    xs    = np.array(xs,    dtype=np.float32)
    adjs  = np.array(adjs,  dtype=np.float32)
    edges = np.array(edges, dtype=np.float32)
    ys    = np.array(ys,    dtype=np.int32)
    smiles_arr = np.array(smiles_list)

    pos_idx = np.where(ys == 1)[0]
    neg_idx = np.where(ys == 0)[0]
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    def split_class(idx, n_tr, n_va, n_te):
        ratio = len(idx) / (num_train + num_val + num_test)
        nt    = round(n_tr * ratio)
        nv    = round(n_va * ratio)
        ntt   = round(n_te * ratio)
        return idx[:nt], idx[nt:nt+nv], idx[nt+nv:nt+nv+ntt]

    p_tr, p_va, p_te = split_class(pos_idx, num_train, num_val, num_test)
    n_tr, n_va, n_te = split_class(neg_idx, num_train, num_val, num_test)

    train_idx = np.concatenate([p_tr, n_tr])
    val_idx   = np.concatenate([p_va, n_va])
    test_idx  = np.concatenate([p_te, n_te])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    def sel(arr, idx): return arr[idx]

    x_val,   adj_val,   edge_val,   y_val   = sel(xs,val_idx),   sel(adjs,val_idx),   sel(edges,val_idx),   sel(ys,val_idx)
    x_test,  adj_test,  edge_test,  y_test  = sel(xs,test_idx),  sel(adjs,test_idx),  sel(edges,test_idx),  sel(ys,test_idx)

    x_train_base    = sel(xs,         train_idx)
    adj_train_base  = sel(adjs,       train_idx)
    edge_train_base = sel(edges,      train_idx)
    y_train_base    = sel(ys,         train_idx)
    smi_train       = sel(smiles_arr, train_idx)

    if not augment:
        return (
            x_train_base, adj_train_base, edge_train_base, y_train_base,
            x_val, adj_val, edge_val, y_val,
            x_test, adj_test, edge_test, y_test,
        )

    aug_xs, aug_adjs, aug_edges, aug_ys = (
        list(x_train_base), list(adj_train_base),
        list(edge_train_base), list(y_train_base)
    )

    print(f"Base training set: {len(y_train_base)} molecules")
    enum_added = 0
    for i, smi in enumerate(smi_train):
        label   = y_train_base[i]
        copies  = enumerate_smiles(smi, n=enum_copies, max_atoms=max_atoms, rng=rng)
        for x_c, adj_c, ef_c in copies:
            aug_xs.append(x_c)
            aug_adjs.append(adj_c)
            aug_edges.append(ef_c)
            aug_ys.append(label)
            enum_added += 1

    print(f"After SMILES enumeration: +{enum_added} → {len(aug_ys)} total")

    mask_xs, mask_adjs, mask_edges, mask_ys = [], [], [], []
    for i in range(len(x_train_base)):
        x_m, adj_m, ef_m = augment_graph(
            x_train_base[i], adj_train_base[i], edge_train_base[i],
            rng, mask_rate=mask_rate, drop_rate=drop_rate
        )
        mask_xs.append(x_m)
        mask_adjs.append(adj_m)
        mask_edges.append(ef_m)
        mask_ys.append(y_train_base[i])

    aug_xs    += mask_xs
    aug_adjs  += mask_adjs
    aug_edges += mask_edges
    aug_ys    += mask_ys
    print(f"After node/edge augmentation: +{len(mask_ys)} → {len(aug_ys)} total")

    aug_xs    = np.array(aug_xs,    dtype=np.float32)
    aug_adjs  = np.array(aug_adjs,  dtype=np.float32)
    aug_edges = np.array(aug_edges, dtype=np.float32)
    aug_ys    = np.array(aug_ys,    dtype=np.int32)

    perm = rng.permutation(len(aug_ys))
    aug_xs, aug_adjs, aug_edges, aug_ys = (
        aug_xs[perm], aug_adjs[perm], aug_edges[perm], aug_ys[perm]
    )

    return (
        aug_xs, aug_adjs, aug_edges, aug_ys,
        x_val, adj_val, edge_val, y_val,
        x_test, adj_test, edge_test, y_test,
    )
