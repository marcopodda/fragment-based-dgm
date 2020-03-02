from rdkit import Chem
from rdkit.Chem import rdmolops


def mol_to_smiles(mol):
    smi = Chem.MolToSmiles(mol, isomericSmiles=True)
    return canonicalize(smi)


def mol_from_smiles(smi):
    smi = canonicalize(smi)
    return Chem.MolFromSmiles(smi)


def canonicalize(smi, clear_stereo=False):
    mol = Chem.MolFromSmiles(smi)
    if clear_stereo:
        Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def mols_to_smiles(mols):
    return [mol_to_smiles(m) for m in mols]


def mols_from_smiles(mols):
    return [mol_from_smiles(m) for m in mols]


def mol_to_graph_data(mol):
    A = rdmolops.GetAdjacencyMatrix(mol)
    node_features, edge_features = {}, {}

    bondidxs = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx())
                for b in mol.GetBonds()]

    for idx in range(A.shape[0]):
        atomic_num = mol.GetAtomWithIdx(idx).GetAtomicNum()
        node_features[idx]["label"] = int(atomic_num)

    for b1, b2 in bondidxs:
        btype = mol.GetBondBetweenAtoms(b1, b2).GetBondTypeAsDouble()
        edge_features[(b1, b2)]["label"] = int(btype)

    return A, node_features, edge_features
