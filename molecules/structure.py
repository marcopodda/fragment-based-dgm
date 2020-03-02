import copy
import pandas as pd

from collections import OrderedDict
from joblib import Parallel, delayed
from rdkit import Chem

from .conversion import mols_from_smiles


def _ordered_dict(lst):
    return OrderedDict(zip(lst, [0] * len(lst)))


def count_atoms(mol, atomlist):
    count = _ordered_dict(atomlist)
    if mol:
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            if symbol not in count:
                count["Other"] += 1
            else:
                count[symbol] += 1
    return count


def count_bonds(mol, bondlist):
    count = _ordered_dict(bondlist)
    if mol:
        mol = copy.deepcopy(mol)
        Chem.Kekulize(mol, clearAromaticFlags=True)
        for bond in mol.GetBonds():
            count[str(bond.GetBondType())] += 1
    return count


def count_rings(mol, ringlist):
    ring_sizes = {i: r for (i, r) in zip(range(3, 7), ringlist)}
    count = _ordered_dict(ringlist)
    if mol:
        ring_info = Chem.GetSymmSSSR(mol)
        for ring in ring_info:
            ring_length = len(list(ring))
            if ring_length in ring_sizes:
                ring_name = ring_sizes[ring_length]
                count[ring_name] += 1
    return count


def _add_counts(dataset, fn, names, n_jobs):
    smiles = dataset.smiles.tolist()
    mols = mols_from_smiles(smiles)
    pjob = Parallel(n_jobs=n_jobs, verbose=0)
    counts = pjob(delayed(fn)(mol, names) for mol in mols)
    return pd.concat([dataset, pd.DataFrame(counts)], axis=1, sort=False)


def add_atom_counts(dataset, info, n_jobs):
    return _add_counts(dataset, count_atoms, info['atoms'], n_jobs)


def add_bond_counts(dataset, info, n_jobs):
    return _add_counts(dataset, count_bonds, info['bonds'], n_jobs)


def add_ring_counts(dataset, info, n_jobs):
    return _add_counts(dataset, count_rings, info['rings'], n_jobs)
