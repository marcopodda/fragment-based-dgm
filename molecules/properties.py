
import pandas as pd
from joblib import Parallel, delayed

from rdkit.Chem import Crippen, QED

from .conversion import mols_from_smiles
from .sascorer.sascorer import calculateScore


def logp(mol):
    return Crippen.MolLogP(mol) if mol else None


def mr(mol):
    return Crippen.MolMR(mol) if mol else None


def qed(mol):
    return QED.qed(mol) if mol else None


def sas(mol):
    return calculateScore(mol) if mol else None


def add_property(dataset, name, n_jobs):
    fn = {"qed": qed, "SAS": sas, "logP": logp, "mr": mr}[name]
    smiles = dataset.smiles.tolist()
    mols = mols_from_smiles(smiles)
    pjob = Parallel(n_jobs=n_jobs, verbose=0)
    prop = pjob(delayed(fn)(mol) for mol in mols)
    new_data = pd.DataFrame(prop, columns=[name])
    return pd.concat([dataset, new_data], axis=1, sort=False)
