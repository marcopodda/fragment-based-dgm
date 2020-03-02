import os
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed

from molecules.conversion import (
    mols_from_smiles, mol_to_smiles, mols_to_smiles, canonicalize)
from molecules.fragmentation import fragment_iterative, reconstruct
from molecules.properties import add_property
from molecules.structure import (
    add_atom_counts, add_bond_counts, add_ring_counts)
from utils.config import DATA_DIR, get_dataset_info


def fetch_dataset(name):
    info = get_dataset_info(name)
    filename = Path(info['filename'])
    url = info['url']
    unzip = info['unzip']

    folder = Path("./temp").absolute()
    if not folder.exists():
        os.makedirs(folder)

    os.system(f'wget -P {folder} {url}')

    raw_path = DATA_DIR / name / 'RAW'
    if not raw_path.exists():
        os.makedirs(raw_path)

    processed_path = DATA_DIR / name / 'PROCESSED'
    if not processed_path.exists():
        os.makedirs(processed_path)

    path = folder / filename

    if unzip is True:
        if ".tar.gz" in info['url']:
            os.system(f'tar xvf {path}.tar.gz -C {folder}')
        elif '.zip' in info['url']:
            os.system(f'unzip {path.with_suffix(".zip")} -d {folder}')
        elif '.gz' in info['url']:
            os.system(f'gunzip {path}.gz')

    source = folder / filename
    dest = raw_path / filename

    shutil.move(source, dest)
    shutil.rmtree(folder)


def break_into_fragments(mol, smi):
    frags = fragment_iterative(mol)

    if len(frags) == 0:
        return smi, np.nan, 0

    if len(frags) == 1:
        return smi, smi, 1

    rec, frags = reconstruct(frags)
    if rec and mol_to_smiles(rec) == smi:
        fragments = mols_to_smiles(frags)
        return smi, " ".join(fragments), len(frags)

    return smi, np.nan, 0


def read_and_clean_dataset(info):
    raw_path = DATA_DIR / info['name'] / 'RAW'

    if not raw_path.exists():
        fetch_dataset(info['name'])

    dataset = pd.read_csv(
        raw_path / info['filename'],
        index_col=info['index_col'])

    if info['drop'] != []:
        dataset = dataset.drop(info['drop'], axis=1)

    if info['name'] == 'ZINC':
        dataset = dataset.replace(r'\n', '', regex=True)

    if info['name'] == 'GDB17':
        dataset = dataset.sample(n=info['random_sample'])
        dataset.columns = ['smiles']

    if info['name'] == 'PCBA':
        cols = dataset.columns.str.startswith('PCBA')
        dataset = dataset.loc[:, ~cols]
        dataset = dataset.drop_duplicates()
        dataset = dataset[~dataset.smiles.str.contains("\.")]

    if info['name'] == 'QM9':
        correct_smiles = pd.read_csv(raw_path / 'gdb9_smiles_correct.csv')
        dataset.smiles = correct_smiles.smiles
        dataset = dataset.sample(frac=1, random_state=42)

    smiles = dataset.smiles.tolist()
    dataset.smiles = [canonicalize(smi, clear_stereo=True) for smi in smiles]
    dataset = dataset[dataset.smiles.notnull()].reset_index(drop=True)

    return dataset


def add_fragments(dataset, info, n_jobs):
    smiles = dataset.smiles.tolist()
    mols = mols_from_smiles(smiles)
    pjob = Parallel(n_jobs=n_jobs, verbose=1)
    fun = delayed(break_into_fragments)
    results = pjob(fun(m, s) for m, s in zip(mols, smiles))
    smiles, fragments, lengths = zip(*results)
    dataset["smiles"] = smiles
    dataset["fragments"] = fragments
    dataset["n_fragments"] = lengths

    return dataset


def save_dataset(dataset, info):
    dataset = dataset[info['column_order']]
    testset = dataset[dataset.fragments.notnull()]
    trainset = testset[testset.n_fragments >= info['min_length']]
    trainset = trainset[trainset.n_fragments <= info['max_length']]
    processed_path = DATA_DIR / info['name'] / 'PROCESSED'
    trainset.to_csv(processed_path / 'train.smi')
    dataset.to_csv(processed_path / 'test.smi')


def preprocess_dataset(name, n_jobs):
    info = get_dataset_info(name)
    dataset = read_and_clean_dataset(info)
    dataset = add_atom_counts(dataset, info, n_jobs)
    dataset = add_bond_counts(dataset, info, n_jobs)
    dataset = add_ring_counts(dataset, info, n_jobs)

    for prop in info['properties']:
        if prop not in dataset.columns:
            dataset = add_property(dataset, prop, n_jobs)

    dataset = add_fragments(dataset, info, n_jobs)

    save_dataset(dataset, info)
