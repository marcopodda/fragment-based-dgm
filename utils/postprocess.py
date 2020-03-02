import time
import numpy as np
import pandas as pd

from molecules.conversion import mol_from_smiles
from molecules.properties import add_property
from molecules.structure import (
    add_atom_counts, add_bond_counts, add_ring_counts)

from .config import get_dataset_info
from .filesystem import load_dataset

SCORES = ["validity", "novelty", "uniqueness"]


def dump_scores(config, scores, epoch):
    filename = config.path('performance') / "scores.csv"
    df = pd.DataFrame([scores], columns=SCORES)

    if not filename.exists():
        df.to_csv(filename)
        is_max = True
    else:
        ref = pd.read_csv(filename, index_col=0)
        is_max = scores[2] >= ref.uniqueness.max()
        ref = pd.concat([ref, df], axis=0, sort=False, ignore_index=True)
        ref.to_csv(filename)

    return is_max


def retrieve_samples(config):
    dfs = []
    filenames = config.path('samples').glob('*_*.csv')

    for filename in filenames:
        dfs.append(pd.read_csv(filename, index_col=0))

    samples = pd.concat(dfs, axis=0, ignore_index=True, sort=False)
    samples = samples.reset_index(drop=True)
    return samples.copy()


def mask_valid_molecules(smiles):
    valid_mask = []

    for smi in smiles:
        try:
            mol = mol_from_smiles(smi)
            valid_mask.append(mol is not None)
        except Exception:
            valid_mask.append(False)

    return np.array(valid_mask)


def mask_novel_molecules(smiles, data_smiles):
    novel_mask = []

    for smi in smiles:
        novel_mask.append(smi not in data_smiles)

    return np.array(novel_mask)


def mask_unique_molecules(smiles):
    uniques, unique_mask = set(), []

    for smi in smiles:
        unique_mask.append(smi not in uniques)
        uniques.add(smi)

    return np.array(unique_mask)


def score_samples(samples, dataset, calc=True):
    def ratio(mask):
        total = mask.shape[0]
        if total == 0:
            return 0.0
        return mask.sum() / total

    if isinstance(samples, pd.DataFrame):
        smiles = samples.smiles.tolist()
    elif isinstance(samples, list):
        smiles = [s[0] for s in samples]
    data_smiles = dataset.smiles.tolist()

    valid_mask = mask_valid_molecules(smiles)
    novel_mask = mask_novel_molecules(smiles, data_smiles)
    unique_mask = mask_unique_molecules(smiles)

    scores = []
    if calc:
        start = time.time()
        print("Start scoring...")
        validity_score = ratio(valid_mask)
        novelty_score = ratio(novel_mask[valid_mask])
        uniqueness_score = ratio(unique_mask[valid_mask])

        print(f"valid: {validity_score} - "
              f"novel: {novelty_score} - "
              f"unique: {uniqueness_score}")

        scores = [validity_score, novelty_score, uniqueness_score]
        end = time.time() - start
        elapsed = time.strftime("%H:%M:%S", time.gmtime(end))
        print(f'Done. Time elapsed: {elapsed}.')

    return valid_mask * novel_mask * unique_mask, scores


def postprocess_samples(config, use_train=False, n_jobs=-1):
    start = time.time()
    print("Start postprocessing...", end=" ")
    kind = 'train' if use_train else 'test'
    dataset = load_dataset(config, kind=kind)
    samples = retrieve_samples(config)

    mask, _ = score_samples(samples, dataset, calc=False)
    samples = samples.iloc[mask, :].reset_index(drop=True)

    info = get_dataset_info(config.get('dataset'))
    samples = add_atom_counts(samples, info, n_jobs)
    samples = add_bond_counts(samples, info, n_jobs)
    samples = add_ring_counts(samples, info, n_jobs)

    for prop in info['properties']:
        samples = add_property(samples, prop, n_jobs)

    samples = samples[info['column_order']]
    samples['who'] = 'OURS'
    dataset['who'] = info['name']

    data = [samples, dataset]
    aggregated = pd.concat(data, axis=0, ignore_index=True, sort=False)
    aggregated.to_csv(config.path('samples') / 'aggregated.csv')

    end = time.time() - start
    elapsed = time.strftime("%H:%M:%S", time.gmtime(end))
    print(f'Done. Time elapsed: {elapsed}.')
