import time
import numpy as np
import pandas as pd
import itertools
from datetime import datetime

import torch
from torch.autograd import Variable
from torch.nn import functional as F

from molecules.conversion import (
    mols_from_smiles, mols_to_smiles, mol_to_smiles)
from molecules.fragmentation import reconstruct

from utils.config import set_random_seed


def remove_consecutive(fragments):
    return [i for i, _ in itertools.groupby(fragments)]


def generate_molecules(samples, vocab):
    result = []
    num_samples = samples.shape[0]

    for idx in range(num_samples):
        frag_smiles = vocab.translate(samples[idx, :])
        frag_smiles = remove_consecutive(frag_smiles)

        if len(frag_smiles) <= 1:
            continue

        try:
            frag_mols = mols_from_smiles(frag_smiles)
            mol, frags = reconstruct(frag_mols)

            if mol is not None:
                smiles = mol_to_smiles(mol)
                num_frags = len(frags)
                frags = " ".join(mols_to_smiles(frags))
                result.append((smiles, frags, num_frags))
        except Exception:
            continue

    return result


def dump_samples(config, samples):
    columns = ["smiles", "fragments", "n_fragments"]
    df = pd.DataFrame(samples, columns=columns)
    date = datetime.now().strftime('%Y-%m-%d@%X')
    filename = config.path('samples') / f"{date}_{len(samples)}.csv"
    df.to_csv(filename)


class Sampler:
    def __init__(self, config, vocab, model):
        self.config = config
        self.vocab = vocab
        self.model = model

    def sample(self, num_samples, save_results=True, seed=None):
        self.model = self.model.cpu()
        self.model.eval()
        vocab = self.vocab

        hidden_layers = self.model.hidden_layers
        hidden_size = self.model.hidden_size

        def row_filter(row):
            return (row == vocab.EOS).any()
        
        count = 0
        total_time = 0
        batch_size = 100
        samples, sampled = [], 0

        max_length = self.config.get('max_length')
        temperature = self.config.get('temperature')

        seed = set_random_seed()
        self.config.set('sampling_seed', seed)
        print("Sampling seed:", self.config.get('sampling_seed'))

        with torch.no_grad():
            while len(samples) < num_samples:
                start = time.time()

                # sample vector from latent space
                z = self.model.encoder.sample_normal(batch_size)

                # get the initial state
                state = self.model.latent2rnn(z)
                state = state.view(hidden_layers, batch_size, hidden_size)

                # all idx of batch
                sequence_idx = torch.arange(0, batch_size).long()

                # all idx of batch which are still generating
                running = torch.arange(0, batch_size).long()
                sequence_mask = torch.ones(batch_size, dtype=torch.bool)

                # idx of still generating sequences
                # with respect to current loop
                running_seqs = torch.arange(0, batch_size).long()
                lengths = [1] * batch_size

                generated = torch.Tensor(batch_size, max_length).long()
                generated.fill_(vocab.PAD)

                inputs = Variable(torch.Tensor(batch_size).long())
                inputs.fill_(vocab.SOS).long()

                step = 0

                while(step < max_length and len(running_seqs) > 0):
                    inputs = inputs.unsqueeze(1)
                    emb = self.model.embedder(inputs)
                    scores, state = self.model.decoder(emb, state, lengths)
                    scores = scores.squeeze(1)

                    probs = F.softmax(scores / temperature, dim=1)
                    inputs = torch.argmax(probs, 1).reshape(1, -1)

                    # save next input
                    generated = self.update(generated, inputs, running, step)
                    # update global running sequence
                    sequence_mask[running] = (inputs != vocab.EOS)
                    running = sequence_idx.masked_select(sequence_mask)

                    # update local running sequences
                    running_mask = (inputs != vocab.EOS)
                    running_seqs = running_seqs.masked_select(running_mask)

                    # prune input and hidden state according to local update
                    run_length = len(running_seqs)
                    if run_length > 0:
                        inputs = inputs.squeeze(0)
                        inputs = inputs[running_seqs]
                        state = state[:, running_seqs]
                        running_seqs = torch.arange(0, run_length).long()

                    lengths = [1] * run_length
                    step += 1

                new_samples = generated.numpy()
                # print(new_samples)
                mask = np.apply_along_axis(row_filter, 1, new_samples)
                samples += generate_molecules(new_samples[mask], vocab)

                end = time.time() - start
                total_time += end

                if len(samples) > sampled:
                    sampled = len(samples)
                    count = 0 
                else: 
                    count += 1

                if len(samples) % 1000 < 50:
                    elapsed = time.strftime("%H:%M:%S", time.gmtime(end))
                    print(f'Sampled {len(samples)} molecules. '
                          f'Time elapsed: {elapsed}')

                if count >= 100000:
                    break 

        if save_results:
            dump_samples(self.config, samples)

        elapsed = time.strftime("%H:%M:%S", time.gmtime(total_time))
        print(f'Done. Total time elapsed: {elapsed}.')

        set_random_seed(self.config.get('random_seed'))

        return samples

    def update(self, save_to, sample, running_seqs, step):
        # select only still running
        running_latest = save_to[running_seqs]
        # update token at step position
        running_latest[:, step] = sample.data
        # save back
        save_to[running_seqs] = running_latest

        return save_to
