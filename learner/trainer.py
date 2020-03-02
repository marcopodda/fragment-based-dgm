import time
import numpy as np
import pandas as pd

import torch
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from tensorboardX import SummaryWriter


from .model import Loss, Frag2Mol
from .sampler import Sampler
from utils.filesystem import load_dataset
from utils.postprocess import score_samples


SCORES = ["validity", "novelty", "uniqueness"]


def save_ckpt(trainer, epoch, filename):
    path = trainer.config.path('ckpt') / filename
    torch.save({
        'epoch': epoch,
        'best_loss': trainer.best_loss,
        'losses': trainer.losses,
        'best_score': trainer.best_score,
        'scores': trainer.scores,
        'model': trainer.model.state_dict(),
        'optimizer': trainer.optimizer.state_dict(),
        'scheduler': trainer.scheduler.state_dict(),
        'criterion': trainer.criterion.state_dict()
    }, path)


def load_ckpt(trainer, last=False):
    filename = 'last.pt' if last is True else 'best_valid.pt'
    path = trainer.config.path('ckpt') / filename

    if trainer.config.get('use_gpu') is False:
        checkpoint = torch.load(
            path, map_location=lambda storage, loc: storage)
    else:
        checkpoint = torch.load(path)

    print(f"loading {filename} at epoch {checkpoint['epoch']+1}...")

    trainer.model.load_state_dict(checkpoint['model'])
    trainer.optimizer.load_state_dict(checkpoint['optimizer'])
    trainer.scheduler.load_state_dict(checkpoint['scheduler'])
    trainer.criterion.load_state_dict(checkpoint['criterion'])
    trainer.best_loss = checkpoint['best_loss']
    trainer.losses = checkpoint['losses']
    trainer.best_score = checkpoint['best_score']
    trainer.scores = checkpoint['scores']
    return checkpoint['epoch']


def get_optimizer(config, model):
    return Adam(model.parameters(), lr=config.get('optim_lr'))


def get_scheduler(config, optimizer):
    return StepLR(optimizer,
                  step_size=config.get('sched_step_size'),
                  gamma=config.get('sched_gamma'))


def dump(config, losses, scores):
    df = pd.DataFrame(losses, columns=["loss"])
    filename = config.path('performance') / "loss.csv"
    df.to_csv(filename)

    if scores != []:
        df = pd.DataFrame(scores, columns=SCORES)
        filename = config.path('performance') / "scores.csv"
        df.to_csv(filename)


class TBLogger:
    def __init__(self, config):
        self.config = config
        self.writer = SummaryWriter(config.path('tb').as_posix())
        config.write_summary(self.writer)

    def log(self, name, value, epoch):
        self.writer.add_scalar(name, value, epoch)


class Trainer:
    @classmethod
    def load(cls, config, vocab, last):
        trainer = Trainer(config, vocab)
        epoch = load_ckpt(trainer, last=last)
        return trainer, epoch

    def __init__(self, config, vocab):
        self.config = config
        self.vocab = vocab

        self.model = Frag2Mol(config, vocab)
        self.optimizer = get_optimizer(config, self.model)
        self.scheduler = get_scheduler(config, self.optimizer)
        self.criterion = Loss(config, pad=vocab.PAD)

        if self.config.get('use_gpu'):
            self.model = self.model.cuda()

        self.losses = []
        self.best_loss = np.float('inf')
        self.scores = []
        self.best_score = - np.float('inf')

    def _train_epoch(self, epoch, loader):
        self.model.train()
        epoch_loss = 0

        if epoch > 0 and self.config.get('use_scheduler'):
            self.scheduler.step()

        for idx, (src, tgt, lengths) in enumerate(loader):
            self.optimizer.zero_grad()

            src, tgt = Variable(src), Variable(tgt)
            if self.config.get('use_gpu'):
                src = src.cuda()
                tgt = tgt.cuda()

            output, mu, sigma = self.model(src, lengths)

            loss = self.criterion(output, tgt, mu, sigma, epoch)
            loss.backward()
            clip_grad_norm_(self.model.parameters(),
                            self.config.get('clip_norm'))

            epoch_loss += loss.item()

            self.optimizer.step()

        return epoch_loss / len(loader)

    def _valid_epoch(self, epoch, loader):
        use_gpu = self.config.get('use_gpu')
        self.config.set('use_gpu', False)

        num_samples = self.config.get('validation_samples')
        trainer, _ = Trainer.load(self.config, self.vocab, last=True)
        sampler = Sampler(self.config, self.vocab, trainer.model)
        samples = sampler.sample(num_samples, save_results=False)
        dataset = load_dataset(self.config, kind="test")
        _, scores = score_samples(samples, dataset)

        self.config.set('use_gpu', use_gpu)
        return scores

    def log_epoch(self, start_time, epoch, epoch_loss, epoch_scores):
        end = time.time() - start_time
        elapsed = time.strftime("%H:%M:%S", time.gmtime(end))

        print(f'epoch {epoch:06d} - '
              f'loss {epoch_loss:6.4f} - ',
              end=' ')

        if epoch_scores is not None:
            for (name, score) in zip(SCORES, epoch_scores):
                print(f'{name} {score:6.4f} - ', end='')

        print(f'elapsed {elapsed}')

    def train(self, loader, start_epoch):
        num_epochs = self.config.get('num_epochs')

        logger = TBLogger(self.config)

        for epoch in range(start_epoch, start_epoch + num_epochs):
            start = time.time()

            epoch_loss = self._train_epoch(epoch, loader)
            self.losses.append(epoch_loss)
            logger.log('loss', epoch_loss, epoch)
            save_ckpt(self, epoch, filename="last.pt")
            
            if epoch_loss < self.best_loss:
                self.best_loss = epoch_loss
                save_ckpt(self, epoch, filename=f'best_loss.pt')

            epoch_scores = None

            if epoch_loss < self.config.get('validate_after'):
                epoch_scores = self._valid_epoch(epoch, loader)
                self.scores.append(epoch_scores)

                if epoch_scores[2] >= self.best_score:
                    self.best_score = epoch_scores[2]
                    save_ckpt(self, epoch, filename=f'best_valid.pt')

                logger.log('validity', epoch_scores[0], epoch)
                logger.log('novelty', epoch_scores[1], epoch)
                logger.log('uniqueness', epoch_scores[2], epoch)

            self.log_epoch(start, epoch, epoch_loss, epoch_scores)

        dump(self.config, self.losses, self.scores)
