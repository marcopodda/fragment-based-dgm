from learner.dataset import FragmentDataset
from learner.sampler import Sampler
from learner.trainer import Trainer, save_ckpt
from utils.config import Config
from utils.parser import command_parser
from utils.plots import plot_paper_figures
from utils.preprocess import preprocess_dataset
from utils.postprocess import postprocess_samples, score_samples, dump_scores
from utils.filesystem import load_dataset


from rdkit import rdBase
rdBase.DisableLog('rdApp.*')


def train_model(config):
    dataset = FragmentDataset(config)
    vocab = dataset.get_vocab()
    trainer = Trainer(config, vocab)
    trainer.train(dataset.get_loader(), 0)


def resume_model(config):
    dataset = FragmentDataset(config)
    vocab = dataset.get_vocab()
    load_last = config.get('load_last')
    trainer, epoch = Trainer.load(config, vocab, last=load_last)
    trainer.train(dataset.get_loader(), epoch + 1)


def sample_model(config):
    dataset = FragmentDataset(config)
    vocab = dataset.get_vocab()
    load_last = config.get('load_last')
    trainer, epoch = Trainer.load(config, vocab, last=load_last)
    sampler = Sampler(config, vocab, trainer.model)
    seed = config.get('sampling_seed') if config.get('reproduce') else None
    samples = sampler.sample(config.get('num_samples'), seed=seed)
    dataset = load_dataset(config, kind="test")
    _, scores = score_samples(samples, dataset)
    is_max = dump_scores(config, scores, epoch)
    if is_max:
        save_ckpt(trainer, epoch, filename=f"best.pt")
    config.save()


if __name__ == "__main__":
    parser = command_parser()
    args = vars(parser.parse_args())
    command = args.pop('command')

    if command == 'preprocess':
        dataset = args.pop('dataset')
        n_jobs = args.pop('n_jobs')
        preprocess_dataset(dataset, n_jobs)

    elif command == 'train':
        config = Config(args.pop('dataset'), **args)
        train_model(config)

    elif command == 'resume':
        run_dir = args.pop('run_dir')
        config = Config.load(run_dir, **args)
        resume_model(config)

    elif command == 'sample':
        args.update(use_gpu=False)
        run_dir = args.pop('run_dir')
        config = Config.load(run_dir, **args)
        sample_model(config)

    elif command == 'postprocess':
        run_dir = args.pop('run_dir')
        config = Config.load(run_dir, **args)
        postprocess_samples(config, **args)
    
    elif command == 'plot':
        run_dir = args.pop('run_dir')
        plot_paper_figures(run_dir)
