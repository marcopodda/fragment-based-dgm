import argparse


def command_parser():
    parser = argparse.ArgumentParser()

    subps = parser.add_subparsers()

    subps_train = subps.add_parser(
        'train',
        help='train from scratch.')
    subps_train.add_argument(
        "--dataset",
        choices=['ZINC', 'PCBA'],
        help="dataset name.")
    subps_train.add_argument(
        '--title',
        help="title of the experiment",
        default='Molecule Generator')
    subps_train.add_argument(
        '--description',
        help="description of the experiment",
        default='An RNN-based Molecule Generator')
    subps_train.add_argument(
        '--random_seed',
        type=int, default=42,
        help="seed for random number generation")
    subps_train.add_argument(
        '--use_gpu',
        action='store_true',
        help="use GPU")
    subps_train.add_argument(
        '--mask_freq',
        type=int, default=2,
        help="masking frequency")
    subps_train.add_argument(
        '--num_clusters',
        type=int, default=5,
        help="number of low frequency clusters.")
    subps_train.add_argument(
        '--batch_size',
        type=int, default=32,
        help='batch size')
    subps_train.add_argument(
        '--no_shuffling', dest='shuffle',
        action='store_false',
        help="don't shuffle batches")
    subps_train.add_argument(
        '--no_mask', dest='use_mask',
        action='store_false',
        help="don't mask low-drequency fragments")
    subps_train.add_argument(
        '--num_epochs',
        default=9, type=int,
        help='number of epochs to train')
    subps_train.add_argument(
        '--embed_size',
        default=64, type=int,
        help='size of the embedding layer')
    subps_train.add_argument(
        '--embed_window',
        default=3, type=int,
        help='window for word2vec embedding')
    subps_train.add_argument(
        '--hidden_size',
        default=64, type=int,
        help='number of recurrent neurons per layer')
    subps_train.add_argument(
        '--hidden_layers',
        default=2, type=int,
        help='number of recurrent layers')
    subps_train.add_argument(
        '--dropout',
        default=0.3, type=float,
        help='dropout for the recurrent layers')
    subps_train.add_argument(
        '--latent_size',
        default=100, type=int,
        help='size of the VAE latent space')
    subps_train.add_argument(
        '--lr', dest='optim_lr',
        default=0.00001, type=float,
        help='learning rate')
    subps_train.add_argument(
        '--no_scheduler', dest='use_scheduler',
        action='store_false',
        help="don't use learning rate scheduler")
    subps_train.add_argument(
        '--step_size', dest='sched_step_size',
        default=2, type=int,
        help='annealing step size for the scheduler')
    subps_train.add_argument(
        '--gamma', dest='sched_gamma',
        default=0.9, type=float,
        help='annealing rate for the scheduler')
    subps_train.add_argument(
        '--clip_norm',
        default=5.0, type=float,
        help='threshold to clip the gradient norm')
    subps_train.add_argument(
        '--validate_after',
        default=0.0, type=float,
        help='threshold below which validation starts')
    subps_train.add_argument(
        '--validation_samples',
        default=100, type=int,
        help='number of validation samples')
    subps_train.set_defaults(command='train')

    subps_resume = subps.add_parser(
        'resume',
        help='resume training.')
    subps_resume.add_argument(
        "--run_dir", metavar="FOLDER",
        help="directory of the run to resume.")
    subps_resume.add_argument(
        '--num_epochs',
        default=10, type=int,
        help='stop at this epoch number')
    subps_resume.add_argument(
        '--load_last', action="store_true",
        help='load last model instead of best')
    subps_resume.set_defaults(command='resume')

    subps_sample = subps.add_parser(
        'sample',
        help='reload model and sample')
    subps_sample.add_argument(
        "--run_dir", metavar="FOLDER",
        help="directory of the run to resume")
    subps_sample.add_argument(
        '--load_last', action="store_true",
        help='load last model instead of best')
    subps_sample.add_argument(
        '--num_samples',
        default=20000, type=int,
        help='number of samples to draw from the model')
    subps_sample.add_argument(
        '--max_length',
        default=10, type=int,
        help='maximum length of the sampled sequence')
    subps_sample.add_argument(
        '--temperature',
        default=1.0, type=float,
        help='sampling temperature')
    subps_sample.add_argument(
        '--reproduce', action="store_true", default=False,
        help='reproducible sampling')
    subps_sample.set_defaults(command='sample')

    subps_postprocess = subps.add_parser(
        'postprocess',
        help='postprocess results.')
    subps_postprocess.add_argument(
        "--run_dir", metavar="FOLDER",
        help="directory of the run to resume.")
    subps_postprocess.add_argument(
        '--use_train', action="store_true",
        help='use the training set to calculate scores')
    subps_postprocess.add_argument(
        '--n_jobs', default=-1, type=int,
        help='number of parallel jobs')
    subps_postprocess.set_defaults(command='postprocess')

    subps_preprocess = subps.add_parser(
        'preprocess',
        help='preprocess (download & clean) datsets.')
    subps_preprocess.add_argument(
        '--dataset', default="ZINC",
        choices=['ZINC', 'PCBA'],
        help='dataset name.')
    subps_preprocess.add_argument(
        '--n_jobs', default=-1, type=int,
        help='number of parallel jobs')
    subps_preprocess.set_defaults(command='preprocess')

    subps_plot = subps.add_parser(
        'plot',
        help='plot paper figure.')
    subps_plot.add_argument(
        "--run_dir", metavar="FOLDER",
        help="directory of the run.")
    subps_plot.set_defaults(command='plot')

    return parser
