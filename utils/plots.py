import os
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns


sns.set('paper')
sns.set_style('whitegrid', {'axes.grid': False})
params = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica'],
    'legend.fontsize': 'x-small',
    'legend.handlelength': 1,
    'legend.handletextpad': 0.2,
    'legend.columnspacing': 0.8,
    'xtick.labelsize': 'x-small',
    'ytick.labelsize': 'x-small'}
plt.rcParams.update(params)

ratio = 0.4

props = ['qed', 'SAS', 'logP']

feats = {
    'atoms': ['C', 'F', 'N', 'O', 'Other'],
    'bonds': ['SINGLE', 'DOUBLE', 'TRIPLE'],
    'rings': ['Tri', 'Quad', 'Pent', 'Hex']
}

MODEL = 'OURS'

def plot_property(df, name, prop, ax=None):
    new_names = dict([(p, p.upper()) for p in props])
    df.rename(columns=new_names, inplace=True)
    sns.distplot(df[prop.upper()][df.who==name], hist=False, label=name, ax=ax)
    ax = sns.distplot(df[prop.upper()][df.who==MODEL], hist=False, label=MODEL, ax=ax)
    ax.set_aspect(1.0/ax.get_data_ratio()*ratio)


def plot_count(df, name, feat, ax=None):
    s1 = df[feats[feat]][df.who==name].mean(axis=0)
    s2 = df[feats[feat]][df.who==MODEL].mean(axis=0)
    data = pd.DataFrame([s1, s2], index=[name, MODEL])
    ax = data.plot(kind='bar', stacked=True, ax=ax, rot=0)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.9),
          ncol=len(feats[feat]), framealpha=0, borderpad=1, title=feat.upper())


def plot_counts(df, dataset_name):
    fig, axs = plt.subplots(1, 3)
    for i, f in enumerate(feats):
        plot_count(df, dataset_name, f, ax=axs.flat[i])
        axs.flat[i].set_aspect(1.0/axs.flat[i].get_data_ratio()*ratio)
    fig.savefig(f'counts_{dataset_name}.svg')


def plot_props(df, dataset_name):
    fig, axs = plt.subplots(1, 3)
    for i, p in enumerate(props):
        plot_property(df, dataset_name, p, ax=axs.flat[i])
        axs.flat[i].set_aspect(1.0/axs.flat[i].get_data_ratio()*ratio)
    fig.savefig(f'props_{dataset_name}.svg')


def plot_paper_figures(run_dir):
    dataset_name = "ZINC" if "ZINC" in run_dir else "PCBA"
    df = pd.read_csv(os.path.join(run_dir, 'results/samples/aggregated.csv'))
    plot_counts(df, dataset_name)
    plot_props(df, dataset_name)

