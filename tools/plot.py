from argparse import ArgumentParser
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import csv

from rlkit.launchers import config
PROJECT_DIR = config.rlkit_project_dir
LOG_DIR = config.LOCAL_LOG_DIR

LOG_MAP = {
    'return': 'AverageReturn',
    'epoch': 'Epoch',
    'step': 'Number of train steps total',
    'time': 'Total Train Time (s)',
}


def get_index_from_csv_head(desc, name):
    try:
        return desc.index(name)
    except:
        print('%s doesn\'t exist in progress csv!' % name)
        raise


def plot_rewards(progress_csvs, env_name, save_dir,
    value='return', by='epoch', do_fit=False, fit_order=6):

    ax = plt.gca()

    exp_names = []
    for progress in progress_csvs:
        desc, data, name = progress['desc'], progress['data'], progress['name']
        exp_names.append(name)

        x = data[:, desc.index(LOG_MAP[by])]

        value_idx = get_index_from_csv_head(desc, LOG_MAP[value])
        y = data[:, value_idx]

        color = next(ax._get_lines.prop_cycler)['color']
        if do_fit:
            plt.plot(x, y, color=color, alpha=0.5)
            fit = np.polyfit(x, y, fit_order)
            fit = np.polyval(fit, x)
            plt.plot(x, fit, lw=2, label=name, color=color)
        else:
            plt.plot(x, y, label=name)

    plt.ticklabel_format(style='sci', scilimits=(-1e3, 1e3), axis='x')
    plt.ticklabel_format(style='sci', scilimits=(-1e5, 1e5), axis='y')
    plt.legend(loc='lower right')
    plt.xlabel(LOG_MAP[by])
    plt.ylabel(LOG_MAP[value])
    plt.grid(True)
    plt.title('Training Result on ' + env_name)

    os.makedirs(save_dir, exist_ok=True)
    env_save_dir = os.path.join(save_dir, env_name)
    os.makedirs(env_save_dir, exist_ok=True)
    plt.savefig(os.path.join(env_save_dir, env_name + '_' + '_'.join(exp_names) + '_by_' + by + ".png"))
    plt.close()


def main():

    parser = ArgumentParser(description='Plot')
    parser.add_argument('--env-name', type=str)
    parser.add_argument('--exp-name', type=str, nargs='+')
    parser.add_argument('--value', type=str, default='return')
    parser.add_argument('--fit', default=False, action='store_true')
    parser.add_argument('--order', type=int, default=6)
    parser.add_argument('--by', default='epoch')
    args = parser.parse_args()
    for val in [args.value, args.by]:
        assert val in LOG_MAP, '%s is invalid argument!' % val

    progress_csvs = []
    for exp_name in args.exp_name:
        file_name = os.path.join(LOG_DIR, args.env_name, args.env_name + '_' + exp_name, 'progress.csv')
        with open(file_name, 'r') as f:
            reader = csv.reader(f)
            description = next(reader)
        data = np.genfromtxt(file_name, delimiter=',')[1:]
        progress_csvs.append({'desc': description, 'data': data, 'name': exp_name})

    save_dir = os.path.join(PROJECT_DIR, 'plot')
    plot_rewards(progress_csvs, args.env_name, save_dir,
        value=args.value, by=args.by, do_fit=args.fit, fit_order=args.order)


if __name__ == '__main__':
    main()
