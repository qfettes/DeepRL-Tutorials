import glob
import os

# switch backend in driver file
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt


def smooth_reward_curve(x, y):
    # Halfwidth of our smoothing convolution
    halfwidth = min(31, int(np.ceil(len(x) / 30)))
    k = halfwidth
    xsmoo = x[k:-k]
    ysmoo = np.convolve(y, np.ones(2 * k + 1), mode='valid') / \
        np.convolve(np.ones_like(y), np.ones(2 * k + 1), mode='valid')
    downsample = max(int(np.floor(len(xsmoo) / 1e3)), 1)
    return xsmoo[::downsample], ysmoo[::downsample]


def fix_point(x, y, interval):
    np.insert(x, 0, 0)
    np.insert(y, 0, 0)

    fx, fy = [], []
    pointer = 0

    ninterval = int(max(x) / interval + 1)

    for i in range(ninterval):
        tmpx = interval * i

        while pointer + 1 < len(x) and tmpx > x[pointer + 1]:
            pointer += 1

        if pointer + 1 < len(x):
            alpha = (y[pointer + 1] - y[pointer]) / \
                (x[pointer + 1] - x[pointer])
            tmpy = y[pointer] + alpha * (tmpx - x[pointer])
            fx.append(tmpx)
            fy.append(tmpy)

    return fx, fy


def load_reward_data(indir, smooth, bin_size):
    datas = []
    infiles = glob.glob(os.path.join(indir, '*.monitor.csv'))

    for inf in infiles:
        with open(inf, 'r') as f:
            f.readline()
            f.readline()
            for line in f:
                tmp = line.split(',')
                t_time = float(tmp[2])
                tmp = [t_time, int(tmp[1]), float(tmp[0])]
                datas.append(tmp)

    datas = sorted(datas, key=lambda d_entry: d_entry[0])
    result = []
    timesteps = 0
    for i in range(len(datas)):
        result.append([timesteps, datas[i][-1]])
        timesteps += datas[i][1]

    if len(result) < bin_size:
        return [None, None]

    x, y = np.array(result)[:, 0], np.array(result)[:, 1]

    if smooth == 1:
        x, y = smooth_reward_curve(x, y)

    if smooth == 2:
        y = medfilt(y, kernel_size=9)

    x, y = fix_point(x, y, bin_size)
    return [x, y]

# TODO: only works for Experience Replay style training for now


def load_custom_data(indir, stat_file, smooth, bin_size):
    datas = []
    infiles = glob.glob(os.path.join(indir, stat_file))

    for inf in infiles:  # should be 1
        with open(inf, 'r') as f:
            for line in f:
                tmp = line.split(',')
                tmp = [int(tmp[0]), float(tmp[1])]
                datas.append(tmp)

    datas = sorted(datas, key=lambda d_entry: d_entry[0])
    result = []
    for i in range(len(datas)):
        result.append([datas[i][0], datas[i][1]])

    if len(result) < bin_size:
        return [None, None]

    x, y = np.array(result)[:, 0], np.array(result)[:, 1]

    if smooth == 1:
        x, y = smooth_reward_curve(x, y)

    if smooth == 2:
        y = medfilt(y, kernel_size=9)

    x, y = fix_point(x, y, bin_size)
    return [x, y]

# TODO: only works for Experience Replay style training for now


def load_action_data(indir, smooth, bin_size):
    datas = []
    infiles = glob.glob(os.path.join(indir, 'action_log.csv'))

    for inf in infiles:  # should be 1
        with open(inf, 'r') as f:
            for line in f:
                tmp = line.split(',')
                tmp = [int(tmp[0])] + [float(tmp[i])
                                       for i in range(1, len(tmp))]
                datas.append(tmp)

    datas = sorted(datas, key=lambda d_entry: d_entry[0])
    result = datas
    # for i in range(len(datas)):
    #    result.append([datas[i][0], datas[i][1]])

    if len(result) < bin_size:
        return [None, None]

    x, y = np.array(result)[:, 0], np.array(result)[:, 1:]

    '''if smooth == 1:
        x, y = smooth_reward_curve(x, y)

    if smooth == 2:
        y = medfilt(y, kernel_size=9)

    x, y = fix_point(x, y, bin_size)'''
    return [x, np.transpose(y)]


def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


def plot_all_data(folder, game, name, num_steps, bin_size=(10, 1), smooth=1, time=None, save_filename='results.png', ipynb=False):
    matplotlib.rcParams.update({'font.size': 20})
    params = {
        'xtick.labelsize': 20,
        'ytick.labelsize': 15,
        'legend.fontsize': 15
    }
    plt.rcParams.update(params)

    tx, ty = load_reward_data(folder, smooth, bin_size[0])

    if tx is None or ty is None:
        return

    if time is not None:
        title = 'Avg. Last 10 Rewards: ' + \
            str(np.round(np.mean(ty[-10]))) + ' || ' + game + \
            ' || Elapsed Time: ' + str(time).split('.')[0]
    else:
        title = 'Avg. Last 10 Rewards: ' + \
            str(np.round(np.mean(ty[-10]))) + ' || ' + game

    tick_fractions = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
    ticks = tick_fractions * num_steps
    tick_names = ["{:.0e}".format(tick) for tick in ticks]

    fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10) = plt.subplots(10, 1, figsize=(
        20, 40), subplot_kw=dict(xticks=ticks, xlim=(0, num_steps*1.15), xlabel='Timestep', title=title))
    ax1.set_xticklabels(tick_names)
    ax2.set_xticklabels(tick_names)
    ax3.set_xticklabels(tick_names)
    ax4.set_xticklabels(tick_names)
    ax5.set_xticklabels(tick_names)
    ax6.set_xticklabels(tick_names)
    ax7.set_xticklabels(tick_names)
    ax8.set_xticklabels(tick_names)
    ax9.set_xticklabels(tick_names)
    ax10.set_xticklabels(tick_names)

    ax1.set_ylabel('Reward')

    p1, = ax1.plot(tx, ty, label="Reward")
    #lines = [p1]

    ax1.yaxis.label.set_color(p1.get_color())
    ax1.tick_params(axis='y', colors=p1.get_color())

    g1_lines = [p1]

    '''tx, ty = load_custom_data(folder, 'max_dist.csv', smooth, bin_size[1])

    if tx is not None or ty is not None:
        #need to update g2 title if sig data will be included
        ax1.set_title('Reward/Maximum x distance traveled vs Timestep')

        ax2 = ax1.twinx()

        ax2.set_ylabel('Max x distance')
        p2, = ax2.plot(tx, ty, 'g-', label='Max x dist.')
        g1_lines += [p2]

        ax2.yaxis.label.set_color(p2.get_color())
        ax2.tick_params(axis='y', colors=p2.get_color())'''

    # remake g2 legend because we have a new line
    ax1.legend(g1_lines, [l.get_label() for l in g1_lines], loc=4)

    tx, ty = load_custom_data(folder, 'max_dist.csv', smooth, bin_size[1])
    subplot_generic(ax2, 'X Distance Traveled vs. Time', 'X Distance', tx, ty)

    tx, ty = load_custom_data(folder, 'total_loss.csv', smooth, bin_size[1])
    subplot_generic(ax3, 'Total Loss vs. Time', 'Total Loss', tx, ty)

    tx, ty = load_custom_data(folder, 'policy_loss.csv', smooth, bin_size[1])
    subplot_generic(ax4, 'Policy Loss vs. Time', 'Policy Loss', tx, ty)

    tx, ty = load_custom_data(folder, 'value_loss.csv', smooth, bin_size[1])
    subplot_generic(ax5, 'Value Loss vs. Time', 'Value Loss', tx, ty)

    tx, ty = load_custom_data(folder, 'dynamics_loss.csv', smooth, bin_size[1])
    subplot_generic(ax6, 'Dynamics Loss vs. Time', 'Dynamics Loss', tx, ty)

    tx, ty = load_custom_data(
        folder, 'policy_entropy.csv', smooth, bin_size[1])
    subplot_generic(ax7, 'Policy Entropy vs. Time', 'Entropy', tx, ty)

    tx, ty = load_custom_data(folder, 'grad_norms.csv', smooth, bin_size[1])
    subplot_generic(ax8, 'Grad Norm vs. Time', 'Grad Norm', tx, ty)

    tx, ty = load_custom_data(
        folder, 'value_estimate.csv', smooth, bin_size[1])
    subplot_generic(ax9, 'Avg Value Estimate vs. Time',
                    'Value Estimate', tx, ty)

    tx, ty = load_custom_data(folder, 'learning_rate.csv', smooth, bin_size[1])
    subplot_generic(ax10, 'Learning Rate vs. Time', 'Learning Rate', tx, ty)

    '''#Load td data if it exists
    tx, ty = load_custom_data(folder, 'max_dist.csv', smooth, bin_size[1])

    ax2.set_title('Maximum x distance traveled vs Timestep')

    if tx is not None or ty is not None:
        ax2.set_ylabel('X Distance')
        p2, = ax2.plot(tx, ty, 'r-', label='X distance')
        g2_lines = [p2]

        ax2.yaxis.label.set_color(p2.get_color())
        ax2.tick_params(axis='y', colors=p2.get_color())

        ax2.legend(g2_lines, [l.get_label() for l in g2_lines], loc=4)'''

    # Load action selection data if it exists
    '''tx, ty = load_action_data(folder, smooth, bin_size[3])

    ax3.set_title('Action Selection Frequency(%) vs Timestep')

    if tx is not None or ty is not None:
        ax3.set_ylabel('Action Selection Frequency(%)')
        labels = ['Action {}'.format(i) for i in range(ty.shape[0])]
        p3 = ax3.stackplot(tx, ty, labels=labels)

        base = 0.0
        for percent, index in zip(ty, range(ty.shape[0])):
            offset = base + percent[-1]/3.0
            ax3.annotate(str('{:.2f}'.format(ty[index][-1])), xy=(tx[-1], offset), color=p3[index].get_facecolor().ravel())
            base += percent[-1]

        #ax3.yaxis.label.set_color(p3.get_color())
        #ax3.tick_params(axis='y', colors=p3.get_color())

        ax3.legend(loc=4) #remake g2 legend because we have a new line'''

    plt.tight_layout()  # prevent label cutoff

    if ipynb:
        plt.show()
    else:
        plt.savefig(os.path.join(folder, save_filename))
    plt.clf()
    plt.close()

    # return np.round(np.mean(ty[-10:]))


def subplot_generic(ax, title, ylabel, tx, ty):
    if tx is not None or ty is not None:
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        p, = ax.plot(tx, ty, 'r-', label=ylabel)
        g_lines = [p]

        ax.yaxis.label.set_color(p.get_color())
        ax.tick_params(axis='y', colors=p.get_color())

        ax.legend(g_lines, [l.get_label() for l in g_lines], loc=4)


def plot_reward(folder, game, num_steps, bin_size=10, smooth=1, time=None, save_filename='results.png', ipynb=False):
    matplotlib.rcParams.update({'font.size': 20})
    params = {
        'xtick.labelsize': 20,
        'ytick.labelsize': 15,
        'legend.fontsize': 15
    }
    plt.rcParams.update(params)

    tx, ty = load_reward_data(folder, smooth, bin_size)

    if tx is None or ty is None:
        return

    tx = np.array(tx, dtype=int)
    tx = tx.tolist()

    if time is not None:
        title = 'Avg. Last 10 Rewards: ' + \
            str(np.round(np.mean(ty[-10]))) + ' || ' + game + \
            ' || Elapsed Time: ' + str(time).split('.')[0]
    else:
        title = 'Avg. Last 10 Rewards: ' + \
            str(np.round(np.mean(ty[-10]))) + ' || ' + game

    tick_fractions = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
    ticks = tick_fractions * num_steps
    tick_names = ["{:.0e}".format(tick) for tick in ticks]

    fig, ax1 = plt.subplots(1, 1, figsize=(20, 5), subplot_kw=dict(
        xticks=ticks, xlim=(0, num_steps*1.15), xlabel='Timestep', title=title))
    ax1.set_xticklabels(tick_names)

    ax1.set_ylabel('Reward')

    p1, = ax1.plot(tx, ty, label="Reward")
    #lines = [p1]

    ax1.yaxis.label.set_color(p1.get_color())
    ax1.tick_params(axis='y', colors=p1.get_color())

    g1_lines = [p1]

    # remake g2 legend because we have a new line
    ax1.legend(g1_lines, [l.get_label() for l in g1_lines], loc=4)

    plt.tight_layout()  # prevent label cutoff

    if ipynb:
        plt.show()
    else:
        plt.savefig(os.path.join(folder, save_filename))
    plt.clf()
    plt.close()
