import numpy as np

#switch backend in driver file
import matplotlib
import matplotlib.pyplot as plt

import os
import glob
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

#TODO: only works for Experience Replay style training for now
def load_custom_data(indir, stat_file, smooth, bin_size):
    datas = []
    infiles = glob.glob(os.path.join(indir, stat_file))

    for inf in infiles: #should be 1
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

#TODO: only works for Experience Replay style training for now
def load_action_data(indir, smooth, bin_size):
    datas = []
    infiles = glob.glob(os.path.join(indir, 'action_log.csv'))

    for inf in infiles: #should be 1
        with open(inf, 'r') as f:
            for line in f:
                tmp = line.split(',')
                tmp = [int(tmp[0])] + [float(tmp[i]) for i in range(1, len(tmp))]
                datas.append(tmp)

    datas = sorted(datas, key=lambda d_entry: d_entry[0])
    result = datas
    #for i in range(len(datas)):
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

def visdom_plot(viz, win, folder, game, name, num_steps, bin_size=100, smooth=1):
    tx, ty = load_reward_data(folder, smooth, bin_size)
    if tx is None or ty is None:
        return win

    fig = plt.figure()
    plt.plot(tx, ty, label="{}".format(name))

    tick_fractions = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
    ticks = tick_fractions * num_steps
    tick_names = ["{:.0e}".format(tick) for tick in ticks]
    plt.xticks(ticks, tick_names)
    plt.xlim(0, num_steps * 1.01)

    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')

    plt.title(game)
    plt.legend(loc=4)
    plt.show()
    plt.draw()

    
    image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3, ))
    plt.close(fig)

    #Show it in visdom
    image = np.transpose(image, (2, 0, 1))

    return viz.image(image, win=win)

def plot(folder, game, name, num_steps, bin_size=100, smooth=1):
    matplotlib.rcParams.update({'font.size': 20})
    tx, ty = load_reward_data(folder, smooth, bin_size)

    if tx is None or ty is None:
        return

    fig = plt.figure(figsize=(20,5))
    plt.plot(tx, ty, label="{}".format(name))

    tick_fractions = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
    ticks = tick_fractions * num_steps
    tick_names = ["{:.0e}".format(tick) for tick in ticks]
    plt.xticks(ticks, tick_names)
    plt.xlim(0, num_steps * 1.01)

    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')

    plt.title(game)
    plt.legend(loc=4)
    plt.show()

def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)

def plot_all_data(folder, game, name, num_steps, bin_size=(10, 100, 100, 1), smooth=1, time=None, save_filename='results.png', ipynb=False):
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
        title = 'Avg. Last 10 Rewards: ' +  str(np.round(np.mean(ty[-10]))) + ' || ' +  game + ' || Elapsed Time: ' + str(time)
    else:
        title = 'Avg. Last 10 Rewards: ' +  str(np.round(np.mean(ty[-10]))) + ' || ' +  game

    tick_fractions = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
    ticks = tick_fractions * num_steps
    tick_names = ["{:.0e}".format(tick) for tick in ticks]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 15), subplot_kw = dict(xticks=ticks, xlim=(0, num_steps*1.15), xlabel='Timestep', title=title))
    ax1.set_xticklabels(tick_names)
    ax2.set_xticklabels(tick_names)
    ax3.set_xticklabels(tick_names)

    ax1.set_ylabel('Reward')

    p1, = ax1.plot(tx, ty, label="Reward")
    #lines = [p1]

    ax1.yaxis.label.set_color(p1.get_color())
    ax1.tick_params(axis='y', colors=p1.get_color())

    ax1.legend([p1], [p1.get_label()], loc=4)

    
    #Load td data if it exists
    tx, ty = load_custom_data(folder, 'td.csv', smooth, bin_size[1])

    ax2.set_title('Loss vs Timestep')

    if tx is not None or ty is not None:
        ax2.set_ylabel('Avg .Temporal Difference')
        p2, = ax2.plot(tx, ty, 'r-', label='Avg. TD')
        g2_lines = [p2]

        ax2.yaxis.label.set_color(p2.get_color())
        ax2.tick_params(axis='y', colors=p2.get_color())

        ax2.legend(g2_lines, [l.get_label() for l in g2_lines], loc=4)
    
    #Load Sigma Parameter Data if it exists
    tx, ty = load_custom_data(folder, 'sig_param_mag.csv', smooth, bin_size[2])

    if tx is not None or ty is not None:
        #need to update g2 title if sig data will be included
        ax2.set_title('Loss/Avg. Sigma Parameter Magnitude vs Timestep')

        ax4 = ax2.twinx()

        ax4.set_ylabel('Avg. Sigma Parameter Mag.')
        p4, = ax4.plot(tx, ty, 'g-', label='Avg. Sigma Mag.')
        g2_lines += [p4]

        ax4.yaxis.label.set_color(p4.get_color())
        ax4.tick_params(axis='y', colors=p4.get_color())

        #ax4.spines["right"].set_position(("axes", 1.05))
        #make_patch_spines_invisible(ax4)
        #ax4.spines["right"].set_visible(True)

        ax2.legend(g2_lines, [l.get_label() for l in g2_lines], loc=4) #remake g2 legend because we have a new line

    #Load action selection data if it exists
    tx, ty = load_action_data(folder, smooth, bin_size[3])

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

        ax3.legend(loc=4) #remake g2 legend because we have a new line

    plt.tight_layout() # prevent label cutoff

    if ipynb:
        plt.show()
    else:
        plt.savefig(save_filename)
    plt.clf()
    plt.close()
    
    #return np.round(np.mean(ty[-10:]))

def plot_reward(folder, game, name, num_steps, bin_size=10, smooth=1, time=None, save_filename='results.png', ipynb=False):
    matplotlib.rcParams.update({'font.size': 20})
    tx, ty = load_reward_data(folder, smooth, bin_size)

    if tx is None or ty is None:
        return

    fig = plt.figure(figsize=(20,5))
    plt.plot(tx, ty, label="{}".format(name))

    tick_fractions = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
    ticks = tick_fractions * num_steps
    tick_names = ["{:.0e}".format(tick) for tick in ticks]
    plt.xticks(ticks, tick_names)
    plt.xlim(0, num_steps * 1.01)

    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')

    if time is not None:
        plt.title(game + ' || Last 10: ' + str(np.round(np.mean(ty[-10]))) + ' || Elapsed Time: ' + str(time))
    else:
        plt.title(game + ' || Last 10: ' + str(np.round(np.mean(ty[-10]))))
    plt.legend(loc=4)
    if ipynb:
        plt.show()
    else:
        plt.savefig(save_filename)
    plt.clf()
    plt.close()
    
    return np.round(np.mean(ty[-10]))

'''def plot_td(folder, game, name, num_steps, bin_size=10, smooth=1, time=None, save_filename='td.png', ipynb=False):
    matplotlib.rcParams.update({'font.size': 20})
    tx, ty = load_custom_data(folder, 'td.csv', smooth, bin_size)

    if tx is None or ty is None:
        return

    fig = plt.figure(figsize=(20,5))
    plt.plot(tx, ty, label="{}".format(name))

    tick_fractions = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
    ticks = tick_fractions * num_steps
    tick_names = ["{:.0e}".format(tick) for tick in ticks]
    plt.xticks(ticks, tick_names)
    plt.xlim(0, num_steps * 1.01)

    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')

    if time is not None:
        plt.title(game + ' || Last 10: ' + str(np.round(np.mean(ty[-1]))) + ' || Elapsed Time: ' + str(time))
    else:
        plt.title(game + ' || Last 10: ' + str(np.round(np.mean(ty[-1]))))
    plt.legend(loc=4)
    if ipynb:
        plt.show()
    else:
        plt.savefig(save_filename)
    plt.clf()
    plt.close()
    
    return np.round(np.mean(ty[-1]))

def plot_sig(folder, game, name, num_steps, bin_size=10, smooth=1, time=None, save_filename='sig.png', ipynb=False):
    matplotlib.rcParams.update({'font.size': 20})
    tx, ty = load_custom_data(folder, 'sig_param_mag.csv', smooth, bin_size)

    if tx is None or ty is None:
        return

    fig = plt.figure(figsize=(20,5))
    plt.plot(tx, ty, label="{}".format(name))

    tick_fractions = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
    ticks = tick_fractions * num_steps
    tick_names = ["{:.0e}".format(tick) for tick in ticks]
    plt.xticks(ticks, tick_names)
    plt.xlim(0, num_steps * 1.01)

    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')

    if time is not None:
        plt.title(game + ' || Last 10: ' + str(np.round(np.mean(ty[-1]))) + ' || Elapsed Time: ' + str(time))
    else:
        plt.title(game + ' || Last 10: ' + str(np.round(np.mean(ty[-1]))))
    plt.legend(loc=4)
    if ipynb:
        plt.show()
    else:
        plt.savefig(save_filename)
    plt.clf()
    plt.close()
    
    return np.round(np.mean(ty[-1]))'''