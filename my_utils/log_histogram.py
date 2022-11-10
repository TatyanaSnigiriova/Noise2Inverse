import time
from os.path import join
from math import log2
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


def log_full_hist(data, logs_path, bins=1000):
    h, e = np.histogram(data, bins=bins)
    log_hist_diap(h, e, logs_path, log_prefix="full_")


def log_hist_diap(h, e, logs_path, log_prefix="", ):
    # Многомерные массивы сжимаются до одной оси.
    # Интервалы ширины каждой ячейки гистограммы являются полуоткрытыми, кроме самой правой.
    print("\nh")
    print(list(enumerate(h)))

    min_intense = e[0]
    max_intense = e[-1]
    print(f"\nmin_intense = {min_intense * 1000:.2f}e-4, max_intense = {max_intense * 1000:.2f}e-4")

    diap = dict(zip(e, h))
    n_bars = len(diap)
    diap_keys = list(diap.keys())

    log_file_patch = join(
        logs_path,
        f"{log_prefix}hist_for_imgs__{n_bars}bars"
    )
    log_file = open(log_file_patch + ".csv", 'w')
    log_file.write(str(diap).replace(',', '\n'))
    log_file.close()

    diap_vals = np.array(list(diap.values()))
    yticks = diap_keys.copy()
    isx_step = int(log2(len(yticks))) - 2
    start_idx = isx_step - (len(yticks) // 2) % isx_step
    yticks.append(e[-1])
    yticks = [
        f"{key * 1000:.2f}e-4"
        if not idx % isx_step else ""
        for (idx, key) in zip(
            range(start_idx, len(yticks) + start_idx),
            yticks
        )
    ]
    fig_barh, axs = plt.subplots(1, 1, figsize=(10, 5))
    axs.barh(range(len(diap_keys)), diap_vals, align="edge")
    axs.set_yticks(range(len(yticks)), yticks)
    axs.tick_params(labelsize=8)
    plt.xticks(fontsize=8)  # rotation=90
    plt.savefig(log_file_patch + ".png")
    plt.clf()
    time.sleep(5)
    diap_vals_log = np.zeros(diap_vals.shape)
    non_zeros = diap_vals != 0
    diap_vals_log[non_zeros] = np.log(diap_vals[non_zeros])

    log_file = open(log_file_patch + "_prologarithmic.csv", 'w')
    log_file.write(str(dict(zip(diap_keys, diap_vals_log))).replace(',', '\n'))
    log_file.close()

    plt.barh(range(len(diap_keys)), diap_vals_log, align="edge")
    plt.yticks(fontsize=8, )
    plt.yticks(
        range(len(yticks)),
        yticks
    )
    plt.savefig(log_file_patch + "_prologarithmic.png")
    plt.clf()
    time.sleep(5)
