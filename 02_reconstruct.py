import os
from os import listdir
from os.path import split, realpath
from os.path import join, exists
import argparse
import tifffile
import tomosipo as ts
import tomopy
import matplotlib.pyplot as plt
import numpy as np
import random
from noise2inverse import tiffs, tomo, fig
import time
from math import log2
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
from tomopy import ufo_fbp
from noise2inverse import tiffs, tomo, fig
from noise2inverse.tomo import filter_proj_data
from copy import deepcopy
import tomosipo.torch_support
import torch
# Load data
import dxchange
from tqdm import tqdm
from IPython.display import Audio, display, HTML

display(HTML("<style>.container { width:97.5% !important; }</style>"))
from ipywidgets import interact
from pathlib import Path
import re
from noise2inverse import fig, tiffs, noise
# Load data
import h5py
import cupy as cp
from os.path import join
import ts_algorithms
import tomopy.util.dtype as dtype
from noise2inverse import tiffs, tomo, fig
# Load data
from noise2inverse import fig, tiffs, noise
from my_utils.log_histogram import *
import localtomo
from localtomo.tomography import AstraToolbox

class FBPSinosPreprocessor:
    '''
    Класс для центрирования данных для метода noise2inverse.tomo.fbp
    '''

    def __init__(self):
        self.trained = False

    def fit(self, height, num_angles, width, center):
        # Determine geometry
        self.height, self.num_angles, self.width = height, num_angles, width
        if center == -1:
            self.center = self.width // 2
        else:
            self.center = center

        if self.center > self.width / 2:
            self.new_width = 2 * self.center
            self.left_pad = 0
            # self.right_pad = 2 * self.center - self.width
        elif self.center < self.width / 2:
            self.new_width = 2 * (self.width - self.center)
            self.left_pad = self.width - 2 * self.center
        else:
            self.new_width = self.width
            self.left_pad = 0
        self.trained = True
        return self

    def transform(self, sinos):
        assert self.trained, "FBPSinosPreprocessor NotTrainedError: Трансформмер еще не был обучен, вызовите fit-метод"
        new_sinos = np.zeros((self.height, self.num_angles, self.new_width))
        new_sinos[:, :, self.left_pad:self.left_pad + self.width] = sinos
        return new_sinos

    def fit_transform(self, sinos, center, height_dim=0, angles_dim=1, width_dim=2):
        self.fit(
            height=sinos.shape[height_dim],
            num_angles=sinos.shape[angles_dim],
            width=sinos.shape[width_dim],
            center=center,
        )
        return self.transform(sinos)


class ReconstructByAnglesSlice:
    supported_postprocessing = ["no", "circ_mask", "median_filter"]
    supported_libraries_and_methods = {
        "localtomo": ["fbp", ],
        "noise2inverse": ["fbp", ]
    }

    def __init__(
            self, library, method,
            height, num_angles, width,
            center, rec_width,
            postprocessing_args={"circ_mask": [('ratio', 1), ("axis", 0), ]}
            # ToDo: postprocessing_args - уникальность по ключу
            #  Если захочу применить какую-то трансфрмацию еще раз позже,
            #  то первое значение по этому ключу будет заменено
    ):
        print("postprocessing_args:", postprocessing_args)

        assert library in ReconstructByAnglesSlice.supported_libraries_and_methods.keys(), \
            f"ReconstructByAnglesSlice Error: Not supported library {library} for ReconstructByAnglesSlice class."
        assert method in ReconstructByAnglesSlice.supported_libraries_and_methods[library], \
            f"ReconstructByAnglesSlice Error: Not supported reconstruction method {method} in library {library}" + \
            "for ReconstructByAnglesSlice class."
        self.library = library
        self.method = method

        postprocessing_list = np.array([*postprocessing_args.keys()])
        print("postprocessing_list", postprocessing_list)
        not_supported_postprocessing = set(postprocessing_list).difference(
            set(ReconstructByAnglesSlice.supported_postprocessing)
        )
        if len(not_supported_postprocessing) != 0:
            print(
                "ReconstructByAnglesSlice: Следующие методы постобработки реконструкций не поддерживаются:\t",
                not_supported_postprocessing
            )
            self.postprocessing_list = postprocessing_list[
                list(map(
                    lambda name: name not in not_supported_postprocessing,
                    postprocessing_list
                ))
            ]
        else:
            self.postprocessing_list = postprocessing_list
        self.postprocessing_args = deepcopy(postprocessing_args)

        self.height, self.num_angles = height, num_angles
        # ToDo - Сейчас масимальное сжатие реконструкции - в 2 раза
        if self.library == "noise2inverse" and rec_width >= width // 2:
            self.width = rec_width
        else:
            self.width = width
        self.angles = np.linspace(0, np.pi, num=self.num_angles, endpoint=False)
        print("\nangles:", len(self.angles), self.angles)

        if self.library == "noise2inverse":
            self.pr_obj = FBPSinosPreprocessor()
            self.pr_obj.fit(height, num_angles, width, center)
            self.new_width = self.pr_obj.new_width
            self.center = self.pr_obj.center

            # Determine geometry
            # ToDo - может вместо vol_width тоже новый размер self.width = self.pr_obj.new_width?
            self.vol_shape = (self.height, self.width, self.width)
            # т.е. здесь можно задавать меньший размер
            self.det_shape = (self.height, self.new_width)
            print("\tvol_shape:", self.vol_shape, "\tdet_shape", self.det_shape)

            self.vg = ts.volume(size=self.vol_shape, pos=0, shape=self.vol_shape)  # pos=(0, 0, 0)
            self.pg = ts.parallel(angles=self.angles, size=self.det_shape, shape=self.det_shape)

            self.A = ts.operator(self.vg, self.pg)
        else:
            self.center = self.width / 2  # ToDo Или найти через tomopy

            if self.library == "localtomo":
                # ToDo - не работает, возможно, здесь требуется обрабатывать каждую отдельную синограму
                #  по измерению self.height
                self.astratb_obj = AstraToolbox(
                    slice_shape=(self.width, self.width),
                    dwidth=self.width,
                    angles=self.angles, rot_center=self.center
                )
            else:
                assert False, "#ToDo"


    def postprocess(self, recs):
        for name_transform in self.postprocessing_list:
            if name_transform == "circ_mask":
                recs = tomopy.circ_mask(recs, **dict(self.postprocessing_args[name_transform]))

            if name_transform == "median_filter":
                recs = tomopy.misc.corr.median_filter(recs, **dict(self.postprocessing_args[name_transform]))

        return recs

    def gen_recs(self, sinos):
        if self.library == "localtomo":
            recs = self.astratb_obj.fbp(sinos)
        elif self.library == "noise2inverse":
            new_sinos = self.pr_obj.transform(sinos)
            print("\nnew_sinos:", new_sinos.shape)

            recs = tomo.fbp(self.A, new_sinos)

        else:
            assert False, "#ToDo"

        return self.postprocess(recs)

    def gen_recs_by_angles_splits(self, sinos, num_splits):
        if self.library == "noise2inverse":
            new_sinos = self.pr_obj.transform(sinos)
        else:
            new_sinos = deepcopy(sinos)

        for j in range(num_splits):
            sinos_split = new_sinos[:, j::num_splits, :]
            print("\nsinos_split:", sinos_split.shape, )

            if self.library == "localtomo":
                recs_split = self.astratb_obj.fbp(sinos_split)
            elif self.library == "noise2inverse":
                pg_split = self.pg[j::num_splits]
                A_split = ts.operator(self.vg, pg_split)
                recs_split = tomo.fbp(A_split, sinos_split)
                '''
                    filtered_sino_split = filter_proj_data(
                        torch.from_numpy(sinos_split)
                    ).detach().numpy()
                    recs_noisy_split = my_tomopy_recon(
                        filtered_sino_split, angles,
                        center=center,
                        sinogram_order=True
                    )
                '''

            else:
                assert False, "#ToDo"

            yield self.postprocess(recs_split)


# Инициализация проекта и библиотеки tf должны происходить после настройки ус-ва
def setup_seed(seed):
    random.seed(seed)  # Set random seed for Python
    np.random.seed(seed)  # Set random seed for numpy
    os.environ['TF_DETERMINISTIC_OPS'] = '1'  # tf gpu fix seed, please `pip install tensorflow-determinism` first


def get_sample_name(projs_path):
    sample_name = split(projs_path)[-1]
    if sample_name.find('.') != -1:
        sample_name = sample_name[:sample_name.find('.')]
    return sample_name


def main(
        device, global_rs,  # ToDo setRS
        rec_library, rec_method,
        main_data_path, clean_projections_dir, noisy_projections_dir,
        main_logs_path,
        center, rec_width,
        modes_correction_list, num_splits,
        normalize_rec, hist_percent,
        circ_mask_ratio,
        output_recs_prefix, log_rec_idx,
):
    print(
        f'\n\t\tdevice = {device}',
        f'global_rs={global_rs}',
        f'rec_library={rec_library}',
        f'rec_method={rec_method}',
        f'main_data_path={main_data_path}',
        f'clean_projections_dir={clean_projections_dir}',
        f'noisy_projections_dir={noisy_projections_dir}',
        f'main_logs_path={main_logs_path}',
        f'center={center}',
        f'rec_width={rec_width}',
        f'modes_correction_list={modes_correction_list}',
        f'splits={num_splits}',
        f'normalize_rec={normalize_rec}',
        f'hist_percent={hist_percent}',
        f'circ_mask_ratio={circ_mask_ratio}',
        f'output_recs_prefix={output_recs_prefix}',
        f'log_rec_idx={log_rec_idx}\n', sep='\n\t\t'
    )
    dir_pattern_name = f"lib={rec_library}_method={rec_method}" + \
         f"_norm={normalize_rec}_modes={'&'.join(modes_correction_list)}_hist={hist_percent}%"
    output_logs_dir = join(main_logs_path, dir_pattern_name, "reconstructions")
    os.makedirs(output_logs_dir, exist_ok=True)
    output_data_dir = join(main_data_path, dir_pattern_name, "reconstructions")
    os.makedirs(output_data_dir, exist_ok=True)

    if clean_projections_dir:
        output_clean_dir = join(output_data_dir, "clean")
        os.makedirs(output_clean_dir, exist_ok=True)
        sinos_clean = tiffs.load_sino(tiffs.glob(join(main_data_path, clean_projections_dir)))
        # sinos_clean = torch.from_numpy(sinos_clean)
        # sinos_clean = sinos_clean.astype(np.float32)
        # sinos_clean = np.moveaxis(sinos_clean, 1, 0)
        print("sino_clean:", sinos_clean.shape, type(sinos_clean), sinos_clean.dtype)

    output_noisy_dir = join(output_data_dir, "noisy")
    os.makedirs(output_noisy_dir, exist_ok=True)
    # Load data
    sinos_noisy = tiffs.load_sino(tiffs.glob(join(main_data_path, noisy_projections_dir)))
    # sinos_noisy = torch.from_numpy(sinos_noisy)
    # sinos_noisy = sinos_noisy.astype(np.float32)
    # sinos_noisy = np.moveaxis(sinos_noisy, 1, 0)
    print("sino_noisy:", sinos_noisy.shape, type(sinos_noisy), sinos_noisy.dtype)

    # ToDo А у кого торчат усы как у лисы?
    postprocessing_args = dict()
    if "circ_mask" in modes_correction_list:
        postprocessing_args["circ_mask"] = [('ratio', circ_mask_ratio), ('axis', 0), ]
    if "median_filter" in modes_correction_list:
        postprocessing_args["median"] = [('size', 3), ('axis', 0), ]
    print("postprocessing_args:", postprocessing_args)
    log_images = dict()

    height, num_angles, width = sinos_noisy.shape
    rec_model = ReconstructByAnglesSlice(
        rec_library, rec_method,
        height, num_angles, width,
        center, rec_width,
        postprocessing_args
    )
    if clean_projections_dir:
        # Reconstruct clean
        recs_clean = rec_model.gen_recs(sinos_clean)
        print("\nrecs_clean:", recs_clean.shape)
        tiffs.save_stack(output_clean_dir, recs_clean)
        log_images['slice_clean'] = tifffile.imread(
            join(output_clean_dir, f"{output_recs_prefix}_{log_rec_idx:05d}.tif")
        )

    # Reconstruct noisy
    recs_noisy = rec_model.gen_recs(sinos_noisy)
    print("\nrecs_noisy:", recs_noisy.shape)
    tiffs.save_stack(output_noisy_dir, recs_noisy)
    log_images['slice_noisy'] = tifffile.imread(
        join(output_noisy_dir, f"{output_recs_prefix}_{log_rec_idx:05d}.tif")
    )

    if normalize_rec:
        log_full_hist(recs_noisy, output_logs_dir, bins=100)

        h, e = np.histogram(recs_noisy, bins=1000)  # Многомерные массивы сжимаются до одной оси.
        # Интервалы ширины каждой ячейки гистограммы являются полуоткрытыми, кроме самой правой.
        print("\nh")
        print(list(enumerate(h)))

        k = (100 - hist_percent) / 100
        # for k in np.arange(0.013, 0.02 + 0.0001, 0.001):
        # for k in np.arange(0.015 - 0.0005, 0.015 + 0.0005, 0.0005):
        # for k in np.arange(0.001, 0.01 + 0.0001, 0.0005):
        print("\nk =", k)
        print("\nnp.max(h) =", np.max(h), "np.max(h) * k =", np.max(h) * k)
        stend = np.where(h > np.max(h) * k)  # Индексы
        # stend = np.where(h > 100) # Индексы
        st = stend[0][0]
        end = stend[0][-1]
        print(st, end)
        min_intense = e[st]
        max_intense = e[end + 1]
        print(f"\nmin_intense = {min_intense * 1000:.2f}e-4, max_intense = {max_intense * 1000:.2f}e-4")
        diap_name = f"({min_intense * 1000:.2f}e-4, {max_intense * 1000:.2f}e-4)"

        log_hist_diap(h[st:end + 1], e[st:end + 1], logs_path=output_logs_dir)

        print("diap_name:", diap_name, type(diap_name))
        tiffs.norm_and_save_stack(output_noisy_dir, recs_noisy, min_intense=min_intense, max_intense=max_intense)
    else:
        tiffs.save_stack(output_noisy_dir, recs_noisy)

    log_images['slice_noisy'] = tifffile.imread(
        join(output_noisy_dir, f"{output_recs_prefix}_{log_rec_idx:05d}.tif")
    )

    # Reconstruct noisy splits
    for j, recs_noisy_split in enumerate(rec_model.gen_recs_by_angles_splits(sinos_noisy, num_splits)):
        output_split_dir = join(output_data_dir, str(j))
        os.makedirs(output_split_dir, exist_ok=True)
        print("\nrecs_noisy_split:", recs_noisy_split.shape)
        if normalize_rec:
            tiffs.norm_and_save_stack(
                output_split_dir, recs_noisy_split,
                min_intense=min_intense, max_intense=max_intense
            )
        else:
            tiffs.save_stack(output_split_dir, recs_noisy_split)

        log_images[f'slice_noisy_{j}'] = tifffile.imread(
            join(output_split_dir, f"{output_recs_prefix}_{log_rec_idx:05d}.tif")
        )

    print(f"{'clean_' if clean_projections_dir else ''}" +
          f"noisy_sub{'_'.join(list(map(str, range(num_splits))))}" +
          f"_norm={str(normalize_rec)}" +
          f"{'_diap=' + str(diap_name) if normalize_rec else ''}" +
          ".png")
    fig.plot_imgs(
        **log_images,
        width=20.0,
        name=join(
            output_logs_dir,
            f"{'clean_' if clean_projections_dir else ''}" +
            f"noisy_sub{'_'.join(map(str, list(range(num_splits))))}" +
            f"_norm={str(normalize_rec)}" +
            f"{'_diap=' + str(diap_name) if normalize_rec else ''}" +
            ".png"
        )
    )


if __name__ == '__main__':
    print("------------------------------------------------------------------------------------------------")
    parser = argparse.ArgumentParser()
    # -----------------------------------------------------------------------------------------
    # Настройка окружения
    parser.add_argument('-cpu', '--cpu', default=1, type=int, choices=[0, 1], required=False)
    parser.add_argument('-global_rs', '--global_random_seed', default=42, type=int, required=False)
    parser.add_argument('-rec_lib', '--reconstruction_library', default="noise2inverse", type=str, required=False)
    parser.add_argument('-rec_method', '--reconstruction_method', default="fbp", type=str, required=False)

    # -----------------------------------------------------------------------------------------
    # Наборы данных и сохранение результата
    parser.add_argument('-sample_dir', '--sample_dir_path', default=join(".", "phantom"), type=str, required=False)
    parser.add_argument('-data_dir', '--data_dir_name', default="data", type=str, required=False)
    parser.add_argument('-logs_dir', '--logs_dir_name', default="logs", type=str, required=False)
    parser.add_argument('-proj_dir', '--proj_dir_prefix', default="projections", type=str, required=False)
    parser.add_argument('-prefix', '--output_proj_prefix', default="output", type=str, required=False)
    parser.add_argument('-log_idx', '--log_rec_idx', default=50, type=int, required=False)
    # -----------------------------------------------------------------------------------------
    # Инициализация параметров реконструкции
    parser.add_argument('-center', '--center', default=-1, type=int, required=False)
    # Если отрицательный - середина
    parser.add_argument('-width', '--rec_width', default=-1, type=int, required=False)
    # Если отрицательный - изначальный
    parser.add_argument('-modes', '--modes_correction', default="circ_mask", type=str, required=False)
    parser.add_argument('-splits', '--num_splits', default=4, type=int, required=False)
    parser.add_argument('-norm', '--normalize_rec', default=0, type=int, choices=[0, 1], required=False)
    parser.add_argument('-hist', '--hist_percent', default=100, type=int, required=False)
    parser.add_argument('-circ_mask', '--circ_mask_ratio', default=1.0, type=float, required=False)

    # Для минимакс-нормализации будет использован весь диапазон интенсивностей

    args = parser.parse_args()

    print("sample_dir_path:\t", args.sample_dir_path)
    sample_dir_path = realpath(args.sample_dir_path)
    print("sample_dir_path:\t", sample_dir_path)
    assert exists(sample_dir_path), f"Указанный путь к директории образца {sample_dir_path} не существует"

    main_data_path = join(sample_dir_path, args.data_dir_name)
    assert exists(main_data_path), f"По указанному пути {main_data_path} не найдена директория с данными образца"
    projections_dirs = list(filter(
        lambda dir_name: dir_name.find(args.proj_dir_prefix) != -1,
        listdir(main_data_path)
    ))
    print("projections_dirs:", projections_dirs)
    assert len(projections_dirs) > 0, \
        f"Директория с данными образца {main_data_path} не содержит директории с .tif прокциями"

    # Ожидаю найти максимум одну чистую и одну зашумленную директорию проекций
    clean_projections_dir = list(filter(lambda projections_dir: projections_dir.rfind('clean') != -1, projections_dirs))
    print("clean_projections_dir:", clean_projections_dir)
    assert len(clean_projections_dir) < 2, \
        f"Директория с данными образца {main_data_path} должна содержать только" \
        f" директории с шумными и чистыми проекциями," \
        f" найдено несколько директорий для чистых проекций {clean_projections_dir}"

    if len(clean_projections_dir) == 1:
        clean_projections_dir = clean_projections_dir[0]
        assert len(listdir(join(main_data_path, clean_projections_dir))) > 0, \
            f"Директория чистых проекций {clean_projections_dir} не содержит проекций"
    else:
        print("Не найдено директории для чистых от шума проекций. Обработка только шумных проекций.")
        clean_projections_dir = None

    noisy_projections_dir = list(filter(lambda projections_dir: projections_dir.rfind('noisy') != -1, projections_dirs))
    print("noisy_projections_dir:", noisy_projections_dir)
    assert len(noisy_projections_dir) < 2, \
        f"Директория с данными образца {main_data_path} должна содержать только" \
        f" директории с шумными и чистыми проекциями," \
        f" найдено несколько директорий для зашумленных проекций {noisy_projections_dir}"
    assert len(noisy_projections_dir) > 0, \
        f"Директория с данными образца {main_data_path} не содержит директории зашумленных проекций"
    noisy_projections_dir = noisy_projections_dir[0]

    assert len(listdir(join(main_data_path, noisy_projections_dir))) > 0, \
        f"Директория зашумленных проекций {noisy_projections_dir} не содержит проекций"

    del projections_dirs

    main_logs_path = join(sample_dir_path, args.logs_dir_name)
    os.makedirs(main_logs_path, exist_ok=True)

    if args.cpu:
        device = 'cpu'
    else:
        device = 'gpu'

    if device == 'cpu':
        print("Training on CPU...")
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        print("Training on GPU...")

    modes_correction_list = []
    # Любой раздлитель будет правильно обработан
    list(map(
        lambda mode: list(map(
            lambda mode_: modes_correction_list.append(mode_.strip().lower()),
            mode.strip().split(' ')
        )), args.modes_correction.split(',')
    ))
    modes_correction_list = np.array(modes_correction_list)
    modes_correction_list = modes_correction_list[modes_correction_list != ""]

    main(
        device, args.global_random_seed,
        args.reconstruction_library, args.reconstruction_method,
        main_data_path, clean_projections_dir, noisy_projections_dir,
        main_logs_path,
        args.center, args.rec_width,
        modes_correction_list, args.num_splits,
        args.normalize_rec, args.hist_percent,
        args.circ_mask_ratio,
        args.output_proj_prefix, args.log_rec_idx,
    )
