import numpy as np
import tomopy
import foam_ct_phantom
from noise2inverse import tiffs, noise, fig
import h5py
import tifffile
import random
import sys
import os
from os import makedirs, getcwd
from os.path import join, split, exists, isdir, realpath
import argparse
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from pathlib import Path


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    if len(
            list(filter(
                lambda name: name == 'tensorflow',
                [m.__name__ for m in sys.modules.values() if m]
    ))) > 0:
        os.environ['TF_DETERMINISTIC_OPS'] = '1'  # tf gpu fix seed, please `pip install tensorflow-determinism` first

def get_sample_name(projs_path):
    sample_name = split(projs_path)[-1]
    if sample_name.find('.') != -1:
        sample_name = sample_name[:sample_name.find('.')]
    return sample_name

def main(
    device, global_random_seed,
    main_data_path, main_logs_path,
    output_projs_prefix, log_proj_idx
):
    print(
        f'\n\t\tdevice = {device}',
        f'global_random_seed={global_random_seed}',
        f'main_data_path={main_data_path}',
        f'main_logs_path={main_logs_path}',
        f'output_projs_prefix={output_projs_prefix}',
        f'log_proj_idx={log_proj_idx}\n', sep='\n\t\t'
    )

    print()
    sample_name = get_sample_name(projs_path)
    print(sample_name)

    setup_seed(global_random_seed)


    out_projs_noisy_dir = join(main_data_path, "projections_noisy")
    Path(out_projs_noisy_dir).mkdir(exist_ok=True)
    
    # Phantom parameters
    if sample_name == "phantom":
        # ToDo - вынести в аогументы запуска?
        num_voxels = 512
        num_slices = 512
        num_angles = 1024
        supersampling = 2

        # Noise
        photon_count = 10
        attenuation_factor = 2.76  # corresponds to absorption of 50% # ToDo attenuation_factor - ???
        # Paths
        out_projs_dir = join(main_data_path, "projections_clean")
        out_projs_path = join(out_projs_dir, "clean_projs.h5")
        Path(out_projs_dir).mkdir(exist_ok=True)

        '''
        https://github.com/dmpelt/foam_ct_phantom?ysclid=l7rnmey4hs872099801  
        https://dmpelt.github.io/foam_ct_phantom/  
        https://dmpelt.github.io/foam_ct_phantom/auto_examples/01_generate_phantom.html#sphx-glr-auto-examples-01-generate-phantom-py - 
        Часть с генерацией здесь опущена  
        Сразу используем готовую синтетическую модель
    
        ParallelGeometry - https://github.com/dmpelt/foam_ct_phantom/blob/master/foam_ct_phantom/geometry.py  
        По всей видимости, есть возможность генерировать 3D проекции
        '''
        print(out_projs_dir)
        print(Path(out_projs_dir).expanduser())
        print(Path(out_projs_dir).expanduser().resolve())


        # Read in phantom sphere locations:
        phantom = foam_ct_phantom.FoamPhantom(projs_path)

        # Define projection geometry
        pg = foam_ct_phantom.ParallelGeometry(
            nx=num_voxels * 3 // 2,  # 768
            ny=num_slices,  # 512
            angles=np.linspace(0, np.pi, num_angles, False),  # num_angles = 1024
            pixsize=2 / num_voxels,  # ToDo pixsize = 0,00390625 ?
            supersampling=supersampling,
        )

        # Save/generate projections
        # if not Path(out_proj_path).exists():
        print("Generating projections")
        phantom.generate_projections(out_projs_path, pg)
        projs = foam_ct_phantom.load_projections(out_projs_path)
        print("Saving tiff stack of projections")
        tiffs.save_stack(out_projs_dir, projs)

        # Add noise

        '''
        transmittance - пропускаемость
        absorption - поглощение, абсорбация
        '''
        projs *= attenuation_factor  # attenuation_factor = 2.76
        print(f"Absorption: {noise.absorption(projs) * 100:0.0f}%")
        print(f"Shape:      {projs.shape}")
        projs_noisy = noise.apply_noise(projs, photon_count)  # photon_count - 10
        projs_noisy /= attenuation_factor
        projs_noisy = projs_noisy.astype(np.float32)

        # Save noisy projections
        tiffs.save_stack(out_projs_noisy_dir, projs_noisy)

        # Show results
        projs_clean = tifffile.imread(join(out_projs_dir, f"{output_projs_prefix}_{log_proj_idx:05d}.tif"))
        projs_noisy = tifffile.imread(join(out_projs_noisy_dir, f"{output_projs_prefix}_{log_proj_idx:05d}.tif"))

        fig.plot_imgs(
            clean=projs_clean,
            noisy=projs_noisy,
            width=6.0,
            name=join(main_logs_path, "projs_clean_noisy.png")
        )


    else:
        # Define projection geometry
        print("-----------------")
        # Save/generate projections
        print(os.path.exists(projs_path))
        '''
        projs_noisy, flat, dark, theta = dxchange.read_aps_32id(fname=str(projs_path),)
        plt.figure(figsize=(10, 15))
        plt.imshow(projs_noisy[:, 0, :], cmap='gray')
        plt.show()
        '''
        with h5py.File(str(projs_path), 'r') as f:
            projs_noisy = f['exchange']['data'][:]
            flat = f['exchange']['data_white'][:]
            dark = f['exchange']['data_dark'][:]
            theta = f['exchange']['theta'][:]

            print(
                "\nProj shape is (ang, z, x):", projs_noisy.shape,
                "flat shape is", flat.shape,
                "dark shape is", dark.shape,
                "theta shape is", theta.shape
            )

            print(
                "\nisnan or isinf intensity pixels in dark %=",
                np.sum(np.logical_or(np.isnan(dark), np.isinf(dark))) / (
                        dark.shape[0] * dark.shape[1] * dark.shape[2]) * 100
            )
            print(
                "\nisnan or isinf intensity pixels in white %=",
                np.sum(np.logical_or(np.isnan(flat), np.isinf(flat))) / (
                        flat.shape[0] * flat.shape[1] * flat.shape[2]) * 100
            )
            print(theta)
            print("Saving tiff stack of projections")

            # projs = (projs - np.mean(dark, axis=0)) / (np.mean(white, axis=0) - np.mean(dark, axis=0))
            projs_noisy = tomopy.normalize(projs_noisy, flat=flat, dark=dark, cutoff=None, ncore=None, out=None)

            projs_noisy = tomopy.minus_log(projs_noisy)
            print(
                "\nisnan or isinf intensity pixels in projs %=",
                np.sum(np.logical_or(np.isnan(projs_noisy), np.isinf(projs_noisy))) / (
                            projs_noisy.shape[0] * projs_noisy.shape[1] * projs_noisy.shape[2]) * 100
            )
            projs_noisy[np.logical_or(np.isnan(projs_noisy), np.isinf(projs_noisy))] = 0
            print("Saving tiff stack of projections")
            tiffs.save_stack(out_projs_noisy_dir, projs_noisy, prefix=output_projs_prefix, exist_ok=True, parents=False)

            # Если считаны через dxchange, требуется расскоментить:
            #projs_noisy = np.moveaxis(projs_noisy, 1, 0)
            # Save noisy projections
            tiffs.save_stack(out_projs_noisy_dir, projs_noisy, prefix=output_projs_prefix)

            fig.plot_imgs(
                white=flat[10],
                dark=dark[10],
                width=20.0,
                name=join(main_logs_path, f"data_samples_white_black.png")
            )

            fig.plot_imgs(
                sample=tifffile.imread(join(out_projs_noisy_dir, f"{output_projs_prefix}_{log_proj_idx:05d}.tif")),
                width=20.0,
                name=join(main_logs_path, f"data_samples.png")
            )

if __name__ == '__main__':
    print("------------------------------------------------------------------------------------------------")
    parser = argparse.ArgumentParser()
    # -----------------------------------------------------------------------------------------
    # Настройка окружения и инициализации обучаемых параметров
    parser.add_argument('-cpu', '--cpu', default=1, type=int, choices=[0, 1], required=False)
    parser.add_argument('-global_r_s', '--global_random_seed', default=42, type=int, required=False)
    # -----------------------------------------------------------------------------------------
    # Наборы данных и сохранение результата
    parser.add_argument('-dir', '--working_dir', default=join(".", "phantom"), type=str, required=False)
    parser.add_argument('-projs', '--projs_path', default="phantom.h5", type=str, required=False)
    parser.add_argument('-data_dir', '--data_dir_name', default="data", type=str, required=False)
    parser.add_argument('-logs_dir', '--logs_dir_name', default="logs", type=str, required=False)
    parser.add_argument('-prefix', '--output_projs_prefix', default="output", type=str, required=False)
    parser.add_argument('-log_idx', '--log_proj_idx', default=50, type=int, required=False)

    args = parser.parse_args()

    if args.cpu:
        device = 'cpu'
    else:
        device = 'gpu'
    # ToDo - настройка устройства под pytorch
    # ToDo - запуск на GPU
    if len(
            list(filter(
                lambda name: name == 'tensorflow',
                [m.__name__ for m in sys.modules.values() if m]
    ))) > 0:
        if device == 'cpu':
            print("Training on CPU...")
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        else:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            print("Training on GPU...")

    print("working_dir:\t", args.working_dir)
    working_dir = realpath(args.working_dir)
    print("working_dir:\t", working_dir)
    projs_path = args.projs_path
    print("projs_path:\t", projs_path)
    print("getcwd:\t", getcwd())
    print(split(projs_path))
    print(os.path.abspath(projs_path))
    # Если указано наименование файла - первый элемент пути ''
    if not (os.path.isabs(projs_path) or projs_path.startswith('.')):
        # Пытаемся считать из директории, из которой был запущен код
        # getcwd() выдаст путь, из которого производится запуск скрипта
        if exists(working_dir):
            if getcwd() == working_dir:
                assert exists(join(working_dir, projs_path)),\
                    f"exists({join(working_dir, projs_path)}):  Файл {projs_path} отсутстует в указанной рабочей директории проекта {working_dir}"
                projs_path = join(working_dir, projs_path)
            else:
                assert exists(join(getcwd(), projs_path)) or exists(join(working_dir, projs_path)),\
                    f"exists({join(getcwd(), projs_path)}) or exists({join(working_dir, projs_path)}): Файл {projs_path} отсутстует в указанной рабочей директории проекта {working_dir} "\
                    f"и в текущем катологе {getcwd()}"
                projs_path = join(working_dir, projs_path) if exists(join(working_dir, projs_path))\
                                                                else join(getcwd(), projs_path)
        else:
            assert exists(join(getcwd(), projs_path)), \
                f"exists({join(getcwd(), projs_path)}): Файл {projs_path} отсутстует в текущей директории {getcwd()}, укажите полный путь до файла"
            projs_path = join(getcwd(), projs_path)
    else:
        assert exists(projs_path), \
            f"По указанному пити {join(split(projs_path)[:-1])} не найден файл {split(projs_path)[-1]}"

    if not exists(working_dir):
        makedirs(working_dir)

    sample_name = get_sample_name(projs_path)
    print(sample_name)
    main_data_path = join(working_dir, args.data_dir_name)
    main_logs_path = join(working_dir, args.logs_dir_name)
    os.makedirs(main_data_path, exist_ok=True)
    os.makedirs(main_logs_path, exist_ok=True)

    main(
        device, args.global_random_seed,
        main_data_path, main_logs_path,
        args.output_projs_prefix, args.log_proj_idx
    )