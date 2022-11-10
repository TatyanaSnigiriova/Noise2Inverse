import argparse
import os
from os import listdir
from os.path import join, exists, split, realpath
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from noise2inverse.datasets import (
    TiffDataset,
    Noise2InverseDataset,
)

import tifffile
import matplotlib.pyplot as plt
import numpy as np
import warnings

from tqdm import tqdm
from noise2inverse import fig, tiffs, noise
from my_utils.log_histogram import *

warnings.filterwarnings("ignore", category=FutureWarning)


def main(
        device, multi_gpu, global_rs,
        main_sub_recs_data_path, main_model_chkpt_path, main_logs_path,
        num_splits,
        strategy, model_name,
        epochs, batch_size, retrain_model,
        clean_reconstructions_path, noisy_reconstructions_path,
        denoised_recs_dir_path, output_denoised_recs_prefix, log_rec_idx,
):
    print(
        f'\n\t\tdevice = {device}',
        f'global_rs={global_rs}',
        f'main_sub_recs_data_path={main_sub_recs_data_path}',
        f'main_model_chkpt_path={main_model_chkpt_path}',
        f'main_logs_path={main_logs_path}',
        f'splits={num_splits}',
        f'strategy={strategy}',
        f'model_name={model_name}',
        f'denoised_recs_dir_path={denoised_recs_dir_path}',
        f'output_denoised_recs_prefix={output_denoised_recs_prefix}',
        f'log_rec_idx={log_rec_idx}\n', sep='\n\t\t'
    )
    weights_path = Path(join(main_model_chkpt_path, "weights.torch"))

    # Scale pixel intensities during training such that its values roughly occupy the range [0,1].
    # This improves convergence.
    #data_scaling = 200

    datasets = [TiffDataset(join(main_sub_recs_data_path, f"{j}", "*.tif")) for j in range(num_splits)]
    train_ds = Noise2InverseDataset(*datasets, strategy=strategy)

    print(train_ds.num_slices, train_ds.num_splits)
    print(datasets[3].paths)

    fig.plot_imgs(
        input=train_ds[0][0].detach().squeeze(),
        target=train_ds[0][1].detach().squeeze(),
        # vmin=-0.004,
        # vmax=0.008,
        name=join(main_logs_path, "input_target.png")
    )

    # Dataloader and modelwork:
    dl = DataLoader(train_ds, batch_size, shuffle=True, )
    # Option a) Use MSD modelwork
    if model_name == "msd":
        from msd_pytorch import MSDRegressionModel

        model = MSDRegressionModel(1, 1, 100, 1, parallel=multi_gpu)
        model = model.model
        optimizer = model.optimizerizer

    # Option b) Use UNet
    if model_name == "unet":
        from noise2inverse.unet import UNet

        model = UNet(1, 1).to(device)  # 1 input channel, 1 output channel
        if multi_gpu:
            model = nn.DataParallel(model)

        optimizer = torch.optim.Adam(model.parameters())

    # Option c) Use DnCNN
    if model_name == "dncnn":
        from noise2inverse import DnCNN

        model = DnCNN(1).to(device)  # 1 input channel, 1 output channel
        if multi_gpu:
            model = nn.DataParallel(model)

        optimizer = torch.optim.Adam(model.parameters())
    print(f"chkpt_name:", [
        chkpt_name[chkpt_name.find("epoch_") + len("epoch_"): chkpt_name.find(".torch")]
        for chkpt_name in listdir(main_model_chkpt_path)
        if chkpt_name.find("epoch_") != -1
    ])



    if not retrain_model:
        if exists(main_model_chkpt_path):
            os.rmdir(main_model_chkpt_path)
            os.makedirs(main_model_chkpt_path)
        start_epoch = 0
    else:
        to_load_weights = None
        if exists(join(main_model_chkpt_path, "weights.torch")):
            to_load_weights = join(main_model_chkpt_path, "weights.torch")
        else:
            chkpt_epochs = list(map(
                lambda chkpt_name: chkpt_name[chkpt_name.find("epoch_") + len("epoch_"): chkpt_name.find(".torch")],
                listdir(main_model_chkpt_path)
            ))
            if len(chkpt_epochs) > 0:
                to_load_weights = f"weights_epoch_{max(chkpt_epochs)}.torch"

        if to_load_weights:
            print(f"Загрузка весов {to_load_weights}")
            checkpoint = torch.load(to_load_weights, map_location=device)
            model.load_state_dict(checkpoint['state_dict']) #  model_state_dict
            optimizer.load_state_dict(checkpoint['optimizer']) # optimizer_state_dict
            start_epoch = checkpoint['epoch'] + 1
            #loss = checkpoint['loss']
        else:
            start_epoch = 0
            print("Задано дообучение модели, но предобученных весов по указаному пути не найдено")


    # The dataset contains multiple input-target pairs for each slice.
    # Therefore, we divide by the number of splits to obtain the effective number of epochs.
    train_epochs = epochs // num_splits
    epoch = start_epoch
    print("current epoch:", epoch, "\tstart_epoch + train_epochs:", start_epoch + train_epochs)
    # training loop
    for epoch in range(epoch, epoch + train_epochs):
        # Train
        for (inp, tgt) in tqdm(dl):
            #inp = inp * data_scaling
            #tgt = tgt * data_scaling
            # inp = inp.cuda(non_blocking=True) * data_scaling
            # tgt = tgt.cuda(non_blocking=True) * data_scaling
            inp = inp.to(device)
            tgt = tgt.to(device)
            # Do training step with masking
            output = model(inp)
            loss = nn.functional.mse_loss(output, tgt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Save modelwork
        torch.save(
            {
                "epoch": int(epoch),
                #"loss": float(loss),
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict()
            },
            join(main_model_chkpt_path, f"weights_epoch_{epoch}.torch")
        )
        # ToDo:
        #  * Loss на валидации
        #  * Подсчет метрик (Какие метрики, что с чем сравнивать без clean image?)

    torch.save(
        {
            "epoch": int(epoch),
            # "loss": float(loss),
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()
        },
        join(main_model_chkpt_path, "weights.torch")
    )

    # Parameters
    batch_size = num_splits
    # Use modelwork whose parameters are stored in weights/weights.torch
    # By default that is an msd modelwork

    # Scale pixel intensities in the same way as during training.
    #data_scaling = 200

    ds = Noise2InverseDataset(*datasets, strategy=strategy)

    # Dataloader and modelwork:
    dl = DataLoader(ds, batch_size, shuffle=False, )

    # Load weights
    # state = torch.load(weights_path)
    # model.load_state_dict(state["state_dict"])
    model = model.to(device)

    # Put modelwork in evaluation mode: this should be done when the modelwork uses batch norm for instance.
    model.eval()
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dl)):
            inp, _ = batch
            inp = inp.to(device) #* data_scaling
            out = model(inp)
            # Take mean over batch dimension (splits):
            out = out.mean(dim=0) #/ data_scaling
            # Obtain 2D numpy array
            out_np = out.detach().cpu().numpy().squeeze()
            out_path = join(denoised_recs_dir_path, f"{output_denoised_recs_prefix}_{i:05d}.tif")
            tifffile.imsave(out_path, out_np)

    log_images = dict()
    # ToDo - префикс для чистых и грязных реконструкций не обязательно совпадает с output_denoised_recs_prefix
    if clean_reconstructions_path:
        log_images['slice_clean'] = tifffile.imread(
            join(clean_reconstructions_path, f"{output_denoised_recs_prefix}_{log_rec_idx:05d}.tif")
        )
    if noisy_reconstructions_path:
        log_images['slice_noisy'] = tifffile.imread(
            join(noisy_reconstructions_path, f"{output_denoised_recs_prefix}_{log_rec_idx:05d}.tif")
        )
    log_images['slice_denoised'] = tifffile.imread(
        join(denoised_recs_dir_path, f"{output_denoised_recs_prefix}_{log_rec_idx:05d}.tif")
    )

    fig.plot_imgs(
        **log_images,
        width=20.0,
        # vmin=0,
        # vmax=0.004,
        name=join(
            main_logs_path,
            f"{'clean_' if clean_reconstructions_path else ''}" +
            f"{'noisy_' if noisy_reconstructions_path else ''}" +
            "denoised.png"
        )
    )

    axes = plt.gcf().axes
    print(len(axes))
    for ax in axes[:2]:
        print(len(ax.images))
        fig.add_zoom_bubble(ax.images[0], roi=(.4, .3), zoom=4, inset_radius=.25)
    plt.savefig(join(
        main_logs_path,
        f"{'clean_' if clean_reconstructions_path else ''}" +
        f"{'noisy_' if noisy_reconstructions_path else ''}" +
        "denoised_zoom.png"
    ))


if __name__ == '__main__':
    print("------------------------------------------------------------------------------------------------")
    parser = argparse.ArgumentParser()
    # -----------------------------------------------------------------------------------------
    # Настройка окружения
    parser.add_argument('-cpu', '--cpu', default=1, type=int, choices=[0, 1], required=False)
    # ToDo - получить список доступных видеокарт
    parser.add_argument('-device', '--device_num', choices=[-1, 0, 1], default=0, type=int, required=False)
    # if -1 - Multi GPU
    parser.add_argument('-global_rs', '--global_random_seed', default=42, type=int, required=False)
    # Аргументы, чтобы получить наименование директории с реконструкциями
    parser.add_argument('-rec_lib', '--reconstruction_library', default="noise2inverse", type=str, required=False)
    parser.add_argument('-rec_method', '--reconstruction_method', default="fbp", type=str, required=False)
    parser.add_argument('-modes', '--modes_correction', default="circ_mask", type=str, required=False)
    parser.add_argument('-hist', '--hist_percent', default=100, type=int, required=False)
    parser.add_argument('-norm', '--normalize_rec', default=0, type=int, choices=[0, 1], required=False)
    # -----------------------------------------------------------------------------------------
    # Наборы данных и сохранение результата
    parser.add_argument('-sample_dir', '--sample_dir_path', default=join(".", "phantom"), type=str, required=False)
    parser.add_argument('-data_dir', '--data_dir_name', default="data", type=str, required=False)
    parser.add_argument('-models_dir', '--models_dir_name', default="models", type=str, required=False)
    parser.add_argument('-logs_dir', '--logs_dir_name', default="logs", type=str, required=False)
    parser.add_argument('-prefix', '--output_denoised_recs_prefix', default="output", type=str, required=False)
    parser.add_argument('-log_idx', '--log_rec_idx', default=50, type=int, required=False)
    # -----------------------------------------------------------------------------------------
    parser.add_argument('-strategy', '--strategy', default="X:1", type=str, choices=["X:1", "x:1", "1:X", "1:x"],
                        required=False)
    parser.add_argument('-model', '--model_name', default="unet", type=str, required=False)
    #     modelwork = "unet"  # msd or unet or dncnn
    parser.add_argument('-epochs', '--num_epochs', default=40, type=int, required=False)
    parser.add_argument('-batch', '--batch_size', default=4, type=int, required=False)
    # NOTE: reduce the batch size to fit training in GPU memory for unet and dncnn
    parser.add_argument('-retrain', '--retrain_model', default=0, choices=[0, 1], type=int, required=False)
    # По умолчанию дообучение модели
    # -----------------------------------------------------------------------------------------

    # Для минимакс-нормализации будет использован весь диапазон интенсивностей

    args = parser.parse_args()
    multi_gpu = False
    if args.cpu:
        device = 'cpu'
    else:
        device = 'gpu'

    if device == 'cpu' or not torch.cuda.is_available():
        print("Training on CPU...")
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        device = torch.device('cpu')
    else:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        print("Training on GPU...")
        print("torch.cuda.is_available()", torch.cuda.is_available())
        print("torch.cuda.device_count()", torch.cuda.device_count())
        print("torch.cuda.current_device()", torch.cuda.current_device())
        for i in range(torch.cuda.device_count()):
            print(f"torch.cuda.device({i})", torch.cuda.device(i))
            print(f"torch.cuda.get_device_name({i})", torch.cuda.get_device_name(i))

        # ToDo - запуск на конкретном устройстве
        # ToDo - множество GPU
        assert torch.cuda.device_count() > args.device_num
        if args.device_num == -1:
            multi_gpu = True
            device = torch.device('cuda')
        else:
            device = torch.device('cuda', args.device_num)

    print("sample_dir_path:\t", args.sample_dir_path)
    sample_dir_path = realpath(args.sample_dir_path)
    print("sample_dir_path:\t", sample_dir_path)
    assert exists(sample_dir_path), f"Указанный путь к директории образца {sample_dir_path} не существует"

    main_data_path = join(sample_dir_path, args.data_dir_name)
    assert exists(main_data_path), f"По указанному пути {main_data_path} не найдена директория с данными образца"

    # Формируем путь до реконструкций
    # ToDo - подумать над тем, как перейти к наименованию директории (нужно как-то упростить имя)
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

    data_dir_pattern = f"lib={args.reconstruction_library}_method={args.reconstruction_method}" + \
                       f"_norm={args.normalize_rec}_modes={'&'.join(modes_correction_list)}_hist={args.hist_percent}%"
    reconstructions_dir_path = join(main_data_path, data_dir_pattern, "reconstructions")
    assert exists(reconstructions_dir_path), \
        f"Директория с данными образца {main_data_path} не содержит директории с .tif реконструкциями"

    # Найдем разбиения и проверим, что директории реконструкций не пустые
    sub_reconstructions_dirs = list(filter(
        lambda sub_reconstruction_dir: sub_reconstruction_dir.isdigit(),
        listdir(reconstructions_dir_path)
    ))
    print(f"Найдено {len(sub_reconstructions_dirs)} директорий суб-реконструкций")
    # ToDo - Проверить терминологию
    assert len(sub_reconstructions_dirs) > 0, \
        f"Директория реконструкций {reconstructions_dir_path} должна содержать " \
        f"директории с реконструкциями, восстановленными на некотором подмножестве проекций по углам," \
        f"не найдено ни одной директори суб-реконструкций.\n" \
        f"{f'Содержимое директории:{listdir(reconstructions_dir_path)}' if len(listdir(reconstructions_dir_path)) > 0 else ''}"

    not_empty_sub_reconstructions_dirs = list(filter(
        lambda sub_reconstructions_dir: len(listdir(join(reconstructions_dir_path, sub_reconstructions_dir))) > 0,
        sub_reconstructions_dirs
    ))
    print("sub_reconstructions_dirs", sub_reconstructions_dirs)
    print("not_empty_sub_reconstructions_dirs", not_empty_sub_reconstructions_dirs)
    assert len(not_empty_sub_reconstructions_dirs) == len(sub_reconstructions_dirs), \
        f"Некоторые из директорий суб-реконструкций являются пустыми: " \
        f"{set(sub_reconstructions_dirs).difference(set(not_empty_sub_reconstructions_dirs))}"

    not_empty_sub_reconstructions_dirs = list(map(int, not_empty_sub_reconstructions_dirs))
    correct_splits_names = list(range(0, max(list(map(int, not_empty_sub_reconstructions_dirs))) + 1))
    assert len(set(not_empty_sub_reconstructions_dirs).intersection(set(correct_splits_names))) == len(correct_splits_names), \
        f"Наименования директорий суб-реконструкций должны представлять собой ряд натуральных чисел"

    # Ожидаю найти максимум одну чистую и максимум одну зашумленную директорию проекций
    clean_reconstructions_dir = list(
        filter(lambda reconstructions_dir: reconstructions_dir.rfind('clean') != -1, listdir(reconstructions_dir_path)))
    print("clean_reconstructions_dir:", clean_reconstructions_dir)
    assert len(clean_reconstructions_dir) < 2, \
        f"Директория с данными образца {reconstructions_dir_path} должна содержать только" \
        f" директории с шумными и чистыми реконструкциями и суб рконструкциями," \
        f" найдено несколько директорий для чистых реконструкций {clean_reconstructions_dir}"

    if len(clean_reconstructions_dir) == 1:
        clean_reconstructions_dir = clean_reconstructions_dir[0]
        assert len(listdir(join(reconstructions_dir_path, clean_reconstructions_dir))) > 0, \
            f"Директория чистых реконструкций {clean_reconstructions_dir} не содержит реконструкций"
        clean_reconstructions_path = join(reconstructions_dir_path, clean_reconstructions_dir)

    else:
        print("Не найдено директории для чистых от шума (таргетных) реконструкций.")
        clean_reconstructions_path = None


    noisy_reconstructions_dir = list(
        filter(lambda reconstructions_dir: reconstructions_dir.rfind('noisy') != -1, listdir(reconstructions_dir_path)))
    print("noisy_reconstructions_dir:", noisy_reconstructions_dir)
    assert len(noisy_reconstructions_dir) < 2, \
        f"Директория с данными образца {reconstructions_dir_path} должна содержать только" \
        f" директории с шумными и чистыми реконструкциями и суб рконструкциями," \
        f" найдено несколько директорий для зашумленных реконструкций по всем углам {noisy_reconstructions_dir}"
    assert len(noisy_reconstructions_dir) > 0, \
        f"Директория с данными образца {reconstructions_dir_path} не содержит директории зашумленных реконструкций"

    noisy_reconstructions_dir = noisy_reconstructions_dir[0]

    assert len(listdir(join(reconstructions_dir_path, noisy_reconstructions_dir))) > 0, \
        f"Директория зашумленных реконструкций {noisy_reconstructions_dir} не содержит реконструкций"
    noisy_reconstructions_path = join(reconstructions_dir_path, noisy_reconstructions_dir)

    denoised_recs_dir_path = join(main_data_path, data_dir_pattern, "denoised")
    os.makedirs(denoised_recs_dir_path, exist_ok=True)

    strategy = args.strategy.upper()
    model_name = args.model_name.lower()
    assert model_name in ["unet"]

    main_logs_path = join(sample_dir_path, args.logs_dir_name, data_dir_pattern, args.models_dir_name, model_name,
                          strategy.replace(":", "&"))
    os.makedirs(main_logs_path, exist_ok=True)

    main_model_chkpt_path = join(sample_dir_path, args.models_dir_name, data_dir_pattern, model_name,
                                 strategy.replace(":", "&"))
    os.makedirs(main_model_chkpt_path, exist_ok=True)

    main(
        device, multi_gpu, args.global_random_seed,
        reconstructions_dir_path, main_model_chkpt_path, main_logs_path,
        len(not_empty_sub_reconstructions_dirs),
        strategy, model_name,
        args.num_epochs, args.batch_size, args.retrain_model,
        clean_reconstructions_path, noisy_reconstructions_path,
        denoised_recs_dir_path, args.output_denoised_recs_prefix, args.log_rec_idx,
    )
