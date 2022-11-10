import astra
import numpy as np
import torch
from tqdm import tqdm
import tomosipo as ts


def filter_in_real_filterspace(n):
    # Makes filter in real filter space.
    # Complex component equals zero.
    filter = np.zeros(n)
    filter[0] = 0.25
    # even indices are zero
    # for odd indices j, filter[j] equals
    #   -1 / (pi * j) ** 2,          when 2 * j <= n
    #   -1 / (pi * (n - j)) ** 2,    when 2 * j >  n

    odd_indices = np.arange(1, n, 2)
    cond = 2 * odd_indices > n
    odd_indices[cond] = n - odd_indices[cond]
    filter[1::2] = -1 / (np.pi * odd_indices) ** 2

    return filter


def filter_proj_data(sino):
    """Filters projection data for FBP

    Uses Ram-Lak filter.

    Follows the approach of:

    Zeng GL. Revisit of the Ramp Filter. IEEE Trans Nucl Sci. 2015;62(1):131–136. doi:10.1109/TNS.2014.2363776

    :param sino: `torch.tensor` projection data
    :returns: `torch.tensor` filtered projection data
    :rtype:

    """
    normalized = False
    print("\nsino:", sino.shape)

    (num_slices, num_angles, num_pixels) = sino.shape
    # We must have:
    # 1) num_pixels + num_padding is even (because rfft wants the #input elements to be even)
    # 2) num_padding // 2 must equal at least num_pixels (not so sure about this actually..)
    if num_pixels % 2 == 0:
        num_padding = num_pixels
    else:
        num_padding = num_pixels + 2
    # num_padding < num_padding_left
    num_padding_left = num_padding // 2
    print("num_padding_left:", num_padding_left, "\tnum_padding:", num_padding)
    # M is always even
    M = num_pixels + num_padding
    print("\nM:\t", M)
    tmp_sino = sino.new_zeros((num_slices, num_angles, M))
    tmp_sino[:, :, num_padding_left:num_padding_left + num_pixels] = sino
    print("\ntmp_sino:\t", tmp_sino.shape)
    # print("\n", tmp_sino)

    # XXX: Consider using torch.stft. This might save us from doing the padding.
    # https://pytorch.org/docs/1.7.1/generated/torch.rfft.html?highlight=rfft#torch.rfft
    fourier_sino = torch.rfft(
        tmp_sino,
        signal_ndim=1,  # ?
        normalized=normalized,
        onesided=True  # ?
    )
    print("\nfourier_sino:\t", fourier_sino.shape)
    # print("\n", fourier_sino)

    real_filter = filter_in_real_filterspace(M).astype(np.float32)
    print("\nreal_filter:\t", real_filter.shape)
    # print("\n", real_filter)
    fourier_filter = torch.rfft(
        torch.from_numpy(real_filter),
        signal_ndim=1,
        normalized=normalized,
        onesided=True  # ?
    )
    print("\nfourier_filter:\t", fourier_filter.shape)
    # print("\n", fourier_filter)
    # Make complex dimension equal to real dimension
    # print("\nfourier_filter:\t", fourier_filter.shape)
    # print("\n", fourier_filter)
    print("\nfourier_sino:\t", fourier_sino.shape)
    # print("\n", fourier_sino)
    print(fourier_sino[:, :, :, 0].shape, "\t*\t", fourier_filter[:, 0].shape)

    new_fourier_sino = torch.zeros(fourier_sino.shape)
    new_fourier_sino[:, :, :, 0] = fourier_sino[:, :, :, 0] * fourier_filter[:, 0] \
                                   - fourier_sino[:, :, :, 1] * fourier_filter[:, 1]
    new_fourier_sino[:, :, :, 1] = fourier_sino[:, :, :, 1] * fourier_filter[:, 0] \
                                   + fourier_sino[:, :, :, 0] * fourier_filter[:, 1]
    '''
    # defaulf
    fourier_filter = fourier_filter[:, 0][:, None]
    new_fourier_sino = fourier_sino * fourier_filter 
    #'''  # Визуально разницы замечено не было

    print("\nnew_fourier_sino:\t", new_fourier_sino.shape)
    # print("\n", new_fourier_sino)
    # https://pytorch.org/docs/1.7.1/generated/torch.irfft.html#torch.irfft
    tmp_filtered = torch.irfft(
        new_fourier_sino,
        signal_ndim=1,
        # signal_sizes=(M,), # default signal_sizes=(M,)
        normalized=normalized,
        onesided=True
    )
    print("\ntmp_filtered:\t", tmp_filtered.shape)
    # print("\n", tmp_filtered)
    # tmp_filtered /= num_angles #/ np.pi
    print("\ntmp_filtered:\t", tmp_filtered.shape)
    # print("\n", tmp_filtered)
    filtered = tmp_filtered.new_empty(sino.shape)
    filtered[:] = tmp_filtered[:, :, num_padding_left:num_padding_left + num_pixels]
    print("\nfiltered:\t", filtered.shape)
    # print("\n", filtered)
    return filtered


def fbp(A, sino):
    # Filter
    filtered_sino = filter_proj_data(
        torch.from_numpy(sino)
    ).detach().numpy()
    # Reconstruct
    return A.T(filtered_sino)
    # ToDo Landweber iteration
    '''
    # https://arxiv.org/pdf/1812.00272v1.pdf  
    eta = 7.9e-3
    x_rec = np.zeros(A.domain_shape, np.float32)  # Zero-initialize the volume
    for i in range(20):
       x_rec = x_rec + eta * A.T(filtered_sino - A(x_rec))
    return x_rec
    '''
