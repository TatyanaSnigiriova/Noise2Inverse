#
#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016, European Synchrotron Radiation Facility
# Main author: Pierre Paleo <pierre.paleo@esrf.fr>
#

from __future__ import division
import numpy as np
from math import sqrt, pi
import astra


def nextpow2(N):
    p = 1
    while p < N:
        p *= 2
    return p


class AstraToolbox:
    """
    ASTRA toolbox wrapper for parallel beam geometry.
    """

    def __init__(self, slice_shape, angles, dwidth=None, rot_center=None, fullscan=False, cudafbp=False,
                 super_sampling=None):
        """
        Create a tomography parallel beam geometry.

        Parameters
        -----------
        slice_shape: int or tuple
            Shape of the slice. Can be in the form (n_x, n_y) or N.
            If the argument is an integer N, the slice is assumed to be square of dimensions (N, N).
            Mind that if providing (n_x, n_y), n_x is the number of columns of the image.
        angles: integer or numpy.ndarray
            Projection angles in radians.
            If this argument is an integer, the projections angles are linear between [0, pi[ if fullscan=False,
            or in [0, 2*pi[ if fullscan=True.
            If this argument is a 1D array, this means that custom angles (in radians) are provided.
        dwidth: (optional) integer
            detector width (number of pixels). If not provided, max(n_x, n_y) is taken.
        rot_center: (optional) float
            user-defined rotation center. If not provided, dwidth/2 is taken.
        fullscan: (optional) boolean, default is False
            if True, use a 360 scan geometry.
        cudafbp: (optionnal) boolean, default is False
            If True, use the built-in FBP of ASTRA instead of using Python to filter the projections.
        super_sampling: integer
            Detector and Pixel supersampling
        """

        if isinstance(slice_shape, int):
            n_x, n_y = slice_shape, slice_shape
        else:
            slice_shape = tuple(slice_shape)
            n_x, n_y = slice_shape
        if dwidth is None: dwidth = max(n_x, n_y)
        angle_max = np.pi
        if fullscan: angle_max *= 2

        if isinstance(angles, int):
            angles = np.linspace(0, angle_max, angles, False)
        n_angles = angles.shape[0]

        self.vol_geom = astra.create_vol_geom(n_x, n_y)
        self.proj_geom = astra.create_proj_geom('parallel', 1.0, dwidth, angles)

        if rot_center:
            o_angles = np.ones(n_angles) if isinstance(n_angles, int) else np.ones_like(n_angles)
            self.proj_geom['option'] = {'ExtraDetectorOffset': (rot_center - n_x / 2.) * o_angles}
        self.proj_id = astra.create_projector('cuda', self.proj_geom, self.vol_geom)

        # vg : Volume geometry
        self.vg = astra.projector.volume_geometry(self.proj_id)
        # pg : projection geometry
        self.pg = astra.projector.projection_geometry(self.proj_id)

        # ---- Configure Projector ------
        # sinogram shape
        self.sshape = astra.functions.geom_size(self.pg)
        # Configure projector
        self.cfg_proj = astra.creators.astra_dict('FP_CUDA')
        self.cfg_proj['ProjectorId'] = self.proj_id
        if super_sampling:
            self.cfg_proj['option'] = {'DetectorSuperSampling': super_sampling}

        # ---- Configure Backprojector ------
        # volume shape
        self.vshape = astra.functions.geom_size(self.vg)
        # Configure backprojector
        if cudafbp:
            self.cfg_backproj = astra.creators.astra_dict('FBP_CUDA')
            self.cfg_backproj['FilterType'] = 'Ram-Lak'
        else:
            self.cfg_backproj = astra.creators.astra_dict('BP_CUDA')
        self.cfg_backproj['ProjectorId'] = self.proj_id
        if super_sampling:
            self.cfg_backproj['option'] = {'PixelSuperSampling': super_sampling}
        # -------------------
        self.n_x = n_x
        self.n_y = n_y
        self.dwidth = dwidth
        self.n_a = angles.shape[0]
        self.rot_center = rot_center if rot_center else dwidth // 2
        self.angles = angles
        self.cudafbp = cudafbp
        if not (cudafbp):
            _ramp = self.compute_ramp_filter(dwidth * 2) * 2.0
            self.rampfilter = np.abs(np.fft.fft(_ramp))
        else:
            self.rampfilter = None

    def __checkArray(self, arr):
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        if arr.flags['C_CONTIGUOUS'] == False:
            arr = np.ascontiguousarray(arr)
        return arr

    @staticmethod
    def compute_ramp_filter(L):
        """
        Compute a discretized ramp filter for FBP in the direct domain.

        The filters provided by ASTRA do not seem to take into account that
        the ramp filter is band limited in practice. In particular,
        the zero frequency (mean value) should not be set to zero ;
        otherwise a bias appears in the reconstructed image.

        See H. Murrel, "Computer-Aided Tomography" in "The Mathematica Journal", 1996, vol. 6, issue 2, pp 60-65
        """
        h = np.zeros(L)
        L2 = L // 2 + 1
        h[0] = 1 / 4.
        j = np.linspace(1, L2, L2 // 2, False)
        # h[2::2] = 0
        h[1:L2:2] = -1. / (pi ** 2 * j ** 2)
        # h[-1:L2-1:-2] = -1./(pi**2 * j**2)
        h[L2:] = np.copy(h[1:L2 - 1][::-1])
        return h

    def backproj(self, s, filt=False, ext=False, method=1):

        if ext:
            s = self.extend_projections(s, method)

        if not (self.cudafbp):
            if filt is True:
                convmode = "linear" if not (ext) else "circular"
                s = self.filter_projections(s, convmode=convmode)
        elif filt == False:  # TODO: create a new backprojector
            print("Warning: in this tomo setting, cudafbp=True. This means that the data *will* be filtered !")

        sino = self.__checkArray(s)
        # In
        sid = astra.data2d.link('-sino', self.pg, sino)
        self.cfg_backproj['ProjectionDataId'] = sid
        # Out
        v = np.zeros(self.vshape, dtype=np.float32)
        vid = astra.data2d.link('-vol', self.vg, v)
        self.cfg_backproj['ReconstructionDataId'] = vid

        bp_id = astra.algorithm.create(self.cfg_backproj)
        astra.algorithm.run(bp_id)
        astra.algorithm.delete(bp_id)
        astra.data2d.delete([sid, vid])
        return v

    def proj(self, v):
        v = self.__checkArray(v)
        # In
        vid = astra.data2d.link('-vol', self.vg, v)
        self.cfg_proj['VolumeDataId'] = vid
        # Out
        s = np.zeros(self.sshape, dtype=np.float32)
        sid = astra.data2d.link('-sino', self.pg, s)
        self.cfg_proj['ProjectionDataId'] = sid

        fp_id = astra.algorithm.create(self.cfg_proj)
        astra.algorithm.run(fp_id)
        astra.algorithm.delete(fp_id)
        astra.data2d.delete([vid, sid])
        return s

    def filter_projections(self, sino, convmode="linear"):
        nb_angles, l_x = sino.shape
        if convmode == "linear":
            # ~ ramp = 1./l_x * np.hstack((np.arange(l_x), np.arange(l_x, 0, -1))) # freq. zero is set to zero => bias in reconstructed slice
            ramp = self.rampfilter
            return np.fft.ifft(ramp * np.fft.fft(sino, 2 * l_x, axis=1), axis=1)[:, :l_x].real
        elif convmode == "circular":  # for padded sinogram
            N = l_x
            n_px = self.n_x
            ramp = 1. / (N // 2) * np.hstack((np.arange(N // 2), np.arange(N // 2, 0, -1)))
            sino_extended_f = np.fft.fft(sino, N, axis=1)
            return np.fft.ifft(ramp * sino_extended_f, axis=1)[:, :n_px].real
        else:
            raise ValueError("Unknown convolution mode")

    def extend_projections(self, sino, method=1):
        """
        Extrapolate the sinogram of n/2 at each side
        This is often used in local tomography where the sinogram is "cropped",
        to avoid truncation artifacts.

        sino: numpy.ndarray
            sinogram, its shape must be consistend with the current tomography configuration
        method: integer
            if 1, extend the sinogram with its boundaries
            if 0, extend the sinogram with zeros
        """
        n_angles, n_px = sino.shape
        if (n_px <= 2048):
            N = nextpow2(2 * n_px)
        else:
            N = 2 * n_px  # memory !

        sino_extended = np.zeros((n_angles, N))
        sino_extended[:, :n_px] = sino
        boundary_right = (sino[:, -1])[:, np.newaxis]
        boundary_left = (sino[:, 0])[:, np.newaxis]
        # TODO : more methods
        method = int(bool(method))
        boundary_left *= method
        boundary_right *= method

        sino_extended[:, n_px:n_px + n_px // 2] = np.tile(boundary_right, (1, n_px // 2))
        sino_extended[:, -n_px // 2:] = np.tile(boundary_left, (1, n_px // 2))
        return sino_extended

    def run_algorithm(self, alg, n_it, data):
        rec_id = astra.data2d.create('-vol', self.vol_geom)
        sino_id = astra.data2d.create('-sino', self.proj_geom, data)
        cfg = astra.astra_dict(alg)
        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId'] = sino_id
        alg_id = astra.algorithm.create(cfg)
        print("Running %s" % alg)
        astra.algorithm.run(alg_id, n_it)
        rec = astra.data2d.get(rec_id)
        astra.algorithm.delete(alg_id)
        astra.data2d.delete(rec_id)
        astra.data2d.delete(sino_id)
        return rec

    def fbp(self, sino, padding=None):
        """
        Runs the Filtered Back-Projection algorithm on the provided sinogram.

        sino : numpy.ndarray
            sinogram. Its shape must be consistent with the current tomography configuration
        padding: integer
            Disabled (None) by default.
            If 0, the sinogram is extended with zeros.
            If 1, the sinogram is extended with its boundaries.
        """
        if padding is None:
            return self.backproj(sino, filt=True)
        else:
            return self.backproj(sino, filt=True, ext=True, method=padding)

    def cleanup(self):
        # ~ astra.data2d.delete(self.rec_id)
        astra.data2d.delete(self.proj_id)


def generate_coords(img_shp, center=None):
    l_r, l_c = float(img_shp[0]), float(img_shp[1])
    R, C = np.mgrid[:l_r, :l_c]
    if center is None:
        center0, center1 = l_r / 2., l_c / 2.
    else:
        center0, center1 = center
    R += 0.5 - center0
    C += 0.5 - center1
    return R, C


def clipCircle(img, radius=None):
    """
    same as astra.extrautils.clipCircle() for square images
    """
    R, C = generate_coords(img.shape)
    res = np.copy(img)
    if radius is None:
        radius = max(img.shape)
    res[R ** 2 + C ** 2 > ((radius - 0) / 2.) ** 2] = 0;
    return res


def clipCircle0(img, radius=None):
    res = np.copy(img)
    astra.extrautils.clipCircle(res)
    return res
