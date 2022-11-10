def reconstract(proj, theta, center, algorithm="gridrec", sinogram_order=False):
    if algorithm in ["fbp", ]:
        recon_gridrec_cpu = tomopy.recon(proj, theta, center=center, algorithm=algorithm, sinogram_order=sinogram_order)
    recon_gridrec_cpu = tomopy.circ_mask(recon_gridrec_cpu, axis=0, ratio=0.9)
    return recon_gridrec_cpu

def my_tomopy_recon(proj, theta, center, sinogram_order=False):
    options = {'proj_type': 'linear', 'method': 'FBP'}
    recon_gridrec_cpu = tomopy.recon(
        proj, theta, center=center, algorithm=tomopy.astra,
        sinogram_order=sinogram_order, options=options,
    )
    recon_gridrec_cpu = tomopy.circ_mask(recon_gridrec_cpu, axis=0, ratio=0.9)
    return recon_gridrec_cpu

def my_tomopy_recon(proj, theta, center, sinogram_order=False, circ_mask_ratio=1.0):
    options = {'proj_type': 'linear', 'method': 'FBP'}
    recon_gridrec_cpu = tomopy.recon(
        proj, theta, center=center, algorithm=tomopy.astra,
        sinogram_order=sinogram_order, options=options,
    )
    recon_gridrec_cpu = tomopy.circ_mask(recon_gridrec_cpu, axis=0, ratio=circ_mask_ratio)
    return recon_gridrec_cpu




