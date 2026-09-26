import numpy as np

# FLOP convention: 1 MAC = 2 FLOPs. Conv/linear MACs + ReLU 1/el + pool 8 compares/window + GAP.


def flops(image_size, batch):
    return 17751 * batch * image_size**2 + 313600 * batch


def memory(image_size, batch):
    # Weights + peak live bytes at MaxPool: x (12) + conv1 (32) + pool (8)
    # + int64 indices (16) = 68 B S^2. cuDNN workspace is not in this form.
    return 4_161_296 + 68 * batch * image_size**2


def bytes_moved(image_size, batch):
    # Logical traffic, not DRAM: layer in+out, MaxPool index write, inplace ReLU.
    return 380 * batch * image_size**2 + 8592 * batch + 4_161_296


def latency(image_size, batch, theta):
    f = flops(image_size, batch)
    q = bytes_moved(image_size, batch)
    return theta["launch_overhead"] + np.maximum(
        f / theta["peak_flops"], q / theta["bandwidth"]
    )


def energy(image_size, batch, theta_energy):
    # Dynamic term is per byte moved, not per FLOP: a joint fit of both
    # puts the whole coefficient on Q and sets the FLOP term to zero.
    q = bytes_moved(image_size, batch)
    t = latency(image_size, batch, theta_energy)
    return theta_energy["p_static"] * t + theta_energy["e_byte"] * q
