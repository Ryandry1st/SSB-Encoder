import numpy as np
import numba
import os, time, pickle, h5py
import matplotlib.pyplot as plt
from functools import partial
from math import sqrt
from Phy import *
from utils import *

# constants
pi = np.pi
dtype = np.complex64

# reproducibility
seed = 27  # arbitrary seed for reproducibility
np.random.seed(seed)
# tfpi = tf.constant(pi, dtype=tf.complex64)
# tf.random.set_seed(seed)

# 5G Specification notes
# Always 12 subcarriers and 14 time symbols per slot. Slot time can vary for transmitting codebook (SSB)

# Example
users = 5
Nx = 8
Ny = 8
Nt = Nx * Ny
Nr = 4
resource_blocks = 128
timeslots = 100
codebook_size = 64

unit_defs = [users, timeslots, resource_blocks, Nr, Nt]

Hset = 1/sqrt(2) * (np.random.randn(*unit_defs) + 1j * np.random.randn(*unit_defs)).astype(np.complex64)

codebook = dft_codebook(codebook_size, Nx, Ny)
# reshape Nx by Ny to Nt
codebook = codebook.reshape(codebook_size, Nt, 1)
# obtain the RSRP
rsrps, indices = rsrp_report(Hset, codebook)
rsrps_db = conv_db(rsrps)
print(f"RSRPs = {rsrps} dB")
print(f"Best beam indices = {indices}")

# obtain the SU-MIMO spectral efficiency at time 0 and time 50
su_SE = np.zeros((users, 2))
t_start = time.time()
for ue in range(users):
    su_SE[ue, 0] = su_mimo(Hset[ue, 0:1], subbands=1, subtimes=1, gain=10)
    su_SE[ue, 1] = su_mimo(Hset[ue, 50:51], subbands=1, subtimes=1, gain=10)
print(f"SU spectral effiencies = \n{su_SE}")
t_end = time.time()
print(f"Time required = {t_end-t_start:.3f}s")

# obtain the MU-MIMO spectral efficiency using a DFT codebook, single-stream, all users
# BD is almost always best when some form of user selection is employed
bd_precoders, assignments = blockdiag_mu_mimo(Hset)
print(f"precoder shape = {bd_precoders.shape}")
capacities = mu_mimo_overall(Hset, bd_precoders, assignments=assignments, timesteps=5, gain=10)
print(capacities)

