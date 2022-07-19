"""
Compresses the fr from matlab generated using uncompressed saving features/converts to numpy saved arrays.
Matlab saving with compress on large files is ~15x slower than uncompressed for about 6x compression, so instead
we will use python's compression algorithms in post processing.
"""
import numpy as np
import h5py
import os
import time


def process_mat_file(filepath, output_path=None, SC_compression_factor=1, time_compression_factor=1, compression_opts=9):
    """
    Load and process one matfile and manually compress if desired in the specific domain.
    :param filepath:
    :param output_path:
    :param SC_compression_factor: Integer values for how many subcarriers to average together
    :param time_compression_factor: Integer value for how many timeslots to average together
    :param compression_opts: passed to h5py gzip compression. 0-9 with default of 9 being highest lossless compression
    :return: filepath for the compressed file, usually the original with _comp unless specified
    """
    SC_compression_factor = int(SC_compression_factor)
    time_compression_factor = int(time_compression_factor)
    if output_path is None:
        output_path = filepath[:-4] + '_comp.hdf5'
    with h5py.File(filepath, 'r') as hf:
        a = np.array(hf['fr'])
        if a.dtype == np.complex64:
            H = a
        else:
            a = np.array(hf['fr']['real'])
            a2 = np.array(hf['fr']['imag'])
            H = a + 1j*a2
    units = H.shape  # [time, frequency, Nt, Nr]
    if time_compression_factor > 1:
        new_t_count = int(np.ceil(units[0] / time_compression_factor))
        Hbar = np.zeros((new_t_count, units[1], units[2], units[3]), dtype=np.complex64)
        H_sub = np.array_split(H, new_t_count, axis=0)
        for i in range(new_t_count):
            Hbar[i, :, :, :] = np.mean(H_sub[i], axis=0)
        del H_sub
        H = Hbar.copy()
        units = H.shape
        del Hbar
    if SC_compression_factor > 1:
        new_sc_count = int(np.ceil(units[1] / SC_compression_factor))
        Hbar = np.zeros((units[0], new_sc_count, units[2], units[3]), dtype=np.complex64)
        H_sub = np.array_split(H, new_sc_count, axis=1)
        for i in range(new_sc_count):
            Hbar[:, i, :, :] = np.mean(H_sub[i], axis=1)
        del H_sub
        H = Hbar.copy()
        units = H.shape
        del Hbar

    with h5py.File(output_path, "w") as f:
        dset = f.create_dataset("fr", data=H, dtype='complex64', compression="gzip", compression_opts=compression_opts)
    old_size = os.path.getsize(filepath) / 1024.0 / 1024 / 1024  # GB
    new_size = os.path.getsize(output_path) / 1024.0 / 1024 / 1024
    print(H.shape)
    print(f"Previous file size was {old_size:.3f} GB and new file size is {new_size:.3f} GB")
    return output_path


tstart = time.time()
test_folder = r'C:/Users/rmdreifu/Documents/GitHub/SSB-Encoder/Matlab_funcs/data/sub6 M-MIMO 0/channels/'
output_folder = r'C:/Users/rmdreifu/Documents/GitHub/SSB-Encoder/data/sub6 M-MIMO 0/'
file_set = os.listdir(test_folder)
for filep in file_set:
    _ = process_mat_file(test_folder+filep, output_folder+filep[:-3]+'hdf5', time_compression_factor=4, compression_opts=4)

tend = time.time()
print(f"Compression took {tend-tstart} sec")
