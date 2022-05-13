import numpy as np
from math import sqrt
from utils import nb_mean_0, nb_det, nb_svd_s, nb_svd
import warnings

# constants
pi = np.pi
dtype = np.complex64
# limitations to prevent endfire, beams should not be aimed beyond these directions
theta_min = 5*pi/24
theta_max = 19*pi/24
phi_min = 5*pi/24
phi_max = 19*pi/24
# azimuth varies from 5/24 = along -x axis, pi/2 along +y axis, 19/24 along +x
# elevation varies from 5/24 = down, pi/2 = horizontal, 19/24 = up

try:
    import numba
    mode = 'numba'
except Exception:
    print("Numba is not available, performance will be slower")
    mode = None


def join_nt(channel):
    """
    reshapes a channel from Nx, Ny to Nt shaping
    :param channel: should be of size [???, Nx, Ny] where any dimension works so long as the last two are Nx, Ny
    :return: channels of shape [???, Nt]
    """
    return channel.reshape((*channel.shape[:-2], -1))


def split_nt(channel, Nx, Ny):
    """
    Opposite of join_nt, splits an Nt to Nx, Ny
    :param channel: should be of size [???, Nt] where any dimension works so long as the last is Nt
    :return: channels of shape [???, Nx, Ny]
    """
    return channel.reshape(*channel.shape[:-1], Nx, Ny)


def observation_from_beams(RSRPs_db, indices, codebook):
    # Creates a 2D representation of the codebook from the RSRP in dB and the indices of an rsrp_report
    units = codebook.shape
    codebook = codebook.transpose(0, 2, 1).reshape(units[0] * units[2], units[1], 1)  # shape [L*Nr, Nt, 1]
    codebook = codebook.reshape(-1, int(np.sqrt(units[-2])), int(np.sqrt(units[-2])))  # [L*Nr, Nx, Ny]

    active_regions = active_beamspace(codebook)
    active_regions = active_regions / np.linalg.norm(active_regions, axis=(1, 2), keepdims=True)
    # scale by the rsrp in dB
    max_rsrp = np.max(RSRPs_db)
    min_rsrp = np.min(RSRPs_db)
    rsrp_std = (RSRPs_db - min_rsrp) / (max_rsrp - min_rsrp)
    rsrp_scaling = rsrp_std * (max_rsrp - min_rsrp) + min_rsrp
    overall_scalings = 1e-7*np.ones((units[0]*units[2], 1, 1))
    for i, ind in enumerate(indices):
        overall_scalings[ind, 0, 0] += rsrp_scaling[i]
    active_regions *= overall_scalings


# %%%%%%%%%%%%%%%%%%%%%%%%%%% Numba Func Definitions %%%%%%%%%%%%%%%%%%%%%%%%%%
if mode == 'numba':
    @numba.jit(fastmath=True, nopython=True)
    def array_resp(theta, N, lambda_c=1, d=None):
        """
        Returns the array response for a ULA in direction theta with N antenna elements. Defaults to lambda/2 spacing
        :param theta: Direction of interest for array response
        :param N: Number of antenna elements
        :param lambda_c: wavelength, although this only needs to be relative to d so it is often not necessary.
        :param d: antenna element spacing. Defaults to lamba_c/2.
        :return: the n-point array response of a ULA
        """
        if d is None or d is False:
            d = lambda_c / 2
        n = np.arange(N)
        vander = np.expand_dims(np.exp((1j * pi * 2 * n * d * np.cos(theta)) / lambda_c), 0)
        return vander
    
    
    @numba.jit(fastmath=True, nopython=True)
    def pattern_ijk(theta, phi, antenna, V):
        x = np.abs(array_resp(theta, antenna).conj() @ V @  array_resp(phi, antenna).conj().T)**2
        return x[0, 0]


    @numba.jit(nopython=True, parallel=True, fastmath=True)
    def active_beamspace(beams):
        # beams should be shape [L, Nx, Ny]
        units = beams.shape
        power_meas = np.zeros(units, dtype=np.float32)
        thetas = np.linspace(theta_min, theta_max, units[1])
        phis = np.linspace(phi_min, phi_max, units[2])
        for k, beam in enumerate(beams):
            for i, theta in enumerate(thetas):
                for j, phi in enumerate(phis):
                    power_meas[k, i, j] = pattern_ijk(theta, phi, units[1], beam)
        return power_meas

    @numba.jit(nopython=True)
    def mat_mul(norm_Hi, Fj, ue_factor):
        return 1/ue_factor * norm_Hi.T @ Fj @ Fj.conj().T @ norm_Hi.conj()


    @numba.jit(nopython=True)
    def inner_mu_mimo_computation(norm_H, F, snr_factor, UEs=8, Nt=64, Nr=4):
        ue_factor = np.array(UEs, dtype=dtype)
        sinrs = np.zeros((UEs, Nr))
        for i in numba.prange(UEs):
            precoder_internal = np.eye(Nr, dtype=dtype) * UEs * Nt / snr_factor[i]
            for j in numba.prange(UEs):
                precoder_internal = precoder_internal + 1/ue_factor * norm_H[i, :, :] @ F[j, :, :] @ F[j, :, :].conj().T @ norm_H[i, :, :].conj().T
            precoder_internal = norm_H[i, :, :].conj().T @ np.linalg.pinv(precoder_internal) @ norm_H[i, :, :]
            for j in numba.prange(Nr):
                sinrs[i, j] = np.real(F[i, :, j].conj() @  precoder_internal @ F[i, :, j].T)
        return sinrs


# %%%%%%%%%%%%%%%%%%%%%%%%%%% non-Numba Func Definitions %%%%%%%%%%%%%%%%%%%%%%%%%%
else:
    def array_resp(theta, N, lambda_c=1, d=None):
        """
        Returns the array response for a ULA in direction theta with N antenna elements. Defaults to lambda/2 spacing
        :param theta: Direction of interest for array response
        :param N: Number of antenna elements
        :param lambda_c: wavelength, although this only needs to be relative to d so it is often not necessary.
        :param d: antenna element spacing. Defaults to lamba_c/2.
        :return: the n-point array response of a ULA
        """
        if d is None or d is False:
            d = lambda_c / 2
        n = np.arange(N)
        vander = np.expand_dims(np.exp((1j * pi * 2 * n * d * np.cos(theta)) / lambda_c), 0)
        return vander


    def active_beamspace(beams):
        # beams should be shape [L, Nx, Ny]
        units = beams.shape
        power_meas = np.zeros(units, dtype=np.float32)
        thetas = np.linspace(theta_min, theta_max, units[1])
        phis = np.linspace(phi_min, phi_max, units[2])
        for k, beam in enumerate(beams):
            for i, theta in enumerate(thetas):
                for j, phi in enumerate(phis):
                    power_meas[k, i, j] = pattern_ijk(theta, phi, units[1], beam)
        return power_meas


    def inner_mu_mimo_computation(norm_H, F, snr_factor, UEs=8, Nt=64, Nr=4):
        ue_factor = np.array(UEs, dtype=dtype)
        sinrs = np.zeros((UEs, Nr))
        for i in range(UEs):
            precoder_internal = np.eye(Nr, dtype=dtype) * UEs * Nt / snr_factor[i]
            for j in range(UEs):
                precoder_internal = precoder_internal + 1/ue_factor * norm_H[i, :, :] @ F[j, :, :] @ F[j, :, :].conj().T @ norm_H[i, :, :].conj().T
            precoder_internal = norm_H[i, :, :].conj().T @ np.linalg.pinv(precoder_internal) @ norm_H[i, :, :]
            for j in range(Nr):
                sinrs[i, j] = np.real(F[i, :, j].conj() @  precoder_internal @ F[i, :, j].T)
        return sinrs


# %%%%%%%%%%%%%%%%%%%%%%%%%%% Codebook algorithms %%%%%%%%%%%%%%%%%%%%%%%%%%
def dft_codebook(n_beams, Nx, Ny, int_y_beams=None, int_x_beams=None, verbose=0):
    """
    Generates a dft codebook with a certain number of elements for a UPA with Nx by Ny antenna elements. Optional
    numbers of beams in each direction can be specified to fit a circumstance, i.e. a base station does not have as many
    elevation beams as azimuth beams. NOTE DFT codebooks are only single-stream. For multi-stream, one should combine
    various beams together and renormalize so that the frobenius norm is sqrt(Nx*Ny), after stacking along all TX antenna
    :param n_beams: Total number of beams to generate in the codebook
    :param Nx: Number of antenna elements in the x direction
    :param Ny: Number of antenna elements in the y direction
    :param int_y_beams: Number of beams in the elevation direction. Should be multiplied with int_x_beams to produce n_beams
    :param int_x_beams: Number of beams in the azimuth direction. Should be multiplied with int_y_beams to produce n_beams
    :param verbose: If verbose != 0 or False, print out various check steps
    :return: a codebook of size [n_beams, Nx, Ny]
    """
    tot_beams = n_beams
    int_rt_beams = int(np.floor(sqrt(tot_beams)))
    if int_y_beams is None and int_x_beams is None:
        if not np.isclose(sqrt(tot_beams), int_rt_beams):  # beams are not evenly split, must throwaway some.
            # not evenly split, throw away top-few
            int_rt_beams = int(np.ceil(sqrt(tot_beams)))

    if int_y_beams is None:
        int_y_beams = int_rt_beams
    if int_x_beams is None:
        int_x_beams = int_rt_beams
    n_throwaway = int_y_beams * int_x_beams - n_beams

    thetas = np.linspace(theta_min, theta_max, num=2*int_x_beams+1)
    phis = np.linspace(phi_min, phi_max, num=2*int_y_beams+1)
    # obtain the center of angles not edges
    thetas = thetas[1::2]
    phis = phis[1::2]
    if verbose:
        print(f"Thetas should be from {(pi/2 - 7*pi/24):.2f}-{(pi/2 + 7*pi/24):.2f}, Phi {5/24*pi:.2f}-{pi/2:.2f}")
        print(thetas, phis)
    # variable for the various beams
    codebook = np.zeros((int_x_beams, int_y_beams, Nx, Ny), dtype=dtype)
    center_x = (Nx-int_x_beams)//2
    center_y = (Ny-int_y_beams)//2

    # Beams that are wider than the array response, i.e. if only 2 beams in elevation but 8 antennas, then some antenna
    # are not used to ensure the codebook covers the entire region. This may need to be changed based on situations
    # but one can always start with a larger codebook and remove elements to not get this effect.
    for i in range(int_x_beams):
        for j in range(int_y_beams):
            tmp = array_resp(thetas[i], int_x_beams).T @ array_resp(phis[j], int_y_beams) / sqrt(int_x_beams*int_y_beams)
            codebook[i, j, center_x:center_x+int_x_beams, center_y:center_y+int_y_beams] = tmp

    codebook = codebook.reshape((int_x_beams*int_y_beams, Nx, Ny))
    if n_throwaway:
        codebook = codebook[n_throwaway:]  # arbitrarily throw away the first beams, which are the upper-left corner

    return codebook


def rsrp_report(Hset, codebook, time_index=0, freq_index=0, gain=1):
    """
    Obtains the RSRP report for a set of receivers using a set of beams
    !! Assumes times are in 1ms portions and frequencies are in resource blocks of 12 subcarriers with at least 20 blocks !!
    :param Hset: set of channels of shape [U, T, SC, Nr, Nt]
    :param codebook: should be shape [L, Nt, Nreff] where L is numnber of beams and Nreff is the number of streams.
    If multiple streams are used, they are treated as separate beams but with shared power and timing.
    :param time_index: The starting time index in the set of channels Hset to use for the first ssb
    :param freq_index: The starting frequency index in the set of channels Hset to use for each ssb
    :param gain: A gain factor to account for various transmit/receive power and other gains
    :return: the RSRP and beam indexes corresponding to the strongest beam for each receiver
    """
    # time_index can be scalar or vector of size len(beams)
    # first, assume perfect channel estimation
    units = np.array(Hset.shape).astype(np.int16)
    noise_units = units[np.array([0, 1, 3])]
    Nreff = codebook.shape[-1]

    if len(codebook) == 8:
        starting_symbols = np.array([4, 8, 16, 20, 32, 36, 44, 48])  # specific for 5G, 30KHz scs, 3-6GHz fc
    else:
        starting_symbols = np.arange(1, len(codebook) + 1) * 4  # arbitrary symbols, not quite in spec but close enough
    starting_slots = np.floor(starting_symbols/14).astype(np.int16)

    all_rsrps = np.zeros((units[0], len(codebook), Nreff))  # len(beams) could be less than Lmax
    codebook = codebook / np.linalg.norm(codebook, axis=(1, 2), keepdims=True)  # ensure normalized
    noise = 1/np.sqrt(2) * (np.random.randn(*noise_units) + 1j * np.random.randn(*noise_units))
#     noise=np.zeros_like(noise)  # remove noise if testing and desired

    Hset = np.transpose(Hset, [0, 1, 2, 4, 3])  # reorder
    time_slots = time_index + starting_slots  # adjust for the current time and next 2ms (ish)
    for ue in range(units[0]):
        channels = Hset[ue, time_slots, freq_index:freq_index+20, :, :]  # assumes the first 240sc used for SSB
        noises = noise[ue, time_slots, :]
        # channels are of shape [L, 240, Nt, Nr], beams are of shape [L, Nt, Nreff]
        for ssb in range(len(codebook)):
            chan = channels[ssb, :, :, :]
            c = chan.transpose(0, 2, 1).conj() @ codebook[ssb]
            cp = np.sqrt(gain) * np.mean(c, axis=0) + np.expand_dims(noises[ssb, :], 1)  # average over the subcarriers, add noise
            all_rsrps[ue, ssb, :] = abs(np.linalg.norm(cp, axis=0))**2  # MRC to achieve the norm sq

    all_rsrps = all_rsrps.reshape(units[0], len(codebook) * Nreff)  # treat spatial multiplexing as different beam
    # select strongest beam and beam index
    rsrps = np.max(all_rsrps, axis=1)
    indices = np.argmax(all_rsrps, axis=1).astype(np.int16)
    return rsrps, indices.astype(np.int16)


def basic_beam_selector(codebook, indices, rsrps):
    """
    provides a beam_assignments setup for computing MU-MIMO results using the strongest rsrp assignments. This determines
    what beam is assigned to what user for use in MU-MIMO code. Some users may request the same precoder, so the strongest
    receiver is allocated that beam, and others are not allocated a codeword. Additional logic could be used to improve this.
    :param codebook: should be shape [L, Nt, Nreff] where L is numnber of beams and Nreff is the number of streams.
    If multiple streams are used, they are treated as separate beams to be allocated.
    :param indices: output of the rsrp_report defining the strongest beam indices
    :param rsrps: output of the rsrp_report defining the rsrp received
    :return: beam assignments
    """
    beam_assignments = -1*np.ones((len(codebook) * codebook.shape[-1],))
    unique_beams, inds, invs, counts = np.unique(indices, return_index=True, return_inverse=True, return_counts=True)
    unique_beams = unique_beams.astype(np.int16)
    for i, ind in enumerate(unique_beams):
        if counts[i] > 1:  # repeated value, select strongest RSRP option
            rsrp_comps = rsrps[indices == ind]
            max_rsrp = np.max(rsrp_comps)
            beam_assignments[ind] = np.argmax(rsrps == max_rsrp)
        else:
            # directly add
            beam_assignments[ind] = inds[i]
    return beam_assignments


# %%%%%%%%%%%%%%%%%%%%%%%%%%% Beamforming algorithms %%%%%%%%%%%%%%%%%%%%%%%%%%
def su_mimo(H, subtimes=1, subbands=1, gain=1, return_precoder=False):
    """
    Perform SU-MIMO for a single user's set of channels using the SVD. Waterfilling is not used, so this is not optimal, but it is
    more reasonable and essentially optimal at high SNR. Currently setup for channels [t, sc, Nr, Nt] where t is a set
    of time samples and sc is a set of subcarriers. If these are not needed, simply pass a channel of shape [Nr, Nt]
    :param H: Should be the complex channel coefficients of size [T, SC, Nr, Nt] or [X, Nr, Nt] or [Nr, Nt]
    :param subtimes: Number of precoders to use in the time domain. Can be up to size T. Very computationally expensive
    :param subbands: Number of precoders to use in the frequency domain. Can be up to size SC.
    :param gain: Adds a gain factor to account for transmit power, receive power, various gains or losses, etc.
    :param return_precoder: select whether to return the associated precoders or not
    :return: If the precoder is not needed, just the capacity is returned. If the precoders are needed, then those are returned
    as well, at the cost of more computation time.
    """
    if len(H.shape) == 2:
        H = np.expand_dims(np.expand_dims(H, 0), 0)
    elif len(H.shape) == 3:
        H = np.expand_dims(H, 0)
    elif len(H.shape) != 4:
        raise AssertionError("Error, shape of channels is {H.shape}, but should be 2, 3, or 4 dimensional")
    units = np.array(H.shape).astype(np.int16)  # [time, sc, Nr, Nt]
    capacities = np.zeros((units[0], subbands))
    # must iteratively split the channel into subbands/subtimes and average to obtain best discrete precoder
    try:
        H_subbands = np.array_split(H, subbands, axis=1)
    except Exception:
        largest_ind = (H.shape[1]//subbands)*subbands
        H_subbands = np.array_split(H[:, :largest_ind], subbands, axis=1)
    H_bar = np.zeros((subbands, units[0], units[2], units[3]), dtype=dtype)
    for i in range(subbands):
        H_bar[i, :, :, :] = nb_mean_0(np.transpose(H_subbands[i][:, :, :, :], (1, 0, 2, 3)))
    try:
        H_subbands = np.array_split(H_bar, subtimes, axis=1)  # has shape [subtimes, subbands, Nt, Nr]
    except Exception:
        largest_ind = (H.shape[1]//subtimes)*subtimes
        H_subbands = np.array_split(H_bar[:, :largest_ind], subbands, axis=1)
    H_bar = np.zeros((subtimes, subbands, units[2], units[3]), dtype=dtype)
    for i in range(subtimes):
        H_bar[i, :, :, :] = nb_mean_0(np.transpose(H_subbands[i][:, :, :, :], (1, 0, 2, 3)))

    Fs = np.zeros((subtimes, subbands, units[3], units[3]), dtype=dtype)
    for sb in range(subbands):
        U, S, Fs[:, sb, :, :] = nb_svd(H_bar[:, sb, :, :])

    Fs = Fs[:, :, :, :units[2]]  # only the strongest Nr singular vectors are used
    # convert subtimes to know which time it is in
    for i in range(units[0]):
        subt = i//(units[0]//subtimes)
        for subb in range(subbands):
            inner_comp = H[i] @ Fs[subt, subb:subb+1] @ np.transpose(Fs[subt, subb:subb+1].conj(), (0, 2, 1)) @ np.transpose(H[i].conj(), (0, 2, 1)) * gain

            capacities[i, subb] = 1/units[1] * np.real(np.sum(np.log2(nb_det(np.expand_dims(np.eye(units[-2], dtype=dtype), axis=0) + inner_comp))))

    capacities = np.mean(capacities, axis=1)
    if return_precoder:
        return capacities, Fs
    else:
        return capacities


def zf_mu_mimo(Hset, active_RX=None, time=0, gain=1):
    """
    Obtains the zero-forcing precoders for a specific time and averaged over all subcarriers.
    :param Hset: set of channels of shape [U, T, SC, Nr, Nt]
    :param active_RX: an array containing the indices of the active users
    :param time: the time slot for the ZF channels to be based on
    :param gain: gain factor to account for transmit, receive power, etc.
    :return: zero forcing precoders and the associated beam assignments
    """
    H = np.mean(Hset[:, time, :, :, :], axis=1)  # [U, Nr, Nt]
    if active_RX is None:
        active_RX = np.arange(len(H))
    H = H[active_RX]
    snrs = np.linalg.norm(H, axis=(1, 2)) * gain
    units = H.shape
    huhu = len(H) * units[3] * np.eye(64, dtype=dtype)  # see Lozano Heath equation (
    F = np.zeros((len(active_RX), units[2], units[1]), dtype=dtype)
    for ue in active_RX:
        huhu = huhu + snrs[ue] * H[ue].conj().T @ H[ue]
    F[:, :, :] = np.linalg.pinv(huhu) @ H.conj().transpose(0, 2, 1)
    F = F / np.linalg.norm(F, axis=(1, 2), keepdims=True) * np.sqrt(units[2])
    F = F.transpose(0, 2, 1).reshape(units[0]*units[1], units[2], 1)
    beam_assignments = np.repeat(active_RX, units[1])
    return F, beam_assignments


def blockdiag_mu_mimo(Hset, active_RX=None, time=0):
    """
    Obtains the block diagonalization precoders for a specific time and averaged over all subcarriers. Note that block
    diagonalization is less restrictive than zero forcing, and so should perform better, but it is intended
    for a specific receive filter which is not commonly done. With LMMSE receiving and uniform power allocation
    the two precoders perform essentially equivalent in my experience.
    :param Hset:
    :param active_RX: an array containing the indices of the active users
    :param time: the time slot for the ZF channels to be based on
    :return: block diagonalization precoders and the associated beam assignments
    """
    # the assignments are always just the repeat of the original UEs order
    H = np.mean(Hset[:, time, :, :, :], axis=1)  # [U, Nr, Nt]
    if active_RX is None:
        active_RX = np.arange(len(H))
    H = H[active_RX]
    units = H.shape
    F = np.zeros((len(active_RX), units[2], units[1]), dtype=dtype)
    for u in range(units[0]):
        Cminus = np.delete(H.copy(), u, axis=0).reshape((units[0]-1)*units[1], units[2])
        U, S, V = np.linalg.svd(Cminus)
        Vnull = V[units[1]:, :].conj().T  # obtain the null space from the sorted SVD right singular vectors
        Hui = H[u] @ Vnull  # direct the remaining channels into the null of the intended user
        Ui, Si, Vi = np.linalg.svd(Hui)
        Vi = Vi[:units[1]]
        F[u, :, :] = Vnull @ Vi.T  # precoders are the null-directed right singular vectors
    F = F / np.linalg.norm(F, axis=(1, 2), keepdims=True) * np.sqrt(units[2])  # normalize
    F = F.transpose(0, 2, 1).reshape(units[0]*units[1], units[2], 1)
    beam_assignments = np.repeat(active_RX, units[1])
    return F, beam_assignments


def mu_mimo_per_resource(Hset, precoders, assignments, snr_eff_ratio=1, gain=1, nb=True):
    """
    Obtain the MU-MIMO spectral efficiency for a given set of precoders and assignments. It is assumed that the receiver
    is an LMMSE receiver that has an effective SNR ratio that is <= 1 with 1 being optimal.
    :param Hset: set of channels for a single time-frequency resource block of shape [U, Nr, Nt]
    :param precoders: A set of precoders that should have size [?, Nt] with a maximum of ? = U*Nr, but could be much smaller
    :param assignments: has shape [?] same as precoder length that has values of -1 for inactive precoders and {0, 1, ..U}
    for all active precoders
    :param snr_eff_ratio: Value describing how effective the channel estimation is at the receiver, with 1 being optimal
    :param gain: Account for various gain factors possible.
    :param nb: Whether or not to use numba. Should be used if possible as it is much faster, but uses most of the CPU processing
    capabilities.
    :return: The sum spectral efficiency of the MU-MIMO system
    """
    # H is of shape [UEs, Nr, Nt] because its for one time and freq unit
    # beam assignments should be of shape [len(beams)] and has values 0,... len(H)=U
    # beams is of shape [Nbeam, Nt]
    units = Hset.shape
    sinrs = np.zeros((units[0], units[2]))
    H_norm_factor = np.linalg.norm(Hset, axis=(1, 2), keepdims=True)
    norm_H = (Hset / H_norm_factor).astype(dtype)
    snr_factor = gain * snr_eff_ratio * H_norm_factor**2  # gather's channel gain and snr_eff.

    F = np.zeros(units, dtype=dtype).transpose(0, 2, 1)
    total_beams_a = np.zeros((units[0],), dtype=np.int16)  # assigned number of beams
    for i, beam_i in enumerate(assignments):
        if beam_i >= 0:
            F[beam_i, :, total_beams_a[beam_i]] = precoders[i, :, 0]
            total_beams_a[beam_i] += 1
    for ue in range(units[0]):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            F[ue, :, :] = F[ue, :, :] / np.linalg.norm(F[ue, :, :total_beams_a[ue]], ord='fro', keepdims=True) * np.sqrt(units[-2])

    F[np.isinf(F)] = 0
    F[np.isnan(F)] = 0
    active_ues = np.sum(total_beams_a > 0)
    if nb:
        sinrs = inner_mu_mimo_computation(norm_H.astype(dtype), F.astype(dtype), snr_factor.astype(dtype), len(norm_H), Nt=units[-1], Nr=units[-2])
    else:
        for i in range(units[0]):
            precoder_internal = np.eye(units[-1], dtype=dtype) * active_ues * units[-2] / snr_factor[i]
            for j in range(units[0]):
                precoder_internal += 1/units[0] * norm_H[i, :, :].T @ F[j, :, :] @ F[j, :, :].conj().T @ norm_H[i, :, :].conj()
            precoder_internal = norm_H[i, :, :].conj() @ np.linalg.pinv(precoder_internal) @ norm_H[i, :, :].T
    #         print(precoder_internal)
            for j in range(units[2]):
                sinrs[i, j] = np.real(F[i, :, j].conj() @  precoder_internal @ F[i, :, j].T)

    return np.sum(np.log2(1+sinrs), axis=1)


def mu_mimo_overall(Hset, precoders, rsrps=None, indices=None, assignments=None, time=0, timesteps=1, snr_eff_ratio=1, gain=1, nb=True):
    """
    Calculate the MU-MIMO spectral efficiency over a set of times using the same precoders and assignments.
    :param Hset: set of channels for a set of resource block of shape [U, T, SC, Nr, Nt]
    :param precoders: A set of precoders that should have size [?, Nt] with a maximum of ? = U*Nr, but could be much smaller
    :param rsrps: rsrps in dB if beam assignments is not provided
    :param indices: indices from rsrps if beam assignments is not provided
    :param assignments: must be provided if rsrps and indices is not. Defines which precoder goes to which UE
    :param time: Starting time in the timesteps
    :param timesteps: Total number of timesteps to calculate for
    :param snr_eff_ratio: The effective SNR ratio assuming the receiver has to estimate the channel. Should be <= 1
    :param gain: A gain factor to account for various scaling of transmit and receive gains
    :param nb: Leave active if numba is available. Improves performance significantly at the cost of more CPU usage
    :return: the MU-MIMO spectral efficiencies over the set of timesteps
    """
    H = Hset[:, time:time+timesteps, :, :, :]
    units = H.shape
    if rsrps is not None and indices is not None and assignments is None:
        assignments = basic_beam_selector(precoders, indices, rsrps)
        assignments = assignments.astype(np.int16)
#     print(beam_selects)
    caps = np.zeros((units[0], units[1], units[2]))
    precoders = precoders / np.linalg.norm(precoders, axis=(-2, -1), keepdims=True) * np.sqrt(units[-2])

    for t in range(units[1]):
        for sc in range(units[2]):
            caps[:, t, sc] = mu_mimo_per_resource(H[:, t, sc, :, :], precoders, assignments, snr_eff_ratio, gain=gain, nb=nb)
    cap_per_ue = np.mean(caps, axis=2)
    return np.sum(cap_per_ue, axis=0)
