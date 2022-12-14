# -*- oding:utf-8 -*-
'''
# @File: Gen_Data.py
# @Author: Mengyuan Ma
# @Contact: mamengyuan410@gmail.com
# @Time: 2022-12-14 11:19 AM
'''
from Global_Vars import *
import numpy as np
import h5py

def pulase_filter(t, Ts, beta):
    '''
    Raised cosine filter
    :param t: time slot
    :param Ts: sampling frequency
    :param beta: roll-off factor
    :return: filtered value
    '''
    if abs(t-Ts/2/beta)/abs(t) <1e-4 or abs(t+Ts/2/beta)/abs(t)<1e-4:
        p = np.pi/4 * np.sinc(1/2/beta)
    else:
        p = np.sinc(t/Ts) * np.cos(np.pi*beta*t/Ts)/(1-(2*beta*t/Ts)**2)
    return p


def array_response(Nh,Nv, Angle_H, Angle_V, f,fc, array_type = 'UPA', AtDs=0.5):
    '''
    This function defines a steering vector for a Nh*Nv uniform planar array (UPA).
    See paper 'Dynamic Hybrid Beamforming With Low-Resolution PSs for Wideband mmWave MIMO-OFDM Systems'
    :param Nh: number of antennas in horizontal direction
    :param Nv: number of antennas in vertical direction
    :param fc: carrier frequency
    :param f: actual frequency
    :param AtDs: normalized antenna spacing distance, set to 0.5 by default
    :return: steering a vector at frequency f with azimuth and elevation angles
    '''
    N = int(Nh*Nv)
    Np = Angle_H.shape[0]
    AtDs_h = AtDs
    AtDs_v = AtDs
    array_matrix = np.zeros([N,Np], dtype=np.complex_)
    if array_type == 'ULA':
        spatial_h = np.sin(Angle_H)
        factor_h = np.array(range(N))
        for n in range(Np):
            array_matrix[:, n] = 1/np.sqrt(N)*np.exp(1j*2*np.pi * AtDs_h* factor_h*f/fc*spatial_h[n])

    else:
        # Nh, Nv = array_dimension(N)
        spatial_h = np.sin(Angle_H) * np.sin(Angle_V)
        spatial_v = np.cos(Angle_V)
        factor_h = np.array(range(Nh))
        factor_v = np.array(range(Nv))
        for n in range(Np):
            steering_vector_h = 1/np.sqrt(Nh) * np.exp(1j*2*np.pi * AtDs_h* factor_h*f/fc*spatial_h[n])
            steering_vector_v = 1/np.sqrt(Nv) * np.exp(1j*2*np.pi* AtDs_v * factor_v*f/fc*spatial_v[n])
            array_matrix[:,n] = np.kron(steering_vector_h, steering_vector_v)
    ccc = 1
    return array_matrix


def channel_model(Nt, Nr, Pulse_Filter = True, fc=fc, Ncl=Ncl, Nray=Nray, bandwidth=Bandwidth):
    Np = Ncl * Nray
    gamma = np.sqrt(Nt * Nr / Np)  # normalization factor
    sigma = 1  # according to the normalization condition of the H
    Ntv = 4
    Nth = Nt // Ntv

    Nrh = 2
    Nrv = Nr // Nrh

    beta = 1
    Ts = 1/Bandwidth
    Delay_taps = int(K/4)
    angle_sigma = 10 / 180 * np.pi  # standard deviation of the angles in azimuth and elevation both of Rx and Tx

    AoH_all = np.zeros([2, Np])  # azimuth angle at Tx and Rx
    AoV_all = np.zeros([2, Np])  # elevation angle at Tx and Rx

    for cc in range(Ncl):
        AoH = np.random.uniform(0, 2, 2) * np.pi
        AoV = np.random.uniform(-0.5, 0.5, 2) * np.pi

        AoH_all[0, cc * Nray:(cc + 1) * Nray] = np.random.uniform(0, 2, Nray) * np.pi
        AoH_all[1, cc * Nray:(cc + 1) * Nray] = np.random.uniform(0, 2, Nray) * np.pi
        AoV_all[0, cc * Nray:(cc + 1) * Nray] = np.random.uniform(-0.5, 0.5, Nray) * np.pi
        AoV_all[1, cc * Nray:(cc + 1) * Nray] = np.random.uniform(-0.5, 0.5, Nray) * np.pi

        # med = np.random.laplace(AoD_m[0], angle_sigma, Nray)

        # AoH_all[0, cc * Nray:(cc + 1) * Nray] = np.random.laplace(AoH[0], angle_sigma, Nray)
        # AoH_all[1, cc * Nray:(cc + 1) * Nray] = np.random.laplace(AoH[1], angle_sigma, Nray)
        # AoV_all[0, cc * Nray:(cc + 1) * Nray] = np.random.laplace(AoV[0], angle_sigma, Nray)
        # AoV_all[1, cc * Nray:(cc + 1) * Nray] = np.random.laplace(AoV[1], angle_sigma, Nray)

    # alpha = np.sqrt(sigma / 2) * (
    #         np.random.normal(0, 1, size=[Np, K]) + 1j * np.random.normal(0, 1, size=[Np, K]))
    alpha = np.sqrt(sigma / 2) * (
            np.random.normal(0, 1, size=[Np, ]) + 1j * np.random.normal(0, 1, size=[Np, ]))
    Delay = np.random.uniform(0, Delay_taps, size=Np) * Ts
    # AoH_all = np.random.uniform(-1, 1, size=[2, Np]) * np.pi
    # AoV_all = np.random.uniform(-0.5, 0.5, size=[2, Np]) * np.pi
    Coef_matrix = np.zeros([Np, K], dtype='complex_')
    H_all = np.zeros([Nr, Nt, K], dtype='complex_')
    At_all = np.zeros([Nt, Np, K], dtype='complex_')  # use to save testing data, used latter in Matlab
    for k in range(K):
        # fk = 2
        fk = fc + bandwidth * (2 * k - K + 1) / (2 * K)
        At = array_response(Nth, Ntv, AoH_all[0, :], AoV_all[0, :], fk, fc, array_type=Array_Type)
        Ar = array_response(Nrh, Nrv, AoH_all[1, :], AoV_all[1, :], fk, fc, array_type=Array_Type)

        # AhA_t=np.matmul(At.conj().T, At)
        # AhA_r = np.matmul(Ar.conj().T, Ar)

        At_all[:, :, k] = At
        for n in range(Np):
            if Pulse_Filter:
                med = 0
                for d in range(Delay_taps):
                    med += pulase_filter(d * Ts - Delay[n], Ts, beta) * np.exp(-1j * 2 * np.pi * k * d / K)
                Coef_matrix[n, k] = med
            else:
                Coef_matrix[n, k] = np.exp(-1j * 2 * np.pi * Delay[n] * fk)
        gain = gamma * Coef_matrix[:, k] * alpha#[:, k]
        H_all[:, :, k] = np.matmul(np.matmul(Ar, np.diag(gain)), At.conj().T)
        # power_H = np.linalg.norm(H_all[:, :, k],'fro') ** 2 / (Nr * Nt)
        # print(f'channel power is {power_H}')

    if Noisy_Channel:
        noise = np.sqrt(1 / 2) * (np.random.normal(0, 1, size=[Nr, Nt, K]) + 1j * np.random.normal(0, 1, size=[Nr, Nt, K]))
        H_all = np.sqrt(1-epision) * H_all + np.sqrt(epision*Nr*Nt) * noise
        ccc = 1
    return H_all, At_all


def gen_data_wideband(Nt, Nr, Nrf, Ns, batch_size=1,
                      Sub_Connected=False,
                      Sub_Structure_Type='fixed',
                      Pulse_Filter=False,
                      fc=fc, Ncl=Ncl, Nray=Nray, bandwidth=Bandwidth):
    # data to get
    N = 2 * Nt * Nrf  # true for Nrf = Ns
    batch_X = np.zeros([batch_size, N], dtype='float32')  # only use when employing supervised learning

    batch_z = np.zeros([batch_size, N, K], dtype='float32')  # input to DNN for training
    batch_B = np.zeros([batch_size, N, N, K], dtype='float32')  # input to DNN to compute loss function
    batch_Bz = np.zeros([batch_size, N, K], dtype='float32')  # input to DNN for training
    batch_BB = np.zeros([batch_size, N, N, K], dtype='float32')  # input to DNN for training
    batch_AA = np.zeros([batch_size, N, N, K], dtype='float32')  # input to DNN for training

    batch_H = np.zeros([batch_size, Nr, Nt, K], dtype='complex_')  # use to save testing data, used latter in Matlab
    batch_Fopt = np.zeros([batch_size, Nt, Ns, K], dtype='complex_')  # use to save testing data, used latter in Matlab
    batch_Wopt = np.zeros([batch_size, Nr, Ns, K], dtype='complex_')  # use to save testing data, used latter in Matlab
    batch_Fbb = np.zeros([batch_size, Nrf, Ns, K], dtype='complex_')  # use to save testing data, used latter in Matlab
    batch_At = np.zeros([batch_size, Nt, Ncl*Nray, K], dtype='complex_')  # use to save testing data, used latter in Matlab

    for ii in range(batch_size):
        if init_scheme == 0:
            FRF = np.exp(1j * np.random.uniform(0, 2 * np.pi, [Nt, Nrf]))  # frequency-flat
            # FRF = normalize(FRF, Nt, Nrf, sub_connected=Sub_Connected, sub_structure_type=Sub_Structure_Type)
            FRF_vec = FRF.flatten('F')
            batch_X[ii, :] = np.concatenate((FRF_vec.real, FRF_vec.imag), axis=0)


        H_ii, At_ii = channel_model(Nt, Nr, Pulse_Filter=Pulse_Filter, fc=fc, Ncl=Ncl, Nray=Nray, bandwidth=bandwidth)
        batch_H[ii, :, :, :] = H_ii
        batch_At[ii, :, :, :] = At_ii

        for k in range(K):
            At = At_ii[:, :, k]
            U, S, VH = np.linalg.svd(H_ii[:, :, k])
            V = VH.T.conj()
            Fopt = V[:, 0:Ns]   # np.sqrt(Ns) *
            Wopt = U[:, 0:Ns]

            ## construct training data
            ztilde = Fopt.flatten('F')
            z = np.concatenate((ztilde.real, ztilde.imag), axis=0)  # convert to real values
            # z_vector = np.matrix(z)
            if init_scheme == 0: # random FRF, FBB = LS solution
                FBB = np.matmul(np.linalg.pinv(FRF), Fopt)
                FBB = np.sqrt(Ns) * FBB / np.linalg.norm(np.matmul(FRF, FBB), 'fro')
            else: # obtain FRF and FBB based on OMP for all frequencies ==> better
                FRF, FBB = OMP(Fopt, At)

            # FBB = np.matmul(np.linalg.pinv(FRF), Fopt)

            Btilde = np.kron(FBB.T, np.identity(Nt))
            B1 = np.concatenate((Btilde.real, -Btilde.imag), axis=1)
            B2 = np.concatenate((Btilde.imag, Btilde.real), axis=1)
            B = np.concatenate((B1, B2), axis=0)
            # print(np.shape(B))

            # new for array response
            AtH = At.conj().T
            Atilde = np.kron(np.identity(Nrf), AtH)
            A1 = np.concatenate((Atilde.real, -Atilde.imag), axis=1)
            A2 = np.concatenate((Atilde.imag, Atilde.real), axis=1)
            A = np.concatenate((A1, A2), axis=0)
            # print(np.shape(A))

            # Assign data to the ii-th batch
            # err = z_vector.dot(B) -np.matmul(B.T, z)
            # err1 = np.matmul(z_vector,B) - z_vector.dot(B)

            batch_Bz[ii, :, k] = np.matmul(B.T, z)
            batch_BB[ii, :, :, k] = np.matmul(B.T, B)
            batch_z[ii, :, k] = z
            batch_B[ii, :, :, k] = B.T
            batch_Fopt[ii, :, :, k] = Fopt
            batch_Wopt[ii, :, :, k] = Wopt
            batch_Fbb[ii, :, :, k] = FBB
            batch_AA[ii, :, :, k] = np.matmul(A.T, A)

    # dis_sum = 0
    # for k in range(K):
    #     med = np.matmul(np.expand_dims(batch_X,1), batch_B[:,:,:, k]).squeeze()
    #     diff = batch_z[ :, :, k] - med
    #     dis_sum += np.sum(np.linalg.norm(diff, 'fro'))
    # print(f'{ii} error:{dis_sum}')
    # ccc = 1
    return batch_Bz, batch_BB, batch_X, batch_z, batch_B, batch_H, batch_Fopt, batch_Wopt, batch_Fbb, batch_AA, batch_At


def gen_data_large(Nt, Nr, Nrf, Ns, Num_batch,batch_size=1, fc=fc, Ncl=Ncl, Nray=Nray, bandwidth=Bandwidth, Pulse_Filter=False, data='taining'):
    def append_data(data_set, num_data, new_data):
        dims = list(data_set.shape)
        num_sp = dims[0] + num_data
        dims_new = list(dims[1:])
        dims_new.insert(0, num_sp)
        data_set.resize(tuple(dims_new))
        data_set[dims[0]:num_sp] = new_data
        return data_set
    # Channel setup
    channel_type = 'geometry'
    # data to get
    data_name = train_data_name
    if data == 'testing':
        data_name = test_data_name
    data_path = dataset_file + data_name
    hf = h5py.File(data_path, 'a')
    batch_Bz_set = hf.get('batch_Bz')
    batch_BB_set = hf.get('batch_BB')
    batch_X_set = hf.get('batch_X')
    batch_Z_set = hf.get('batch_Z')
    batch_B_set = hf.get('batch_B')
    batch_H_real_set = hf.get('batch_H_real')
    batch_H_imag_set = hf.get('batch_H_imag')
    batch_Fopt_real_set = hf.get('batch_Fopt_real')
    batch_Fopt_imag_set = hf.get('batch_Fopt_imag')
    if data == 'testing':
        batch_Wopt_real_set = hf.get('batch_Wopt_real')
        batch_Wopt_imag_set = hf.get('batch_Wopt_imag')
        batch_Fbb_real_set = hf.get('batch_Fbb_real')
        batch_Fbb_imag_set = hf.get('batch_Fbb_imag')
        batch_At_real_set = hf.get('batch_At_real')
        batch_At_imag_set = hf.get('batch_At_imag')


    # data to get
    N = 2 * Nt * Nrf  # true for Nrf = Ns
    batch_X = np.zeros([batch_size, N], dtype='float32')  # only use when employing supervised learning
    batch_z = np.zeros([batch_size, N, K], dtype='float32')  # input to DNN for training
    batch_B = np.zeros([batch_size, N, N, K], dtype='float32')  # input to DNN to compute loss function
    batch_Bz = np.zeros([batch_size, N, K], dtype='float32')  # input to DNN for training
    batch_BB = np.zeros([batch_size, N, N, K], dtype='float32')  # input to DNN for training
    batch_AA = np.zeros([batch_size, N, N, K], dtype='float32')  # input to DNN for training

    batch_H = np.zeros([batch_size, Nr, Nt, K], dtype='complex_')  # use to save testing data, used latter in Matlab
    batch_Fopt = np.zeros([batch_size, Nt, Ns, K], dtype='complex_')  # use to save testing data, used latter in Matlab
    batch_Wopt = np.zeros([batch_size, Nr, Ns, K], dtype='complex_')  # use to save testing data, used latter in Matlab
    batch_Fbb = np.zeros([batch_size, Nrf, Ns, K], dtype='complex_')  # use to save testing data, used latter in Matlab
    batch_At = np.zeros([batch_size, Nt, Ncl*Nray, K], dtype='complex_')  # use to save testing data, used latter in Matlab


    for n in range(Num_batch):
        print(f'Generating {n}th batch data', flush=True)
        for ii in range(batch_size):
            if init_scheme == 0:
                FRF = np.exp(1j * np.random.uniform(0, 2 * np.pi, [Nt, Nrf]))  # frequency-flat
                FRF_vec = FRF.flatten('F')
                batch_X[ii, :] = np.concatenate((FRF_vec.real, FRF_vec.imag), axis=0)

            # generate channel matrix
            if channel_type == 'Rician':
                Hii = 1 / np.sqrt(2) * (np.random.randn(Nr, Nt, K) + 1j * np.random.randn(Nr, Nt, K))
                batch_H[ii, :, :, :] = Hii
            else:
                H_ii, At_ii = channel_model(Nt, Nr, Pulse_Filter=Pulse_Filter, fc=fc, Ncl=Ncl, Nray=Nray, bandwidth=bandwidth)
                batch_H[ii, :, :, :] = H_ii
                batch_At[ii, :, :, :] = At_ii
                for k in range(K):
                    At = At_ii[:, :, k]
                    U, S, VH = np.linalg.svd(H_ii[:, :, k])
                    V = VH.T.conj()
                    Fopt = V[:, 0:Ns]  # np.sqrt(Ns) *
                    Wopt = U[:, 0:Ns]

                    ## construct training data
                    ztilde = Fopt.flatten('F')
                    z = np.concatenate((ztilde.real, ztilde.imag), axis=0)  # convert to real values
                    # z_vector = np.matrix(z)

                    if init_scheme == 0:  # random FRF, FBB = LS solution
                        FBB = np.matmul(np.linalg.pinv(FRF), Fopt)
                        FBB = np.sqrt(Ns) * FBB / np.linalg.norm(np.matmul(FRF, FBB), 'fro')
                    else:  # obtain FRF and FBB based on OMP for all frequencies ==> better
                        FRF, FBB = OMP(Fopt, At)

                    Btilde = np.kron(FBB.T, np.identity(Nt))
                    B1 = np.concatenate((Btilde.real, -Btilde.imag), axis=1)
                    B2 = np.concatenate((Btilde.imag, Btilde.real), axis=1)
                    B = np.concatenate((B1, B2), axis=0)
                    # print(np.shape(B))

                    # new for array response
                    AtH = At.conj().T
                    Atilde = np.kron(np.identity(Nrf), AtH)
                    A1 = np.concatenate((Atilde.real, -Atilde.imag), axis=1)
                    A2 = np.concatenate((Atilde.imag, Atilde.real), axis=1)
                    A = np.concatenate((A1, A2), axis=0)
                    # print(np.shape(A))

                    # Assign data to the ii-th batch
                    # err = z_vector.dot(B) -np.matmul(B.T, z)
                    # err1 = np.matmul(z_vector,B) - z_vector.dot(B)

                    batch_Bz[ii, :, k] = np.matmul(B.T, z)
                    batch_BB[ii, :, :, k] = np.matmul(B.T, B)
                    batch_z[ii, :, k] = z
                    batch_B[ii, :, :, k] = B.T
                    batch_Fopt[ii, :, :, k] = Fopt
                    batch_Wopt[ii, :, :, k] = Wopt
                    batch_Fbb[ii, :, :, k] = FBB
                    batch_AA[ii, :, :, k] = np.matmul(A.T, A)

            # Hgap = np.linalg.norm(H,ord='fro')/np.sqrt(Nt*Nr)
            # print(f'HQ is: {Hgap:.4f}')
            # Compute optimal digital precoder


        batch_Bz_set = append_data(batch_Bz_set, batch_size, batch_Bz)  # add new data into set
        batch_BB_set = append_data(batch_BB_set, batch_size, batch_BB)
        batch_X_set = append_data(batch_X_set, batch_size, batch_X)
        batch_Z_set = append_data(batch_Z_set, batch_size, batch_z)
        batch_B_set = append_data(batch_B_set, batch_size, batch_B)

        batch_H_real_set = append_data(batch_H_real_set, batch_size, batch_H.real)
        batch_H_imag_set = append_data(batch_H_imag_set, batch_size, batch_H.imag)

        batch_Fopt_real_set = append_data(batch_Fopt_real_set, batch_size, batch_Fopt.real)
        batch_Fopt_imag_set = append_data(batch_Fopt_imag_set, batch_size, batch_Fopt.imag)
        if data == 'testing':
            batch_Wopt_real_set = append_data(batch_Wopt_real_set, batch_size, batch_Wopt.real)
            batch_Wopt_imag_set = append_data(batch_Wopt_imag_set, batch_size, batch_Wopt.imag)

            batch_Fbb_real_set = append_data(batch_Fbb_real_set, batch_size, batch_Fbb.real)
            batch_Fbb_imag_set = append_data(batch_Fbb_imag_set, batch_size, batch_Fbb.imag)

            batch_At_real_set = append_data(batch_At_real_set, batch_size, batch_At.real)
            batch_At_imag_set = append_data(batch_At_imag_set, batch_size, batch_At.imag)



    ccc = 1

class Data_Fetch():
    def __init__(self, file_dir, file_name, batch_size, training_set_size, training_set_size_truncated=training_set_size, data_str='training'):
        self.data_path = file_dir + file_name
        self.batch_size = batch_size
        self.data_str = data_str
        self.len = training_set_size+1
        self.len_truncated = training_set_size_truncated +1
        self.reset()

    def reset(self):
        self.pointer = np.random.randint(self.len_truncated)  # initialize the start position
        self.start_idx = self.pointer

    def get_item(self):
        data_all = h5py.File(self.data_path, 'r')

        self.end_idx = self.start_idx + self.batch_size
        if self.end_idx <= self.len_truncated-1:

            Bz = data_all['batch_Bz'][self.start_idx:self.end_idx, :, :]
            BB = data_all['batch_BB'][self.start_idx:self.end_idx, :, :, :]
            X = data_all['batch_X'][self.start_idx:self.end_idx, :]
            Z = data_all['batch_Z'][self.start_idx:self.end_idx, :, :]
            B = data_all['batch_B'][self.start_idx:self.end_idx, :, :, :]
            batch_H_real = data_all['batch_H_real'][self.start_idx:self.end_idx, :, :, :]
            batch_H_imag = data_all['batch_H_imag'][self.start_idx:self.end_idx, :, :, :]
            batch_Fopt_real = data_all['batch_Fopt_real'][self.start_idx:self.end_idx, :, :, :]
            batch_Fopt_imag = data_all['batch_Fopt_imag'][self.start_idx:self.end_idx, :, :, :]


            if self.data_str== 'testing':
                batch_Wopt_real = data_all['batch_Wopt_real'][self.start_idx:self.end_idx, :, :, :]
                batch_Wopt_imag = data_all['batch_Wopt_imag'][self.start_idx:self.end_idx, :, :, :]
                batch_Fbb_real = data_all['batch_Fbb_real'][self.start_idx:self.end_idx, :, :, :]
                batch_Fbb_imag = data_all['batch_Fbb_imag'][self.start_idx:self.end_idx, :, :, :]
                batch_At_real = data_all['batch_At_real'][self.start_idx:self.end_idx, :, :, :]
                batch_At_imag = data_all['batch_At_imag'][self.start_idx:self.end_idx, :, :, :]

                batch_Wopt = batch_Wopt_real + 1j * batch_Wopt_imag
                batch_Fbb = batch_Fbb_real + 1j * batch_Fbb_imag
                batch_At = batch_At_real + 1j * batch_At_imag

            data_all.close()
            self.start_idx = self.end_idx

        else:
            remain_num = self.end_idx - self.len_truncated

            Bz1 = data_all['batch_Bz'][self.start_idx:self.len_truncated, :, :]
            BB1 = data_all['batch_BB'][self.start_idx:self.len_truncated, :, :, :]
            X1 = data_all['batch_X'][self.start_idx:self.len_truncated, :]
            Z1 = data_all['batch_Z'][self.start_idx:self.len_truncated, :, :]
            B1 = data_all['batch_B'][self.start_idx:self.len_truncated, :, :, :]
            batch_H_real1 = data_all['batch_H_real'][self.start_idx:self.len_truncated, :, :, :]
            batch_H_imag1 = data_all['batch_H_imag'][self.start_idx:self.len_truncated, :, :, :]
            batch_Fopt_real1 = data_all['batch_Fopt_real'][self.start_idx:self.len_truncated, :, :, :]
            batch_Fopt_imag1 = data_all['batch_Fopt_imag'][self.start_idx:self.len_truncated, :, :, :]

            Bz2 = data_all['batch_Bz'][:remain_num, :, :]
            BB2 = data_all['batch_BB'][:remain_num, :, :, :]
            X2 = data_all['batch_X'][:remain_num, :]
            Z2 = data_all['batch_Z'][:remain_num, :, :]
            B2 = data_all['batch_B'][:remain_num, :, :, :]
            batch_H_real2 = data_all['batch_H_real'][:remain_num, :, :, :]
            batch_H_imag2 = data_all['batch_H_imag'][:remain_num, :, :, :]
            batch_Fopt_real2 = data_all['batch_Fopt_real'][:remain_num, :, :, :]
            batch_Fopt_imag2 = data_all['batch_Fopt_imag'][:remain_num, :, :, :]

            Bz = np.concatenate((Bz1, Bz2), axis=0)
            BB = np.concatenate((BB1, BB2), axis=0)
            X = np.concatenate((X1, X2), axis=0)
            Z = np.concatenate((Z1, Z2), axis=0)
            B = np.concatenate((B1, B2), axis=0)
            batch_H_real = np.concatenate((batch_H_real1, batch_H_real2), axis=0)
            batch_H_imag = np.concatenate((batch_H_imag1, batch_H_imag2), axis=0)
            batch_Fopt_real = np.concatenate((batch_Fopt_real1, batch_Fopt_real2), axis=0)
            batch_Fopt_imag = np.concatenate((batch_Fopt_imag1, batch_Fopt_imag2), axis=0)


            data_all.close()
            self.start_idx = remain_num

        batch_H = batch_H_real + 1j * batch_H_imag
        batch_Fopt = batch_Fopt_real + 1j * batch_Fopt_imag
        if self.data_str == 'testing':
            return Bz, BB, X, Z, B, batch_H, batch_Fopt, batch_Wopt, batch_Fbb, batch_At
        else:
            return Bz, BB, X, Z, B, batch_H, batch_Fopt


if __name__ == '__main__':



    def generate_training_data():
        # training_set_size = 70
        print('----------------------training data-------------------------')
        batch_Bz, batch_BB, batch_X, batch_Z, batch_B, batch_H, batch_Fopt, batch_Wopt, batch_Fbb, batch_AA, batch_At = gen_data_wideband(
            Nt, Nr, Nrf, Ns, batch_size=1, fc=fc, Ncl=Ncl, Nray=Nray, bandwidth=Bandwidth) # batch_size=Gen_Batch_size
        data_all = {'batch_Bz': batch_Bz, 'batch_BB': batch_BB, 'batch_X': batch_X, 'batch_Z': batch_Z,
                    'batch_B': batch_B,
                    'batch_H_real': batch_H.real,
                    'batch_H_imag': batch_H.imag,
                    'batch_Fopt_real': batch_Fopt.real,
                    'batch_Fopt_imag': batch_Fopt.imag,
                    }

        train_data_path = dataset_file + train_data_name
        file_handle = h5py.File(train_data_path, 'w')
        for name in data_all:
            dshp = data_all[name].shape
            dims = list(dshp[1:])
            dims.insert(0, None)
            # print(f'dshp shape:{dshp}, dims shape:{dims}')
            file_handle.create_dataset(name, data=data_all[name], maxshape=dims, chunks=True, compression='gzip',
                                       compression_opts=9)
        # hf = h5py.File(train_data_path, 'r')
        # print('----------------------training data-------------------------')
        # for key in hf.keys():
        #     print(key, hf[key])


    def generate_testing_data(Pulse_Filter=False):
        print('----------------------testing data-------------------------')
        batch_Bz, batch_BB, batch_X, batch_Z, batch_B, batch_H, batch_Fopt, batch_Wopt, batch_Fbb, batch_AA, batch_At = gen_data_wideband(
            Nt, Nr, Nrf, Ns, batch_size=1, Pulse_Filter=Pulse_Filter, fc=fc, Ncl=Ncl, Nray=Nray, bandwidth=Bandwidth)  # batch_size=Gen_Batch_size
        data_all = {'batch_Bz': batch_Bz, 'batch_BB': batch_BB, 'batch_X': batch_X, 'batch_Z': batch_Z, 'batch_B': batch_B,
                    'batch_H_real': batch_H.real,
                    'batch_H_imag': batch_H.imag,
                    'batch_Fopt_real': batch_Fopt.real,
                    'batch_Fopt_imag': batch_Fopt.imag,
                    'batch_Wopt_real': batch_Wopt.real,
                    'batch_Wopt_imag': batch_Wopt.imag,
                    'batch_Fbb_real': batch_Fbb.real,
                    'batch_Fbb_imag': batch_Fbb.imag,
                    'batch_At_real': batch_At.real,
                    'batch_At_imag':  batch_At.imag
                    }

        test_data_path = dataset_file + test_data_name
        file_handle = h5py.File(test_data_path, 'w')
        for name in data_all:
            dshp = data_all[name].shape
            dims = list(dshp[1:])
            dims.insert(0, None)
            file_handle.create_dataset(name, data=data_all[name], maxshape=dims, chunks=True, compression='gzip',
                                       compression_opts=9)
            # file_handle.create_dataset(name, data=data_all[name], chunks=True, compression='gzip',
            #                            compression_opts=9)
        # print(name)

        hf = h5py.File(test_data_path, 'r')
        print('----------------------testing data-------------------------')
        for key in hf.keys():
            print(key, hf[key])


    generate_testing_data()
    gen_data_large(Nt, Nr, Nrf, Ns, Num_batch=GenNum_Batch_te, batch_size=Gen_Batch_size_te, fc=fc, Ncl=Ncl, Nray=Nray,
                   bandwidth=Bandwidth, data='testing')

    generate_training_data()
    gen_data_large(Nt, Nr, Nrf, Ns, Num_batch=GenNum_Batch_tr, batch_size=Gen_Batch_size_tr, fc=fc, Ncl=Ncl, Nray=Nray, bandwidth=Bandwidth)

    train_data_path = dataset_file + train_data_name
    hf = h5py.File(train_data_path, 'r')
    print('----------------------training data-------------------------')
    for key in hf.keys():
        print(key, hf[key])

    test_data_path = dataset_file + test_data_name
    hf = h5py.File(test_data_path, 'r')
    print('----------------------testing data-------------------------')
    for key in hf.keys():
        print(key, hf[key])

    ccc=1