# -*- coding: utf-8 -*-
"""
@author: Lijie Wang (lijiewang@zhejianglab.com)

function: TDCA
"""


import numpy as np
from scipy import signal as sig
import scipy.io as scio
import math
import os



# class TDCA
class TDCA(object):

    def __init__(self, Fs, point_delay, Nk, f_array, harmonic_num, norm):
        '''
        Parameters
        ----------
        Fs : int
             sampling rate
        point_delay: int
                     the number of delayed points
        Nk : int
             the number of subspace
        f_array: array
                 all stimulus frequency
        harmonic_num: int
                      the number of harmonics
        norm: bool
              whether the data needs to be standardized
        '''
        self.Fs = Fs
        self.point_delay = point_delay
        self.Nk = Nk
        self.f_array = f_array
        self.harmonic_num = harmonic_num
        self.norm = norm
        
        
    def build_reference_template(self, N):
        longest_Tw = 5
        templ_len = int(self.Fs * longest_Tw)
        samp_point = np.linspace(0, (templ_len - 1) / self.Fs, int(templ_len), endpoint=True)
        samp_point = samp_point.reshape(1, len(samp_point))
        template_set = []
        for i in range(self.f_array.shape[0]):
            cs_set = np.zeros((self.harmonic_num * 2, templ_len))
            test_freq = np.linspace(self.f_array[i], self.f_array[i] * self.harmonic_num, int(self.harmonic_num), endpoint=True)
            test_freq = test_freq.reshape(1, len(test_freq))
            num_matrix = 2 * np.pi * np.dot(test_freq.T, samp_point)
            cos_set = np.cos(num_matrix)
            sin_set = np.sin(num_matrix)
            for j in range(self.harmonic_num):
                     cs_set[2 * j, :] = sin_set[j, :]
                     cs_set[2 * j + 1, :]  = cos_set[j, :]
            template_set.append(cs_set)
        self.reference_template = np.array(template_set)   
        
        self.P_matrices = []
        for j in np.arange(self.f_array.shape[0]):
            pre_reftemplate = self.reference_template[j, :, :N] 
            Q, _ = np.linalg.qr(pre_reftemplate.T)
            P = np.dot(Q, Q.T)
            self.P_matrices.append(P)
        
    def z_score_normalization(self, data):
        # function for data normalization
        return (data - np.mean(data, axis=1, keepdims=True)) / np.std(data, axis=1, keepdims=True)
      
      
    def train(self, traindata, Tw):
        '''
        Parameters
        ----------
        traindata: four-dim numpy array
                  (block_num * class_num * channle_num * sample_num)
        Tw: float 64
            present time window
        '''
        block_num, class_num, channel_num, _ = traindata.shape
        N = round(Tw * self.Fs)
        
        # X_a_train
        X_a_train = np.zeros((block_num, class_num, channel_num * (self.point_delay + 1), N * 2))
        
        for j in np.arange(class_num):
            pre_reftemplate = self.reference_template[j, :, :N] 
            Q, _ = np.linalg.qr(pre_reftemplate.T)
            P = np.dot(Q, Q.T)
            self.P_matrices.append(P)
            
            extended_size = channel_num * (self.point_delay + 1)
            x_wave = np.zeros((extended_size, N))
            for i in np.arange(block_num):
                x_wave[:channel_num, :] = traindata[i, j, ...]  # First part is just the data
                for dn in range(1, self.point_delay + 1):
                    x_wave[channel_num*dn:channel_num*(dn+1), :-dn] = traindata[i, j, :, dn:]
                
                # Z-score Normalization
                if self.norm:
                    x_wave = self.z_score_normalization(x_wave)

                x_wave_p = np.dot(x_wave, P)
                if self.norm:
                    x_wave_p = self.z_score_normalization(x_wave_p)
                
                x_a = np.concatenate((x_wave, x_wave_p), axis = 1)
                X_a_train[i, j , ...] = x_a
        
        # Sb Sw individual_template
        Sb = np.zeros((channel_num * (self.point_delay + 1), channel_num * (self.point_delay + 1)))
        Sw = np.zeros_like(Sb)
        
        X_a_train_mean = X_a_train.mean(axis=(0, 1))

        # Calculate the mean value of each class
        class_mean = np.mean(X_a_train, axis=0)  
        for j in range(class_num):
            intra_diff = class_mean[j,...]- X_a_train_mean
            Sb += np.dot(intra_diff, intra_diff.T)
            
            for i in range(block_num):
                if block_num == 1:
                    inter_diff = X_a_train[i,j,:,:] 
                else:
                    inter_diff = X_a_train[i,j,:,:] - class_mean[j,...]
                Sw += np.dot(inter_diff, inter_diff.T)
            
            
        Sb /= class_num
        Sw /= (block_num * class_num)  # Normalized Sw
        
        # w
        e_val, e_vec = np.linalg.eig(np.linalg.inv(Sw) @ Sb)
        # e_val = np.real(e_val)
        # e_vec = np.real(e_vec)
        sorted_indices = np.argsort(e_val)[::-1]
        tdca_W = e_vec[:, sorted_indices][:, :self.Nk]
        
        self.tdca_W = tdca_W
        self.individual_template = class_mean
    
        
    def check(self, testdata):
        '''
        Parameters
        ----------
        testdata: three-dim numpy array
                  classnum * channle_num * sample_num
        '''
        class_num, channel_num, N = testdata.shape
        coef_1 = np.zeros((class_num, class_num))
        
        # testing trial
        for ts_i in range(class_num):
            pre_test = testdata[ts_i, ...]
            
            # To be matched
            for c_i in range(class_num): 

                P = self.P_matrices[c_i]
                
                # Preallocate space for test_wave and avoid using append and concatenate in a loop
                extended_size = channel_num * (self.point_delay + 1)
                test_wave = np.zeros((extended_size, N))
                
                test_wave[:channel_num, :] = pre_test  # First part is just the data
                for dn in range(1, self.point_delay + 1):
                    test_wave[channel_num*dn:channel_num*(dn+1), :-dn] = pre_test[:, dn:]
         
                # Z-score Normalization 
                if self.norm:
                    test_wave = self.z_score_normalization(test_wave)
                    
                test_wave_p = np.dot(test_wave, P)
                if self.norm:
                    test_wave_p = self.z_score_normalization(test_wave_p)
                    
                test_a = np.concatenate((test_wave, test_wave_p), axis = 1)
                
                 
                map_template = self.tdca_W.T.dot(self.individual_template[c_i, ...]).ravel()
                map_test = self.tdca_W.T.dot(test_a).ravel()
                coef_1[ts_i, c_i] += np.corrcoef(map_template, map_test)[0, 1]
                
        return coef_1
    
 

# Encapsulate a process for one subject & one condition
def recognize_onebody(onecon_tup):

    # print process information
    print(onecon_tup)

    
    # get parameters of the present process
    dataset_no, train_block_num, Tw, sub = onecon_tup
    
    # eeg data storage path
    base_path = "data"
    MainPath = base_path + "/Benchmark/dataset" if dataset_no == 1 else base_path + "/BETA/dataset"
    
    
    # dataset parameters
    block_num, subject_num, visual_delay, Nk, point_delay, Fs = (6, 35, 0.14, 8, 5, 250) if dataset_no == 1 else (4, 70, 0.13, 9, 3, 250)   
    latencydelay = round(visual_delay * Fs)
    N_t = round(Fs * Tw)
    num_SN = 5
    harmonic_num = 5
    norm = False
    ssvep_channel = np.array([48, 54, 55, 56, 57, 58, 61, 62, 63]) - 1

    
    # design bank filter
    BankWeight = np.zeros((num_SN))
    for fb_i in range(num_SN):
        BankWeight[fb_i] = math.pow((fb_i + 1),-1.25) + 0.25       
    fb_meter = [[(6,90),(2,100)],
           [(14,90),(10,100)],
           [(22,90),(12,100)],
           [(30,90),(20,100)],
           [(38,90),(28,100)],
           [(46,90),(36,100)],
           [(54,90),(44,100)]]
    gpassstopmeter = [[2,20],
                    [2,25],
                    [3,30],
                    [3,30],
                    [3,40],
                    [3,40],
                    [3,40]]
    filter_bank = []
    for fb_i in range(num_SN):
        Wp = fb_meter[fb_i][0]
        Ws = fb_meter[fb_i][1]
        gpass = gpassstopmeter[fb_i][0]
        gstop = gpassstopmeter[fb_i][1]
        N, Wn = sig.cheb1ord(Wp, Ws, gpass, gstop, fs=Fs)
        B, A = sig.cheby1(N, 0.5, Wn, btype="bandpass", fs=Fs)
        filter_bank.append((B, A))
    # notch filter
    f0 = 50    # Frequency to be removed from signal (Hz) 
    Q = 35     # Quality factor
    NotchB, NotchA = sig.iircomb(f0, Q, ftype='notch',fs = Fs)
    
    # load eeg data
    if dataset_no == 1:
        data_path = f"{MainPath}/S{'0' if sub < 9 else ''}{sub + 1}.mat"
        eeg = scio.loadmat(data_path)
        data = eeg['data']
    else:
        data_path = f"{MainPath}/S{sub + 1}.mat"
        eeg_data = scio.loadmat(data_path)['data']
        data = eeg_data['EEG'][0, 0]
        # Change the order of BETA's array dimensions(Benchmark 64 * 1500 * 40 * 6， BETA 64 * 750/1000 * 4 * 40)
        data = data.transpose(0, 1, 3, 2)

        # get BETA frequency information
        suppl_info = eeg_data['suppl_info'][0, 0]
        BETA_freq = suppl_info['freqs'][0, 0].flatten()

    # set frequency parameters for two datasets and select present frequency list
    if dataset_no == 1:
        Bench_freq = np.array([x + 8.0 for x in range(8)] + \
                                [x + 8.2 for x in range(8)] + \
                                [x + 8.4 for x in range(8)] + \
                                [x + 8.6 for x in range(8)] + \
                                [x + 8.8 for x in range(8)])
        Bench_freq = np.round(Bench_freq, 1)
        f_array = Bench_freq
    else:
        BETA_freq = np.round(BETA_freq, 1)
        f_array = BETA_freq

    # cut data
    data = data[ssvep_channel, ...]
    data = data[:, round(0.5 * Fs) : round(0.5* Fs) + latencydelay + N_t, :, :]
    
    weight_coef = np.zeros((block_num, 40, 40))
  
    # creating a TDCA instance
    tdca_inst = TDCA(Fs, point_delay, Nk, f_array, harmonic_num, norm)
    tdca_inst.build_reference_template(N_t)
     
    for cross_no in np.arange(block_num):                    
        # training
        train_no = np.delete(np.arange(block_num), np.where(np.arange(block_num) == cross_no))
        train_no = train_no[:train_block_num]
      
        subband_coef_1 = np.zeros((num_SN, 40, 40))
                
        # filter
        for fb_i, (B, A) in enumerate(filter_bank):
            fbfilt_traindata = np.zeros((train_block_num, 40, ssvep_channel.shape[0], N_t))
            for bl_i in range(train_block_num):
                for tr_i in range(40):
                     tmp_filt = sig.filtfilt(B, A, data[:, :, tr_i, train_no[bl_i]])
                     fbfilt_traindata[bl_i, tr_i, :, :] = tmp_filt[:, latencydelay : ]
            
            # training
            tdca_inst.train(fbfilt_traindata, Tw)

            # testing
            Precross_test = np.zeros((40, ssvep_channel.shape[0], N_t))
            for ts_i in range(40):
                tmp_filt = sig.filtfilt(B, A, data[:, : , ts_i, cross_no])
                Precross_test[ts_i, ...] = tmp_filt[:, latencydelay : ]
           
            subband_coef_1[fb_i, ...] = tdca_inst.check(Precross_test)
        weight_coef[cross_no, ...] = np.tensordot(subband_coef_1, BankWeight, axes=([0],[0]))

    # save one subject coef result
    savepath_1 = "./Coef/TDCA/BETA/" if dataset_no != 1 else "./Coef/TDCA/Benchmark/"

    # Create the directory if it doesn't exist
    os.makedirs(savepath_1, exist_ok=True)


    accfiletitle1 = savepath_1 + "trnum" + str(train_block_num) + "_Tw" + str(int(10*Tw)) +"_sub" + str(sub+1) + '_coef.npy'
    np.save(accfiletitle1, weight_coef)
    

     
    

# Generate all conditions for all subject
def generate_conditions(dataset_no, block_num, subject_num):
    if dataset_no == 1:
        arr1 = np.tile(np.arange(14).reshape(-1, 1), (1, 100)).reshape(-1, 1, order='F')
    else:
        arr1 = np.tile(np.arange(21).reshape(-1, 1), (1, 80)).reshape(-1, 1, order='F')

    train_condition = np.arange(1, block_num)
    Tw_arr = np.linspace(0.3, 1, num=8, endpoint=True)
    sub_arr = np.arange(subject_num)
    condition_list = []

    # condition list
    index = 0
    for tra in train_condition:
        for t in Tw_arr:
            for sub in sub_arr:
                arr1_element = arr1[index, 0]
                index += 1
                condition_list.append((int(dataset_no), int(tra), round(t, 1), int(sub)))

    return condition_list



if __name__ == "__main__":

    # select dataset
    dataset_no = 2
    block_num, subject_num = (6, 35) if dataset_no == 1 else (4, 70)

    #  condition parameter
    all_conditions = generate_conditions(dataset_no, block_num, subject_num)

    for i in range(len(all_conditions)):
        recognize_onebody(all_conditions[i])

