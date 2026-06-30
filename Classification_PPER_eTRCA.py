# -*- coding: utf-8 -*-
"""
@author: Lijie Wang (lijiewang@zhejianglab.com)

function: PPER-eTRCA
"""


import numpy as np
from scipy import signal as sig
import scipy.io as scio
import math
from PPER import Get_PPESeq, DataAug
import os


# class PPER-eTRCA
class PPER_TRCA(object):
    
    def __init__(self, Fs, norm):
        self.Fs = Fs
        self.norm = norm
        
    
    def train(self, traindata, Tw, pperseq, B, A):
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
        
        if block_num > 1:
            
            # original TRCA
            W_or = np.zeros((channel_num, class_num))
            template_or = np.zeros((class_num, channel_num, N))
            
            for c_i in range(class_num):
                X = traindata[:, c_i, ...]
                S = np.zeros((X.shape[1], X.shape[1]))
                for trial_i in range(X.shape[0]):
                    for trial_j in range(X.shape[0]):
                        S = S + np.dot(X[trial_i,...], X[trial_j,...].T)
                        
                X1 = X[0,...]
                for tri_i in range(1, X.shape[0]):
                    X1 = np.hstack((X1,X[tri_i,...]))
                X1 = X1 - np.mean(X1, axis=1).reshape(X.shape[1],1)
                Q = np.dot(X1,X1.T)
                
                # Compute eigenvalues and vectors
                lambdas, W = np.linalg.eig(np.linalg.solve(Q,S))

                # Select the eigenvector corresponding to the biggest eigenvalue
                W_or[:,c_i] = W[:, np.argmax(lambdas)]
                template_or[c_i, ...] = np.mean(X, axis = 0)
            
            
            # PPER
            W_pper = np.zeros((channel_num, class_num))
            template_pper = np.zeros((class_num, channel_num, N))
            for c_i in range(class_num):
                # instanition for PPER transformation class
                dataaug = DataAug(pperseq[c_i, ...])
                pure_trans_data = []
                for b_j in range(block_num):
                    X = traindata[b_j, c_i, ...]
                    recondata = dataaug.convert(X)
                    # filt
                    for tr_i in np.arange(1, recondata.shape[0]):
                        recondata[tr_i, ...] = sig.filtfilt(B, A, recondata[tr_i, ...])
                    
                    pure_trans_data.append(recondata)                          # [1:, :, :]
                X2 = np.concatenate(pure_trans_data, axis = 0)
                
                # TRCA analysis of PPER data
                S = np.zeros((X2.shape[1], X2.shape[1]))
                for trial_i in range(X2.shape[0]):
                    for trial_j in range(X2.shape[0]):
                        S = S + np.dot(X2[trial_i,...], X2[trial_j,...].T)
                        
                X1 = X2[0,...]
                for tri_i in range(1, X2.shape[0]):
                    X1 = np.hstack((X1,X2[tri_i,...]))
                X1 = X1 - np.mean(X1, axis=1).reshape(X2.shape[1],1)
                Q = np.dot(X1, X1.T)
                
                # Compute eigenvalues and vectors
                lambdas, W = np.linalg.eig(np.linalg.solve(Q,S))
                
                # Select the eigenvector corresponding to the biggest eigenvalue
                W_pper[:,c_i] = W[:, np.argmax(lambdas)]
                template_pper[c_i, ...] = np.mean(X2, axis = 0)
            
            # merge
            self.emerge_W = np.concatenate((W_or, W_pper), axis = 1)
            self.emerge_template = np.concatenate((template_or, template_pper), axis = 2)
            
        else:
            
            # PPER
            W_pper = np.zeros((channel_num, class_num))
            template_pper = np.zeros((class_num, channel_num, N))
            for c_i in range(class_num):
                # instanition for PPER transformation class
                dataaug = DataAug(pperseq[c_i, ...])
                pure_trans_data = []
                for b_j in range(block_num):
                    X = traindata[b_j, c_i, ...]
                    recondata = dataaug.convert(X)
                    # filt
                    for tr_i in np.arange(1, recondata.shape[0]):
                        recondata[tr_i, ...] = sig.filtfilt(B, A, recondata[tr_i, ...])
                    
                    pure_trans_data.append(recondata)
                X2 = np.concatenate(pure_trans_data, axis = 0)
                
                # TRCA analysis of PPER data
                S = np.zeros((X2.shape[1], X2.shape[1]))
                for trial_i in range(X2.shape[0]):
                    for trial_j in range(X2.shape[0]):
                        S = S + np.dot(X2[trial_i,...], X2[trial_j,...].T)
                        
                X1 = X2[0,...]
                for tri_i in range(1, X2.shape[0]):
                    X1 = np.hstack((X1,X2[tri_i,...]))
                X1 = X1 - np.mean(X1, axis=1).reshape(X2.shape[1],1)
                Q = np.dot(X1, X1.T)
                
                # Compute eigenvalues and vectors
                lambdas, W = np.linalg.eig(np.linalg.solve(Q,S))
                
                # Select the eigenvector corresponding to the biggest eigenvalue
                W_pper[:,c_i] = W[:, np.argmax(lambdas)]
                template_pper[c_i, ...] = np.mean(X2, axis = 0)
            
            # merge
            self.emerge_W = W_pper
            self.emerge_template = template_pper
        
                
                
    
    def check(self, testdata, pperseq, B, A, trnum):
        
        '''
        Parameters
        ----------
        testdata: three-dim numpy array
                  classnum * channle_num * sample_num
        '''
        class_num, channel_num, N = testdata.shape
        coef_1 = np.zeros((class_num, class_num))

        for ts_i in range(class_num):
            pre_test = testdata[ts_i, ...]

            for tr_i in range(class_num): 
                # instantiation for PPER augmentation class
                dataaug = DataAug(pperseq[tr_i, ...])
                
                # PPER
                recontest = dataaug.convert(pre_test)
                for j in np.arange(1, recontest.shape[0]):
                    recontest[j, ...] = sig.filtfilt(B, A, recontest[j, ...])

                temp2coef = 0
                if trnum > 1:
                    # combine test signal and transformation data
                    for j in np.arange(recontest.shape[0]):
                        emerge_test = np.concatenate((pre_test, recontest[j, ...] ), axis = 1)
                        map_test = np.dot(self.emerge_W.T, emerge_test).ravel()
                        map_template = np.dot(self.emerge_W.T, self.emerge_template[tr_i, ...]).ravel()
                        temp2coef += np.corrcoef(map_template, map_test)[0, 1]
                else:
                    for j in np.arange(recontest.shape[0]):
                        emerge_test = recontest[j, ...]
                        map_test = np.dot(self.emerge_W.T, emerge_test).ravel()
                        map_template = np.dot(self.emerge_W.T, self.emerge_template[tr_i, ...]).ravel()
                        temp2coef += np.corrcoef(map_template, map_test)[0, 1]
                        
                coef_1[ts_i, tr_i] = temp2coef / recontest.shape[0]
      
        return coef_1
    

# Encapsulate a process for one subject & one condition
def recognize_onebody(onecon_tup):

    # print process information
    print(onecon_tup)

    # get parameters of the present process
    dataset_no, train_block_num, Tw, sub = onecon_tup
    
    # eeg data path
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

    
    # filter bank parameter
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

    # genarate PPER sequnce
    trsod = 4    # The first sequence of sequence list is the original order sequence
    get_ppeseq = Get_PPESeq(N_t, Fs, trsod)
    seqset_t = []
    for fr in f_array:
        seqset_t.append(get_ppeseq.produce(fr))
    seqset_t = np.stack(seqset_t)


    # cut data
    data = data[ssvep_channel, ...]
    data = data[:, round(0.5 * Fs): round(0.5 * Fs) + latencydelay + N_t, :, :]


    weight_coef = np.zeros((block_num, 40, 40))

  
    # instantiate the object of PPER-eTRCA class
    ppertrca_inst = PPER_TRCA(Fs, norm)
     
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
            
            ppertrca_inst.train(fbfilt_traindata, Tw, seqset_t, B, A)
            
            Precross_test = np.zeros((40, ssvep_channel.shape[0], N_t))
            for ts_i in range(40):
                tmp_filt = sig.filtfilt(B, A, data[:, : , ts_i, cross_no])
                # testing
                Precross_test[ts_i, ...] = tmp_filt[:, latencydelay : ]
           
            subband_coef_1[fb_i, ...] = ppertrca_inst.check(Precross_test, seqset_t, B, A, train_block_num)

        weight_coef[cross_no, ...] = np.tensordot(subband_coef_1, BankWeight, axes=([0],[0]))

    # save one subject coef result
    savepath_1 = "./Coef/PPER_eTRCA/BETA/" if dataset_no != 1 else "./Coef/PPER_eTRCA/Benchmark/"

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
    dataset_no = 1
    block_num, subject_num = (6, 35) if dataset_no == 1 else (4, 70)

    # condition parameter
    all_conditions = generate_conditions(dataset_no, block_num, subject_num)

    for i in range(len(all_conditions)):
        recognize_onebody(all_conditions[i])


        