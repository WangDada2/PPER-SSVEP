# -*- coding: utf-8 -*-
"""
@author: Lijie Wang (lijiewang@zhejianglab.com)

function: Point-Position Equivalent Reconstruction

Date: 23.10.24
"""

import numpy as np
import math 
import random

class Get_PPESeq(object):
    
    def __init__(self,sn,fs,trsod):
        '''
        sn: sampling point number of one channel
        fs: sampling rate
        trsod: agumentation sequence number
        '''
        self.SamNum = int(sn)
        self.Fs = fs
        self.trsod = trsod

    def produce(self, fr):
        # fr: stimulus frequency
        OrSeq = np.arange(self.SamNum)
        CySam = math.floor(self.Fs / fr)
        CyNum = math.ceil(self.SamNum * fr / self.Fs)
        
        # Generate original sub vector
        OrSubVec = [[] for _ in range(CySam)]
        for i in np.arange(CySam):
            for j in np.arange(CyNum):
                if (round(j * self.Fs / fr)+i) < self.SamNum:
                    OrSubVec[i].append(round(j * self.Fs / fr)+i)
                else:
                    break
   
        # Reserve points that do not participate in the exchange 
        OrList = OrSeq.tolist()
        SetSubVec = set([i for item in OrSubVec for i in item])
        ReservePos = list(set(OrList).difference(SetSubVec))
        
        # Random disturb
        EquseqSet = np.array(OrSeq).reshape(1,len(OrSeq))
        while (EquseqSet.shape[0] < self.trsod):
            NewSubVec = []
            for i in np.arange(CySam):
                a = OrSubVec[i]
                random.shuffle(a)
                NewSubVec.append(a)
                
            # Generate a new complete sequence
            NewIncomList = []
            for i in np.arange(CyNum):
                for j in  np.arange(CySam):
                    if (len(NewSubVec[j])-1) < i:
                        break
                    else:
                        NewIncomList.append(NewSubVec[j][i])      
            # Merging sequence
            for x in ReservePos:
                NewIncomList.insert(x, x)
            
            # calculate Levenshtein distance
            Newseq = np.array(NewIncomList).reshape(1,len(NewIncomList))
            EquseqSet = np.vstack((EquseqSet,Newseq))

        # output result
        return EquseqSet
    

class DataAug(object):
    
    def __init__(self, augseq):
        '''
        augseq: np.array[m,n] one PPE seauence matrix for one target frequency
        
        '''
        self.augseq = augseq
        
    def convert(self, data):
        '''
        Data: np.array[channel, sample] one raw data matrix
        '''

        NewData = np.empty((self.augseq.shape[0], data.shape[0], data.shape[1]))

        # We use the same augseq index for each channel
        for idx, seq in enumerate(self.augseq):
            NewData[idx] = data[:, seq]
        
        return NewData


    
   
            
                        
                        
                        
                    
                    
                
            
        
        
        
        
                    
        
            
            
        
       
        
       
        
       
        
       
        
       
        
       
        
       
        
       
        
       
        
       
        
       
        
       
        
       
        
       
        
       
        
       
        
       
        
       
        
    

                  