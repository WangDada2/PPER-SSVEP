# -*- coding: utf-8 -*-
"""
function： collect all acc & itr (example for eTRCA)
"""

import numpy as np
from calculate_itr import calculate_itr


if __name__ == "__main__":

    # select dataset
    dataset_no = 1

    # condition parameter
    block_num, subject_num = (6, 35) if dataset_no == 1 else (4, 70)
    num_classes = 40
    Tw_arr = np.linspace(0.3, 1, num=8, endpoint=True)

    # load coef data (example for eTRCA)
    mainpath = "./Coef/eTRCA/BETA/" if dataset_no != 1 else "./Coef/eTRCA/Benchmark/"

    condition_num = Tw_arr.size * (block_num - 1)
    all_acc = np.zeros((condition_num, subject_num, block_num))
    all_itr = np.zeros((condition_num, subject_num, block_num))

    # calculate acc for each block
    for tr_sz in np.arange(block_num - 1):
        for t_od in np.arange(Tw_arr.size):
            selection_time = Tw_arr[t_od] + 0.5
            for s_od in np.arange(subject_num):
                filetile = mainpath + "trnum" + str(tr_sz + 1) + "_Tw" + str(int(10 * Tw_arr[t_od])) + "_sub" + str(s_od+1) + '_coef.npy'

                coef = np.load(filetile)

                for cross in range(block_num):
                    output_label = np.argmax(coef[cross, ...], axis = 1)
                    pre_acc= np.mean(output_label == np.arange(40))
                    all_acc[tr_sz * 8 + t_od, s_od, cross] = pre_acc
                    all_itr[tr_sz * 8 + t_od, s_od, cross] = calculate_itr(selection_time, num_classes, pre_acc)
                    
    # calculate mean acc & ITR for each subject
    submeanacc = np.mean(all_acc, axis = 2)
    submeanitr = np.mean(all_itr, axis=2)

    print('Please refer to submeanacc and submeanitr for the results')



