# -*- coding: utf-8 -*-
"""
function: calculate Information Transfer Rate (ITR)
"""

import math

def calculate_itr(selection_time, num_classes, accuracy):
    """

    Parameters
        ----------
        selection_time: gaze shifting time + Tw
        num_classes: stimulus number
        accuracy: classification accuracy
    """
    if accuracy == 1:
        # 当准确率为100%时，避免数学错误
        accuracy = 1 - 1e-9
    if accuracy != 0:
        term1 = math.log2(num_classes)
        term2 = accuracy * math.log2(accuracy)
        term3 = (1 - accuracy) * math.log2((1 - accuracy) / (num_classes - 1))

        itr = (60 / selection_time) * (term1 + term2 + term3)
    else:
        itr = 0

    return itr

