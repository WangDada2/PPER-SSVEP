# PPER-SSVEP

Official implementation of the paper

**Sample-wise Temporal Reconstruction Enables Few-shot Brain–Computer Interfacing**

***

## Overview

This repository provides the official implementation of the **Point-Position Equivalent Reconstruction (PPER)** framework for few-shot steady-state visual evoked potential (SSVEP) decoding.

PPER performs **sample-wise temporal reconstruction** through phase-aligned stochastic augmentation to generate physiologically consistent neural representations. The framework can be seamlessly integrated with existing SSVEP decoding algorithms without modifying their original decoding pipelines, thereby substantially improving decoding performance under extremely limited calibration data.

The current implementation includes the integration of PPER with two state-of-the-art SSVEP decoders:

*   **eTRCA**

*   **TDCA**

***

## Highlights

*   Sample-wise temporal reconstruction for SSVEP decoding

*   Few-shot calibration with minimal training data

*   Plug-and-play enhancement framework

*   Compatible with eTRCA and TDCA

*   Robust under short stimulus durations

*   Robust to noisy whole-brain electrode configurations

*   Evaluated on Benchmark and BETA public datasets

***

## Framework

The overall framework of the proposed PPER-enhanced SSVEP decoding pipeline is illustrated below.

![](README_md_files/5b7ed340-7450-11f1-ac95-af70a4181d66.jpeg?v=1\&type=image)

***

## Installation

Create a Python environment (Python ≥ 3.9).

```bash
conda create -n pper_env python=3.9
conda activate pper_env

pip install -r requirements.txt
```

***

## Dataset

Experiments were conducted on two publicly available SSVEP datasets.

*   **Benchmark dataset** (Wang et al., 2016)

*   **BETA dataset** (Liu et al., 2020)

The datasets can be downloaded from

<https://bci.med.tsinghua.edu.cn/download.html>

Please modify the dataset path in the corresponding Python scripts if necessary.

***

## Usage

### Baseline algorithms

Run the original SOTA decoding algorithms.

```bash
python Classification_eTRCA.py

python Classification_TDCA.py
```

***

### PPER-enhanced algorithms

Run the proposed PPER-enhanced decoding framework.

```bash
python Classification_PPER_eTRCA.py

python Classification_PPER_TDCA.py
```

***

### Collect decoding results

Aggregate decoding accuracy and Information Transfer Rate (ITR).

```bash
python Collect_acc_itr.py
```

***

### Experimental Settings

Several variables can be modified in the main function.

| Variable         | Description                                                                          |
| ---------------- | ------------------------------------------------------------------------------------ |
| `dataset_no`     | `1` for Benchmark; other integers indicate BETA                                      |
| `all_conditions` | Experimental conditions including training size, stimulus duration and subject index |

Recognition coefficients will be automatically saved to

```text
Coef/
```

***

## Repository Structure

```text
PPER-SSVEP/

├── PPER.py
│   Core implementation of Point-Position Equivalent Reconstruction (PPER)
│
├── Classification_eTRCA.py
│   Baseline eTRCA decoding pipeline
│
├── Classification_TDCA.py
│   Baseline TDCA decoding pipeline
│
├── Classification_PPER_eTRCA.py
│   PPER-enhanced eTRCA decoding pipeline
│
├── Classification_PPER_TDCA.py
│   PPER-enhanced TDCA decoding pipeline
│
├── calculate_itr.py
│   Information Transfer Rate (ITR) calculation
│
├── Collect_acc_itr.py
│   Aggregate decoding accuracy and ITR across all experimental conditions
│
├── requirements.txt
│
└── README.md
```

***

## Why PPER?

Compared with conventional SSVEP decoding methods, PPER offers several advantages.

*   Sample-wise temporal reconstruction instead of frequency-domain decomposition

*   Significantly reduced calibration requirements

*   Improved discriminability of neural representations

*   Robust under short stimulus durations

*   Robust to noisy whole-brain electrode configurations

*   Plug-and-play integration with existing decoding algorithms

***

## Citation

If you find this repository useful in your research, please consider citing our paper.

```bibtex
Citation will be updated upon publication.
```

***

## License

This project is released under the MIT License.

***

## Contact

For questions, suggestions, or collaborations, please contact

**Lijie Wang**

<lijiewang@zhejianglab.org>

***

## Acknowledgements

This work is built upon publicly available SSVEP datasets and existing decoding algorithms.

*   Wang et al., Benchmark Dataset (2016)

*   Liu et al., BETA Dataset (2020)

*   eTRCA

*   TDCA

