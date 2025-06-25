# A Min-max Optimization Framework for Sparse Multi-task Deep Neural Networks
This repository contains the source code of Multi-Task Learning with Min-max Optimization and baselines from the following papers:
1) [A Min-max Optimization Framework for Multi-task Deep Neural Network Compression](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10557958) (In proceedings of ISCAS 2024);
2) A Min-max Optimization Framework for Sparse Multi-task Deep Neural Networks. (Neurocomputing accepted)

All models were written in `PyTorch`. 

## Experiments
### Datasets
Please download the pre-processed `NYUv2` dataset [here](https://www.dropbox.com/scl/fo/p7n54hqfpfyc6fe6n62qk/AKVb28ZmgDiGdRMNkX5WJvo?rlkey=hcf31bdrezqjih36oi8usjait&e=1&dl=0) which is evaluated in the papers. Moreover, if you are interested, the raw 13-class NYUv2 dataset can be downloaded [here](https://github.com/ankurhanda/nyuv2-meta-data) with segmentation labels defined [here](https://github.com/ankurhanda/SceneNetv1.0/). 
### Weight Pruning for Model Compression
The folder `prune_apgda` provides the code of our proposed network using weight pruning strategy to compress the model in 40x and 60x along with all the baselines on `NYUv2` dataset presented in the paper 1. 

### Dynamic Sparse Training and more comparable baselines

**To be further contributed...**

## Acknowledgements
We would sincerely thank Shikun Liu and his group for the Multi-task Attention Network (MTAN) design [Project Page](https://github.com/lorenmt/mtan).

## Contact
If you have any questions, please contact Jiacheng Guo at `j.guo58@vikes.csuohio.edu`.
