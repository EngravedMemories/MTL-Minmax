# A Min-max Optimization Framework for Sparse Multi-task Deep Neural Networks
This repository contains the source code of Multi-Task Learning with Min-max Optimization and baselines from the following papers:
1) [A Min-max Optimization Framework for Multi-task Deep Neural Network Compression](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10557958) (Conference Version, In proceedings of ISCAS 2024);
2) A Min-max Optimization Framework for Sparse Multi-task Deep Neural Networks (Extended Journal Version, Neurocomputing accepted).

All models were written in `PyTorch`. 

## Datasets
We implemented all weighting baselines presented in the paper for computer vision tasks: Dense Prediction Tasks (for NYUv2) and Multi-domain Classification Tasks (for CIFAR-100).

- NYUv2 [3 Tasks]: 13 Class Segmentation + Depth Estimation + Surface Normal Prediction. [288 x 384] Resolution.
- CIFAR-100 [20 Tasks]: 20 Class Object Classification.

Please download the pre-processed `NYUv2` dataset [here](https://www.dropbox.com/scl/fo/p7n54hqfpfyc6fe6n62qk/AKVb28ZmgDiGdRMNkX5WJvo?rlkey=hcf31bdrezqjih36oi8usjait&e=1&dl=0) which is evaluated in the papers. (Moreover, if you are more interested, the raw 13-class NYUv2 dataset can be downloaded [here](https://github.com/ankurhanda/nyuv2-meta-data) with segmentation labels defined [here](https://github.com/ankurhanda/SceneNetv1.0/). )

## Experiments

### Weight Pruning for Model Compression
The folder `prune_apgda` provides the code of our proposed network using weight pruning strategy to compress the model in 40x and 60x along with all the baselines on `NYUv2` dataset presented in paper 1. The basic network structure is established based on [MTAN](https://github.com/lorenmt/mtan). 
We propose a novel weight pruning method to compress the model, and a Min-Max optimization method including APGDA algorithm to further inprove the model performance.

### Dynamic Sparse Training and More Comparable Baselines
The root folder provides the code of our proposed network using dynamic sparse training together with weight pruning for comparison in 60x and 100x along with all the baselines on `NYUv2` and `CIFAR100` datasets presented in paper 2. The basic network structure is established based on [Auto-lambda](https://github.com/lorenmt/auto-lambda).

**Weighting-based settings:**
- Equal: All task weights are 1. `--weight equal`
- Uncertainty: [Uncertainty](https://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf). `--weight uncert`
- Dynamic Weighting Average: [DWA](https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_End-To-End_Multi-Task_Learning_With_Attention_CVPR_2019_paper.pdf). `--weight dwa`
- Min-max: Our proposed method. `--weight minmax`
  
- For DiSparse and AdapMTL methods, please refer to [this link](https://github.com/MitchellX/AdapMTL). You may also use the `environment.yml` document to setup the environment directly.

### Parameter Settings

- The model compression project is located in `prune_apgda` folder. Please use `python trainer_apgda.py` to run the project.

  Please first use `--stage pretrain` to save the dense model. After using `--stage rew` to implement reweighting, finally use `--stage retrain` to retrain the compressed model.
  
  The pruning rate 40x is equivalent to `--prune-ratios 0.975`. The pruning rate 60x is equivalent to `--prune-ratios 0.983`.
  
  Min-max hyperparameter settings: beta = 50, gamma = 5 (pre-settled).

- The dynamic sparse training project is located in the root folder. Please use `python trainer_nyuv2.py` or `python trainer_cifar.py` to run the corresponding project.
  
  Please first use `--stage pretrain` to save the dense model. After using `--stage rew` to implement reweighting, finally use `--stage retrain` to retrain the compressed model.
  
  Directly use `--stage retrain` to implement Dynamic Sparse Training (without loading pretrained model).
  
  The pruning rate 60x is equivalent to `--prune-ratios 0.983`. In this case, the `layer_prune_ratios` and `layer_grow_ratios` should be set to 0.0051.
  
  The pruning rate 100x is equivalent to `--prune-ratios 0.99`. In this case, the `layer_prune_ratios` and `layer_grow_ratios` should be set to 0.003.
  
  Min-max hyperparameter settings: beta = 10, gamma = 5 (pre-settled).

## Acknowledgements
We would sincerely thank Dr. Shikun Liu and his group for the Multi-task Attention Network (MTAN) design. The following links show their [MTAN Project Page](https://github.com/lorenmt/mtan) and [Auto-lambda Project Page](https://github.com/lorenmt/auto-lambda).

## Citations
If you find this code/work to be useful in your own research, please consider citing the following.
- Conference version:
```bash
@inproceedings{guo2024min,
  title={A Min-Max Optimization Framework for Multi-task Deep Neural Network Compression},
  author={Guo, Jiacheng and Sun, Huiming and Qin, Minghai and Yu, Hongkai and Zhang, Tianyun},
  booktitle={2024 IEEE International Symposium on Circuits and Systems (ISCAS)},
  pages={1--5},
  year={2024},
  organization={IEEE}
}
```
- Journal version: To be published!

## Contact
If you have any questions, please contact Jiacheng Guo at `j.guo58@vikes.csuohio.edu`.
