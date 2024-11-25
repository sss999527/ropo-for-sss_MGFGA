# Enhancing Federated Domain Adaptation with Multi-Granular Fine-Grained Alignment
This is the dedicated repository for the ICASSP submission paper titled "Enhancing Federated Domain Adaptation via Multi-granular Fine-grained Alignment," used for storing the code corresponding to the paper.

## Abstract
Traditional unsupervised multi-source domain adaptation usually assumes that all source domain data can be utilized during training. Unfortunately, due to practical concerns such as privacy, data storage, and computational costs, data from different source domains are often isolated from each other. To address this issue, we propose a federated domain adaptation framework based on fine-grained alignment. This method achieves domain adaptation at the model level through iterative training of source and target domains, thereby avoiding the direct use of source domain data. Specifically, our approach employs specialized techniques at various stages—model construction, pseudo-label generation, and model training—to handle fine-grained features that are often overlooked. This enables the model to effectively remove irrelevant information and learn more discriminative features, thus narrowing the distribution gap between domains. Extensive experimental results demonstrate the effectiveness of our proposed method across multiple datasets.

## Method
![F1](https://github.com/sss999527/ropo-for-sss_MGFGA/blob/main/images/F1.png)

## Setup
### Install Package Dependencies

```
Python Environment: >= 3.6
torch >= 1.2.0
torchvision >= 0.4.0
tensorbard >= 2.0.0
numpy
yaml
```

### Install Datasets
We need users to download the DigitFive, Office-Caltech, Office31, DomainNet, or MiniDomainNet, and Office-Home datasets for the MSFDA experiments. They should declare a base path to store the datasets with the following directory structure:

```
base_path
│       
└───dataset
│   │   DigitFive
│       │   mnist_data.mat
│       │   mnistm_with_label.mat
|       |   svhn_test_32x32.mat
|       |   svhn_train_32x32.mat  
│       │   ...
│   │   DomainNet
│       │   Clipart
│       │   Infograph
│       │   ...
│   │   MiniDomainNet
│       │   ...
│   │   OfficeCaltech10
│       │   ...
|   |   Office31
|       |   ...
└───trained_model
│   │	parmater
│   │	runs
...
```

Note that the `dataset` folder is different from the `datasets` folder, one for the dataset and the other for the network model.

## Federated Domain Adaptation
The configuration files are located in the `./config` directory, where you will find four `.yaml`-formatted config files. To execute unsupervised multi-source decentralized domain adaptation on a particular dataset, such as painting in DomainNet, utilize the following commands.

```
python main.py --config MiniDomainNet.yaml --target-domain painting -bp "$(pwd)"
```
You can download the dataset from [DomainNet](https://ai.bu.edu/M3SDA/).

The training parameters of some datasets can be downloaded from Baidu Netdisk:
https://pan.baidu.com/s/1wgzt6fGnqlkjJIv-Jn5MAg?pwd=3why

The training results for four main datasets are as follows:
  * Results on Office-Caltech10, Digit-5 dataset and DomainNet.
![T1](https://github.com/sss999527/ropo-for-sss_MGFGA/blob/main/images/t1.png)
  * Results on Office-Home dataset.
![T2](https://github.com/sss999527/ropo-for-sss_MGFGA/blob/main/images/t2.png)

### t-SNE Feature Visualization
The following four figures show the t-SNE feature visualization results before and after domain adaptation of the model on MNIST and MNIST-M datasets in turn, where figures 1, 3 show the results before domain adaptation and figures 2, 4 show the results after domain adaptation.

  * t-SNE Feature Visualization result on MNIST.
![F5](https://github.com/sss999527/ropo-for-sss_MGFGA/blob/main/images/F9.png)
  * t-SNE Feature Visualization result on MNIST-M.
![F7](https://github.com/sss999527/ropo-for-sss_MGFGA/blob/main/images/F10.png)


## Miscellaneous
We can't put all the code into the repository at the moment as the article is not yet confirmed for acceptance, and we will update the detailed code once the paper is accepted.




