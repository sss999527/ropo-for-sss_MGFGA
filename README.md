# Enhancing Federated Domain Adaptation with Multi-Granular Fine-Grained Alignment
This is the dedicated repository for the ICASSP submission paper titled "Enhancing Federated Domain Adaptation via Multi-granular Fine-grained Alignment," used for storing the code corresponding to the paper.

## Abstract
Traditional unsupervised multi-source domain adaptation usually assumes that all source domain data can be utilized during training. Unfortunately, due to practical concerns such as privacy, data storage, and computational costs, data from different source domains are often isolated from each other. To address this issue, we propose a federated domain adaptation framework based on fine-grained alignment. This method achieves domain adaptation at the model level through iterative training of source and target domains, thereby avoiding the direct use of source domain data. Specifically, our approach employs specialized techniques at various stages—model construction, pseudo-label generation, and model training—to handle fine-grained features that are often overlooked. This enables the model to effectively remove irrelevant information and learn more discriminative features, thus narrowing the distribution gap between domains. Extensive experimental results demonstrate the effectiveness of our proposed method across multiple datasets.

## Method
![F1](https://github.com/sss999527/ropo-for-sss_MGFGA/blob/main/img/F1.pdf)
