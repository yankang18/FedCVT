


If you think our work is helpful and used our code in your work, please cite our paper:
```
@article{10.1145/3510031,
author = {Kang, Yan and Liu, Yang and Liang, Xinle},
title = {FedCVT: Semi-supervised Vertical Federated Learning with Cross-view Training},
year = {2022},
issue_date = {August 2022},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {13},
number = {4},
issn = {2157-6904},
url = {https://doi.org/10.1145/3510031},
doi = {10.1145/3510031},
abstract = {Federated learning allows multiple parties to build machine learning models collaboratively without exposing data. In particular, vertical federated learning (VFL) enables participating parties to build a joint machine learning model based upon distributed features of aligned samples. However, VFL requires all parties to share a sufficient amount of aligned samples. In reality, the set of aligned samples may be small, leaving the majority of the non-aligned data unused. In this article, we propose Federated Cross-view Training (FedCVT), a semi-supervised learning approach that improves the performance of the VFL model with limited aligned samples. More specifically, FedCVT estimates representations for missing features, predicts pseudo-labels for unlabeled samples to expand the training set, and trains three classifiers jointly based upon different views of the expanded training set to improve the VFL model’s performance. FedCVT does not require parties to share their original data and model parameters, thus preserving data privacy. We conduct experiments on NUS-WIDE, Vehicle, and CIFAR10 datasets. The experimental results demonstrate that FedCVT significantly outperforms vanilla VFL that only utilizes aligned samples. Finally, we perform ablation studies to investigate the contribution of each component of FedCVT to the performance of FedCVT.},
journal = {ACM Trans. Intell. Syst. Technol.},
month = {may},
articleno = {64},
numpages = {16},
keywords = {Vertical federated learning, semi-supervised learning, cross-view training}
}
```
