# Fused_Eva : An acceleration algorithm for the second-order optimization function Eva

The Eva code was originally forked from Lin Zhang's [kfac-pytorch](https://github.com/lzhangbv/kfac_pytorch). 

## Install

### Requirements

PyTorch and Horovod are required to use K-FAC.

This code is validated to run with PyTorch-1.10.0, Horovod-0.21.0, CUDA-10.2, cuDNN-7.6, NCCL-2.6.4, and nvidia-cutlass-3.5

### Installation

```
$ git clone https://github.com/ffhh927/fused_eva.git
$ cd fused_eva
$ pip install -r requirements.txt
$ HOROVOD_GPU_OPERATIONS=NCCL pip install horovod
$ cd csrc
$ python setup.py install
```

If pip installation failed, please try to upgrade pip via `pip install --upgrade pip`. If Horovod installation with NCCL failed, please check the installation [guide](https://horovod.readthedocs.io/en/stable/install_include.html). 

## Usage

The Distributed Eva can be easily added to exisiting training scripts that use PyTorch's Distributed Data Parallelism.

```Python
from fused import FusedEVA
... 
model = torch.nn.parallel.DistributedDataParallel(...)
optimizer = optim.SGD(model.parameters(), ...)
preconditioner = FusedEVA(model, ...)
... 
for i, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    preconditioner.step()
    optimizer.step()
...
```

## Comparison of results

The model is resnet18，and the training set is CIFAR100. Trained for 10000 iterations.

(Unit milliseconds / iteration, EVA / FusedEVA)

| batch | forward         | backward          | optimize        | Optimize acceleration |
| ----- | --------------- | ----------------- | --------------- | --------------------- |
| 1     | 6.4200 / 6.3383 | 5.0072 / 4.8228   | 7.1700 / 5.4904 | 1.3054                |
| 2     | 6.5336 / 6.4038 | 5.6027 / 5.5961   | 7.1241 / 5.4604 | 1.3046                |
| 4     | 6.6091 / 6.5987 | 6.2409 / 6.2240   | 7.0037 / 5.4004 | 1.2968                |
| 8     | 6.6967 / 6.6218 | 5.8719 / 5.8318   | 6.8439 / 5.2842 | 1.2952                |
| 16    | 6.1667 / 6.7273 | 6.4003 / 6.3780   | 6.6541 / 5.3138 | 1.2522                |
| 32    | 6.9099 / 6.8073 | 7.2159 / 7.0960   | 6.6377 / 5.3203 | 1.2476                |
| 64    | 7.1130 / 7.0675 | 8.0918 / 8.1358   | 6.5649 / 5.2509 | 1.2503                |
| 128   | 7.0614 / 6.9877 | 11.7406 / 11.7102 | 6.4936 / 5.1126 | 1.2701                |
| 256   | 9.7958 / 9.2351 | 19.6936 / 19.9653 | 6.4830 / 5.1307 | 1.2636                |

