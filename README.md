# Fused_Eva : An acceleration algorithm for the second-order optimization function Eva

The Eva code was originally forked from Lin Zhang's [kfac-pytorch](https://github.com/lzhangbv/kfac_pytorch). 

## Install

### Requirements

PyTorch and Horovod are required to use K-FAC.

This code is validated to run with PyTorch-1.10.0, Horovod-0.21.0, CUDA-10.2, cuDNN-7.6, and NCCL-2.6.4

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

(Unit seconds, average time spent, EVA / FusedEVA)

| batch | forward         | backward        | optimize        | Optimize acceleration | Overall acceleration |
| ----- | --------------- | --------------- | --------------- | --------------------- | -------------------- |
| 1     | 1.2404 / 1.2504 | 0.8890 / 0.8956 | 2.1501 / 1.7675 | 1.2164                | 1.0935               |
| 2     | 1.3266 / 1.2858 | 0.9497 / 0.9535 | 2.1778 / 1.7634 | 1.2349                | 1.1127               |
| 4     | 1.2655 / 1.2631 | 0.9255 / 0.9357 | 2.1189 / 1.7654 | 1.2002                | 1.0872               |
| 8     | 1.1872 / 1.1872 | 0.9101 / 0.9251 | 2.0657 / 1.7451 | 1.1837                | 1.0792               |
| 16    | 1.2826 / 1.2708 | 1.0011 / 0.9951 | 2.1036 / 1.7701 | 1.1884                | 1.0870               |
| 32    | 1.1585 / 1.1254 | 1.0039 / 0.9878 | 2.0468 / 1.7476 | 1.1712                | 1.0902               |
| 64    | 1.1555 / 1.1553 | 1.0030 / 0.9856 | 2.0498 / 1.7403 | 1.1778                | 1.0842               |
| 128   | 1.2845 / 1.2631 | 1.2217 / 1.1967 | 2.1695 / 1.8497 | 1.1728                | 1.0849               |
| 256   | 1.3194 / 1.2981 | 1.6161 / 1.5925 | 2.1970 / 1.8952 | 1.1592                | 1.0724               |

