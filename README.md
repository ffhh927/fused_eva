# Fused_Eva : 对二阶优化函数Eva的一种加速算法

The Eva code was originally forked from Lin Zhang's [kfac-pytorch](https://github.com/lzhangbv/kfac_pytorch). 

## Install

### Requirements

PyTorch and Horovod are required to use K-FAC.

This code is validated to run with PyTorch-1.10.0, Horovod-0.21.0, CUDA-10.2, cuDNN-7.6, NCCL-2.6.4, and cutlass-3.5.0

### Installation

```
$ git clone https://github.com/ffhh927/fused_eva.git
$ cd fused_eva
$ pip install -r requirements.txt
$ HOROVOD_GPU_OPERATIONS=NCCL pip install horovod
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

