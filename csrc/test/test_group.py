import torch
import amp_C
import numpy as np
import time
import os

# 设置 CUDA_LAUNCH_BLOCKING 环境变量为 1
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
def main():
    # 设置参数
    m, n, k = 1, 1, 6
    batchCount = 2
    print("test")
    # 生成测试数据
    A = [torch.tensor([[0.0,2,4,1,3,5]],dtype=torch.float32).cuda(),torch.tensor([[6,2,3,4,1,8]],dtype=torch.float32).cuda()]
    B = [A[0].T,A[1].T]
    C = [torch.tensor([[0.0]],dtype=torch.float32).cuda(),torch.tensor([[0.0]],dtype=torch.float32).cuda()]
    # 调用C++函数
    amp_C.multi_group(A, B, C)
    time.sleep(5)
    # 打印结果
    print("Result for batch")
    print(C)



if __name__ == "__main__":
    main()
