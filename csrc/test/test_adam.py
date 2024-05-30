import torch
import amp_C
# 写一个代码测试multi_tensor_adam.cu代码的功能
print("开始测试multi_tensor_adam.cu代码的功能")

# 创建模拟输入数据
chunk_size = 256
noop_flag = torch.zeros(1, dtype=torch.int32)
tensor_lists = [
    [torch.randn(1024, device='cuda', dtype=torch.float32)],
    [torch.randn(1024, device='cuda', dtype=torch.float32)],
    [torch.randn(1024, device='cuda', dtype=torch.float32)],
    [torch.randn(1024, device='cuda', dtype=torch.float32)]
]
print("初始化:")
print(tensor_lists[0])
print(tensor_lists[1])
print(tensor_lists[2])
print(tensor_lists[3])
lr = 0.01
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
step = 1
mode = 0
bias_correction = 1
weight_decay = 0.01

# 调用multi_tensor_adam.cu的核心计算过程
try:
    amp_C.multi_tensor_adam(chunk_size, noop_flag, tensor_lists, lr, beta1, beta2, epsilon, step, mode, bias_correction, weight_decay)
    print("multi_tensor_adam.cu代码功能测试完成.")
except Exception as e:
    print(f"测试multi_tensor_adam.cu代码功能时出错: {e}")

# 输出计算结果
print("输出计算结果:")
for tensor in tensor_lists[3]:
    print(tensor)