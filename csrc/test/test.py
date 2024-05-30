import torch
import amp_C
def main():
    # 模拟输入数据
    chunk_size = 256
    noop_flag = torch.zeros(1, dtype=torch.int32)
    tensor_lists = [
        [torch.randn(1024, device='cuda', dtype=torch.float32)],
        [torch.randn(1024, device='cuda', dtype=torch.float32)],
        [torch.randn(1024, device='cuda', dtype=torch.float32)]
    ]
    print("初始化:")
    print(tensor_lists[0])
    print(tensor_lists[1])
    print(tensor_lists[2])
    lr = 0.01
    damping = 0.1
    kl_clip = 0.5

    # 调用EVA算法的核心计算过程
    try:
        amp_C.multi_tensor_eva(chunk_size, noop_flag, tensor_lists, lr, damping, kl_clip)
        print("EVA computation completed successfully.")
    except Exception as e:
        print(f"Error during EVA computation: {e}")

    # 输出计算结果
    print("输出计算结果:")

    print(tensor_lists[1])

    expected_result = (tensor_lists[0][0] + tensor_lists[2][0]) / damping
    
    print(expected_result)
    # 对比计算结果和正确结果
    print("对比计算结果和正确结果:")
    
    eva = tensor_lists[1][0].tolist()
    expect = expected_result.tolist()
    for j in range(len(eva)):
        diff = eva[j] - expect[j]
#         print(diff)
#         max_diff = torch.max(torch.abs(diff))
#        print(f"Tensor 1, {j} 最大差异: {max_diff.item()}")

if __name__ == "__main__":
    main()
