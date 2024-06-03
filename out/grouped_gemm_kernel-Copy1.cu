// This file was automatically generated by the CUTLASS 3.5.0 Python interface (https://github.com/nvidia/cutlass/python)

#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include "cutlass/cutlass.h"
#include "cutlass/util/device_memory.h"

// helper function allocating the memory
void* device_memory_allocation(size_t size, int device_id=0) {
    if (size > 0) {
        torch::Device device(torch::kCUDA, device_id);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        torch::TensorOptions options = torch::TensorOptions().dtype(torch::kI8).device(device);
        at::Tensor device_tensor = torch::empty({(long)size,}, options);
        return reinterpret_cast<void*>(device_tensor.data_ptr());
    } else {
        return nullptr;
    }
}


#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/gemm/device/gemm_grouped.h"


// Gemm operator cutlass_tensorop_s1688tf32gemm_grouped_256x128_32x3_tt_align1
using cutlass_tensorop_s1688tf32gemm_grouped_256x128_32x3_tt_align1_base =
  typename cutlass::gemm::kernel::DefaultGemmGrouped<
    float, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 4,
    float, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 4,
    float, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<float, 1, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
    3,
    cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly,
    cutlass::arch::OpMultiplyAdd
>::GemmKernel;

// Define named type
struct cutlass_tensorop_s1688tf32gemm_grouped_256x128_32x3_tt_align1_type :
  public cutlass_tensorop_s1688tf32gemm_grouped_256x128_32x3_tt_align1_base { };

using DeviceKernel = cutlass::gemm::device::GemmGrouped<cutlass_tensorop_s1688tf32gemm_grouped_256x128_32x3_tt_align1_base>;


using ElementCompute = typename DeviceKernel::EpilogueOutputOp::ElementCompute;

int threadblock_count = DeviceKernel::sufficient();

cutlass::Status grouped_gemm_kernel_run(int problem_count, cutlass::gemm::GemmCoord* problem_sizes,
                        DeviceKernel::ElementA** A, DeviceKernel::ElementB** B, DeviceKernel::ElementC** C, DeviceKernel::ElementC** D,
                        int64_t* lda, int64_t* ldb, int64_t* ldc, int64_t* ldd,
                        ElementCompute alpha, ElementCompute beta) {

  typename DeviceKernel::Arguments arguments {
    problem_sizes,
    problem_count,
    threadblock_count,
    {alpha, beta},
    A, B, C, D,
    lda, ldb, ldc, ldd
  };

  size_t workspace_size = DeviceKernel::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  DeviceKernel gemm_op;
  cutlass::Status status = gemm_op.initialize(arguments,
                                              workspace.get(),
                                              nullptr);     // CUDA stream

  if (status != cutlass::Status::kSuccess) {
    return status;
  }

  status = gemm_op();
  return status;
}

std::vector<at::Tensor> grouped_gemm_kernel(const std::vector<at::Tensor>& A, const std::vector<at::Tensor>& B, at::optional<const std::vector<at::Tensor>> C, float alpha, float beta) {
    size_t num = A.size();

    // To avoid performing many small cudaMallocs and host-to-device copies,
    // we serialize the grouped GEMM arguments on the host, allocate one
    // large chunk of device memory, and perform a single cudaMemcpy to
    // copy the host data to the device. Allocation overheads could be
    // avoided by using a memory pool.

    // Calculate the total size of the data to be copied from host to device
    size_t total_size = sizeof(cutlass::gemm::GemmCoord) +
                        sizeof(DeviceKernel::ElementA*) +
                        sizeof(DeviceKernel::ElementB*) +
                        sizeof(DeviceKernel::ElementC*) +
                        sizeof(DeviceKernel::ElementC*) +
                        sizeof(int64_t) +
                        sizeof(int64_t) +
                        sizeof(int64_t);
    total_size *= num;

    // num * sizeof(cutlass::gemm::GemmCoord) may leave one at a non-multiple
    // of sizeof(DeviceKernel::ElementA*) (which will be 64 on a 64-bit system).
    // To ensure that we don't end up having misaligned loads in the kernel,
    // we pad to the nearest multiple of 8.
    //
    // Note that, even on a 32-bit system (for which sizeof(X*) will not equal
    // sizeof(int64_t)), only padding between the list of GemmCoords and the
    // list of ptr_As is sufficient because the set of four equal-length lists of pointers
    // (A*, B*, C*, D*) will ensure that the first list of int64_ts will always
    // start on a multiple of 8.
    int64_t padding = 8 - (total_size % 8);
    total_size += padding;

    uint8_t* host_data = new uint8_t[total_size];
    cutlass::DeviceAllocation<uint8_t> device_data(total_size);

    uint8_t* start = host_data;
    cutlass::gemm::GemmCoord* problem_sizes_host = reinterpret_cast<cutlass::gemm::GemmCoord*>(start);

    // Apply the padding after the list of GemmCoords
    start += num * sizeof(cutlass::gemm::GemmCoord) + padding;

    int64_t ptr_A_offset = start - host_data;
    DeviceKernel::ElementA** ptr_A_host = reinterpret_cast<DeviceKernel::ElementA**>(start);
    start += num * sizeof(DeviceKernel::ElementA*);

    int64_t ptr_B_offset = start - host_data;
    DeviceKernel::ElementB** ptr_B_host = reinterpret_cast<DeviceKernel::ElementB**>(start);
    start += num * sizeof(DeviceKernel::ElementB*);

    int64_t ptr_C_offset = start - host_data;
    DeviceKernel::ElementC** ptr_C_host = reinterpret_cast<DeviceKernel::ElementC**>(start);
    start += num * sizeof(DeviceKernel::ElementC*);

    int64_t ptr_D_offset = start - host_data;
    DeviceKernel::ElementC** ptr_D_host = reinterpret_cast<DeviceKernel::ElementC**>(start);
    start += num * sizeof(DeviceKernel::ElementC*);

    int64_t lda_offset = start - host_data;
    int64_t* lda_host = reinterpret_cast<int64_t*>(start);
    start += num * sizeof(int64_t);

    int64_t ldb_offset = start - host_data;
    int64_t* ldb_host = reinterpret_cast<int64_t*>(start);
    start += num * sizeof(int64_t);

    int64_t ldc_offset = start - host_data;
    int64_t* ldc_host = reinterpret_cast<int64_t*>(start);
    start += num * sizeof(int64_t);

    std::vector<at::Tensor> D(num);

    bool need_C = (C != at::nullopt) && (beta != 0.f);
    for (size_t i = 0; i < num; ++i) {
        int M = A[i].size(0);
        int N = B[i].size(1);
        int K = A[i].size(1);
        *(problem_sizes_host + i) = {M, N, K};
        *(ptr_A_host + i) = reinterpret_cast<typename DeviceKernel::ElementA*>(A[i].contiguous().data_ptr());
        *(ptr_B_host + i) = reinterpret_cast<typename DeviceKernel::ElementB*>(B[i].contiguous().data_ptr());

        if (need_C) {
            *(ptr_C_host + i) = reinterpret_cast<typename DeviceKernel::ElementC*>(C->at(i).contiguous().data_ptr());
        }
        else {
            *(ptr_C_host + i) = nullptr;
        }

        D[i] = B[i].new_empty({M, N}, torch::kF32);
        *(ptr_D_host + i) = reinterpret_cast<typename DeviceKernel::ElementC*>(D[i].contiguous().data_ptr());

        *(lda_host + i) = DeviceKernel::LayoutA::packed({M, K}).stride(0);
        *(ldb_host + i) = DeviceKernel::LayoutB::packed({K, N}).stride(0);
        *(ldc_host + i) = DeviceKernel::LayoutC::packed({M, N}).stride(0);
    }

    device_data.copy_from_host(host_data);

    cutlass::Status status = grouped_gemm_kernel_run(
        num,
        reinterpret_cast<cutlass::gemm::GemmCoord*>(device_data.get()),
        reinterpret_cast<DeviceKernel::ElementA**>(device_data.get() + ptr_A_offset),
        reinterpret_cast<DeviceKernel::ElementB**>(device_data.get() + ptr_B_offset),
        reinterpret_cast<DeviceKernel::ElementC**>(device_data.get() + ptr_C_offset),
        reinterpret_cast<DeviceKernel::ElementC**>(device_data.get() + ptr_D_offset),
        reinterpret_cast<int64_t*>(device_data.get() + lda_offset),
        reinterpret_cast<int64_t*>(device_data.get() + ldb_offset),
        reinterpret_cast<int64_t*>(device_data.get() + ldc_offset),
        reinterpret_cast<int64_t*>(device_data.get() + ldc_offset),
        ElementCompute(alpha), ElementCompute(beta));

    delete[] host_data;

    TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS kernel failed");
    return D;
}

