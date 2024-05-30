#include <torch/extension.h>
using data_type = float;

void multi_tensor_adam_cuda(
  int chunk_size,
  at::Tensor noop_flag,
  std::vector<std::vector<at::Tensor>> tensor_lists,
  const float lr,
  const float beta1,
  const float beta2,
  const float epsilon,
  const int step,
  const int mode,
  const int bias_correction,
  const float weight_decay);

void multi_tensor_adam_capturable_cuda(
  int chunk_size,
  at::Tensor noop_flag,
  std::vector<std::vector<at::Tensor>> tensor_lists,
  at::Tensor lr,
  const float beta1,
  const float beta2,
  const float epsilon,
  at::Tensor step,
  const int mode,
  const int bias_correction,
  const float weight_decay,
  at::Tensor inv_scale);

void multi_tensor_adam_capturable_master_cuda(
  int chunk_size,
  at::Tensor noop_flag,
  std::vector<std::vector<at::Tensor>> tensor_lists,
  at::Tensor lr,
  const float beta1,
  const float beta2,
  const float epsilon,
  at::Tensor step,
  const int mode,
  const int bias_correction,
  const float weight_decay,
  at::Tensor inv_scale);

void multi_tensor_eva_cuda(
    int chunk_size,
    at::Tensor noop_flag,
    std::vector<std::vector<at::Tensor>> tensor_lists,
    const float lr,
    const float damping,
    const float kl_clip);


std::vector<at::Tensor>& multi_group(
                const std::vector<at::Tensor>& A,
                const std::vector<at::Tensor>& B,
                std::vector<at::Tensor>& C);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

  m.def("multi_tensor_adam", &multi_tensor_adam_cuda,
        "Compute and apply gradient update to parameters for Adam optimizer");
  m.def("multi_tensor_adam_capturable", &multi_tensor_adam_capturable_cuda,
        "Compute and apply gradient update to parameters for Adam optimizer with CUDA graph support and LR scheduling");
  m.def("multi_tensor_adam_capturable_master", &multi_tensor_adam_capturable_master_cuda,
        "Compute and apply gradient update to parameters for Adam optimizer with CUDA graph support, LR scheduling and FP32 master weights");
  m.def("multi_tensor_eva", &multi_tensor_eva_cuda,
        "Compute and apply gradient update to parameters for EVA optimizer");
  m.def("multi_group", &multi_group,
        "Accelerated matrix multiplication");
}
