#ifndef _idaklu_JAX_KERNELS_H_
#define _idaklu_JAX_KERNELS_H_

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace idaklu_jax {
struct idakluDescriptor {
  std::int64_t size;
};

void gpu_idaklu_f32(cudaStream_t stream, void** buffers, const char* opaque,
                    std::size_t opaque_len);
void gpu_idaklu_f64(cudaStream_t stream, void** buffers, const char* opaque,
                    std::size_t opaque_len);

}  // namespace idaklu_jax

#endif
