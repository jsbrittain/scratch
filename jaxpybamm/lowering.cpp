#include <cstdint>

/*
 * Custom XLA call signatures is:
 *   void custom_call(void* out, const void** in);
 * See
 *   https://www.tensorflow.org/xla/custom_call
 */
void solve_cpu(void* out, const void** in);





template <typename T>
void solve_cpu(void* out, const void** in) {
  // Parse inputs: [size, array1, array2]
  const std::int64_t size = *reinterpret_cast<const std::int64_t*>(in[0]);
  const T* array1 = reinterpret_cast<const T*>(in[1]);
  const T* array2 = reinterpret_cast<const T*>(in[2]);

  // Pointer to outputs (value and grad)
  void** output = reinterpret_cast<void**>(out);
  T* output1 = reinterpret_cast<T*>(output[0]);
  T* output2 = reinterpret_cast<T*>(output[1]);

  // Compute
  // TODO: IMPLEMENT: Add array1 and array2 and store in output1 for now
  for (std::int64_t i = 0; i < size; ++i) {
    output1[i] = array1[i] + array2[i];
    output2[i] = array1[i] - array2[i];
  }
}
