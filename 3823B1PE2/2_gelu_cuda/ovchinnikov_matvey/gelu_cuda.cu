#include "gelu_cuda.h"
#include <cmath>
#include <future>



__global__ void GeluFor(float* data, float k, size_t s){
    size_t i=(size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if(i<s){
        float x = data[i];
        data[i]=x / (1.0f + expf(k * (x + 0.044715f * x * x * x)));
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
  size_t s = input.size();
  auto fut = std::async(std::launch::async, [s] {
    return std::vector<float>(s);
  });
  static float *d_data=nullptr;
  const float k = -2 * sqrt(2 / M_PI);
  
  
  static size_t allocated=0;
  size_t bs=s*sizeof(float);
  if(bs>allocated){
    cudaFree(d_data);
    cudaMalloc(&d_data, bs);
    allocated=bs;
  }



  cudaMemcpy(d_data, input.data(), bs, cudaMemcpyHostToDevice);
  int p=256;
  int chunks=(s+p-1)/p;
  GeluFor<<<chunks,p>>>(d_data, k, s);
  std::vector<float> res = fut.get();
  cudaMemcpy(res.data(), d_data, bs, cudaMemcpyDeviceToHost);
  return res;
}