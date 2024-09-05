#include <cuda_runtime.h>
#include <stdio.h>

#define cudaCheckError() { \
    cudaError_t e=cudaGetLastError(); \
    if(e!=cudaSuccess) { \
        printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
        exit(0); \
    } \
}

__global__ void gradientDescentKernel(float *x, float *y, float *m, float *b, int n, float learning_rate) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float prediction = (*m) * x[idx] + (*b);
        float error = prediction - y[idx];
        atomicAdd(m, -learning_rate * error * x[idx] / n);
        atomicAdd(b, -learning_rate * error / n);
    }
}

void linearRegression(float *x, float *y, int n, int iterations, float learning_rate) {
    float *d_x, *d_y, *d_m, *d_b;
    cudaMalloc(&d_x, n * sizeof(float));
    cudaMalloc(&d_y, n * sizeof(float));
    cudaMalloc(&d_m, sizeof(float));
    cudaMalloc(&d_b, sizeof(float));
    cudaCheckError();

    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckError();

    float h_m = 0, h_b = 0;
    cudaMemcpy(d_m, &h_m, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &h_b, sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckError();

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    for (int i = 0; i < iterations; i++) {
        gradientDescentKernel<<<numBlocks, blockSize>>>(d_x, d_y, d_m, d_b, n, learning_rate);
        cudaCheckError();
    }

    cudaMemcpy(&h_m, d_m, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_b, d_b, sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckError();

    printf("Slope (m): %f\n", h_m);
    printf("Intercept (b): %f\n", h_b);

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_m);
    cudaFree(d_b);
}

int main() {
    const int N = 1000000;
    float *x = new float[N];
    float *y = new float[N];

    for (int i = 0; i < N; i++) {
        x[i] = static_cast<float>(i) / N;
        y[i] = 2 * x[i] + 1 + static_cast<float>(rand()) / RAND_MAX * 0.1f;
    }

    linearRegression(x, y, N, 1000, 0.1f);

    delete[] x;
    delete[] y;

    return 0;
}