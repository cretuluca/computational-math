#include <stdio.h>
#include <cuda_runtime.h>

__global__ void linearRegression(float *x, float *y, float *w, float *b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float pred = (*w) * x[idx] + (*b);
        float error = pred - y[idx];
        atomicAdd(w, -0.01f * error * x[idx] / n);
        atomicAdd(b, -0.01f * error / n);
    }
}

__global__ void computeMSE(float *x, float *y, float *w, float *b, float *mse, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float pred = (*w) * x[idx] + (*b);
        float error = y[idx] - pred;
        atomicAdd(mse, error * error / n);
    }
}

int main() {
    float *x, *y, *d_x, *d_y;
    float w = 0.0f, b = 0.0f, mse = 0.0f;
    float *d_w, *d_b, *d_mse;

    x = (float*)malloc(1000000 * sizeof(float));
    y = (float*)malloc(1000000 * sizeof(float));

    for (int i = 0; i < 1000000; i++) {
        x[i] = (float)rand() / RAND_MAX;
        y[i] = 2 * x[i] + 1 + ((float)rand() / RAND_MAX - 0.5f) * 0.2f;
    }

    cudaMalloc(&d_x, 1000000 * sizeof(float));
    cudaMalloc(&d_y, 1000000 * sizeof(float));
    cudaMalloc(&d_w, sizeof(float));
    cudaMalloc(&d_b, sizeof(float));
    cudaMalloc(&d_mse, sizeof(float));

    cudaMemcpy(d_x, x, 1000000 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, 1000000 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, &w, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, sizeof(float), cudaMemcpyHostToDevice);

    int blocks = (1000000 + 256 - 1) / 256;
    for (int i = 0; i < 1000; i++) {
        linearRegression<<<blocks, 256>>>(d_x, d_y, d_w, d_b, N);
    }

    cudaMemset(d_mse, 0, sizeof(float));
    computeMSE<<<blocks, 256>>>(d_x, d_y, d_w, d_b, d_mse, N);

    cudaMemcpy(&w, d_w, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&b, d_b, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&mse, d_mse, sizeof(float), cudaMemcpyDeviceToHost);

    printf("Weight: %f\n", w);
    printf("Bias: %f\n", b);
    printf("MSE: %f\n", mse);

    free(x);
    free(y);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_w);
    cudaFree(d_b);
    cudaFree(d_mse);

    return 0;
}