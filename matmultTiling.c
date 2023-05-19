#include <stdio.h>

#define N 500
#define TILE_WIDTH 64

__global__ void matrixMulTiled(int* A, int* B, int* C) {
    __shared__ int tile_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ int tile_B[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    int sum = 0;

    for (int m = 0; m < N / TILE_WIDTH; m++) {
        tile_A[ty][tx] = A[row * N + m * TILE_WIDTH + tx];
        tile_B[ty][tx] = B[(m * TILE_WIDTH + ty) * N + col];

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; k++) {
            sum += tile_A[ty][k] * tile_B[k][tx];
        }

        __syncthreads();
    }

    C[row * N + col] = sum;
}

int main() {
    int *h_A, *h_B, *h_C; 
    int *d_A, *d_B, *d_C; 

    int size = N * N * sizeof(int);

    h_A = (int*)malloc(size);
    h_B = (int*)malloc(size);
    h_C = (int*)malloc(size);

    for (int i = 0; i < N * N; i++) {
        h_A[i] = i;
        h_B[i] = i;
    }

    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid(N / TILE_WIDTH, N / TILE_WIDTH);

    matrixMulTiled<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
