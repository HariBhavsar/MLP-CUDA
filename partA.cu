#include <iostream>
#include <bits/stdc++.h>
#include <chrono>
#include <cuda_runtime.h>
#include <cuda/cmath>
#include <cublas_v2.h>
using namespace std::chrono;

#define EPSILON 0.001
#define BATCH_SIZE 32
#define TILE_SIZE 32
#define IDX(Y,X,N) ((((Y) * (N)) + (X)))
#define NUM_STREAMS 4

__global__ void matmul(float* W, float* X, float*WX, int N, int numBlocks) {
    // Given: X -> (N, BATCH_SIZE), W -> (N, N)
    // Output: WX -> (N, BATCH_SIZE)
    // We will create atmosst 32 blocks.
    // Each block will compute (N/numBlocks, BATCH_SIZE) ka WX
    // (tx, ty, by) -> Compute (by * (N/numBlocks) + ty + i * 32, tx) where i is ranging from 0 to (N/(numBlocks * 32))
    // Shared memory? We need (N/numBlocks,TILE_SIZE) and (TILE_SIZE, BATCH_SIZE) ka shared memory
    extern __shared__ float sMem[];
    int numRows = N/numBlocks;
    float* tileW = sMem;
    float* tileX = &(sMem[numRows * TILE_SIZE]);
    int numTiles = N/TILE_SIZE;
    int numelem = N/(numBlocks * 32);
    for (int tileId = 0; tileId < numTiles; tileId++) {
        tileX[IDX(threadIdx.y,threadIdx.x,BATCH_SIZE)] = X[IDX(threadIdx.y + tileId * TILE_SIZE, threadIdx.x, BATCH_SIZE)];
        // first load
        // only one element of X
        for (int i=0; i < numelem; i++) {
            tileW[IDX(threadIdx.y + i * 32, threadIdx.x, TILE_SIZE)] = W[IDX(blockIdx.y * numRows + threadIdx.y + i * 32, threadIdx.x + tileId * TILE_SIZE, N)];
        }
        __syncthreads();
        for (int i=0; i < numelem; i++) {
            float localSum = 0;
            for (int k=0; k < TILE_SIZE; k++) {
                localSum += tileW[IDX(threadIdx.y + i * 32, k, TILE_SIZE)] * tileX[IDX(k, threadIdx.x, BATCH_SIZE)];           
            }
            WX[IDX(blockIdx.y * numRows + threadIdx.y + i * 32, threadIdx.x, BATCH_SIZE)] += localSum;
        }
        __syncthreads();
    }
}

__global__ void matmulTwo(float *W, float *X, float *WX, int N, int numBlocks, int streamID) {
    // tile for X will now be (BATCH_SIZE/4, TILE_SIZE)
    // tile for W will now be (N/numBlocks , TILE_SIZE)
    extern __shared__ float sMem[];
    int numRows = (N/numBlocks);
    float *tileW = sMem;
    float *tileX = &(sMem[numRows * TILE_SIZE]);
    int numTiles = N/TILE_SIZE;
    int numElem = (N/(numBlocks * 32));
    float localSum = 0.0f;
    for (int tileId = 0; tileId < numTiles; tileId++) {
        tileX[IDX(threadIdx.y, threadIdx.x, (BATCH_SIZE/NUM_STREAMS))] = X[IDX(threadIdx.y + tileId * TILE_SIZE, threadIdx.x + streamID * (BATCH_SIZE/4),BATCH_SIZE)];
        for (int i=0; i<numElem; i++) {
            for (int j=0; j < NUM_STREAMS; j++) {
                tileW[IDX(threadIdx.y + i * 32, threadIdx.x + j * (TILE_SIZE/NUM_STREAMS), TILE_SIZE)] = W[IDX(blockIdx.y * numRows + threadIdx.y + i * 32, threadIdx.x + j * (TILE_SIZE/NUM_STREAMS) + tileId * TILE_SIZE, N)];
            }
        }
        __syncthreads();
        for (int i=0; i < numElem; i++) {
            localSum = 0.0f;
            for (int k=0; k < TILE_SIZE; k++) {
                localSum += tileW[IDX(threadIdx.y + i * 32, k, TILE_SIZE)] * tileX[IDX(k, threadIdx.x, (BATCH_SIZE/NUM_STREAMS))];
            }
            WX[IDX(blockIdx.y * numRows + threadIdx.y + i * 32, threadIdx.x + streamID * (BATCH_SIZE/NUM_STREAMS), BATCH_SIZE)] += localSum;
        }
        __syncthreads();
    }
}

__global__ void reLU(float *input, float *output, int N) {
    // Dimension of input = dimension of output = (N, BATCH_SIZE)
    // We will create N/32 blocks
    // (tx,ty,by) has to check (by * 32 + ty, tx)
    output[IDX(blockIdx.y * 32 + threadIdx.y, threadIdx.x, BATCH_SIZE)] = fmaxf(input[IDX(blockIdx.y * 32 + threadIdx.y, threadIdx.x, BATCH_SIZE)], 0.0f);
}

__global__ void reLUTwo(float *input, float* output, int N, int streamId) {
    output[IDX(blockIdx.y * 32 + threadIdx.y, threadIdx.x + streamId * (BATCH_SIZE/4), BATCH_SIZE)] = fmaxf(input[IDX(blockIdx.y * 32 + threadIdx.y, threadIdx.x + streamId * (BATCH_SIZE/4), BATCH_SIZE)], 0.0f);
}

__global__ void dummy(float *X, float *Y, int N) {
    float x = Y[IDX(threadIdx.y, threadIdx.x, N)];
    x += 1.0f;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr << "Usage: ./partA <Matrix Size> <Mode: 0 -> CPU, 1 -> GPU, 2 -> Both>" << std::endl;
        return 1;       
    }   
    int N = std::stoi(argv[1]);
    int mode = std::stoi(argv[2]);
    float *W1 = new float[N * N];
    float *W2 = new float[N * N];
    float *W1X = new float[N * BATCH_SIZE]; // W1X = W1 * X
    float* Y = new float[N * BATCH_SIZE]; // Y = ReLU(W1X)
    float* Z = new float[N * BATCH_SIZE]; // Z = W2 * Y
    float* X = new float[N * BATCH_SIZE]; 
    float *ZCpy = new float[N * BATCH_SIZE];
    for (int i=0; i < N; i++) {
        for (int j=0; j < N; j++) {
            W1[i * N + j] = (((float)(rand()))/((float)RAND_MAX));
            W2[i * N + j] = (((float)(rand()))/((float)RAND_MAX));
        }
    }
    for (int i=0; i < N; i++) {
        for (int j=0; j < BATCH_SIZE; j++) {
            X[i * BATCH_SIZE + j] = (((float)(rand()))/((float)RAND_MAX));
        }
    }
    dim3 g(32, 32);
    dim3 b(32, 32);
    dummy<<<g,b>>>(W1,W1,N);
    std::cout << "Matrices setup successfully" << std::endl;
    // First, CPU based matrix multiplication
    if (mode & 1) {
        auto prev = high_resolution_clock::now();
        for (int i=0; i < N; i++) {
            for (int j=0; j < BATCH_SIZE; j++) {
                W1X[i * BATCH_SIZE + j] = 0;
                for (int k=0; k < N; k++) {
                    W1X[i * BATCH_SIZE + j] += W1[i * N + k] * X[k * BATCH_SIZE + j];
                }
                Y[i * BATCH_SIZE + j] = std::max(0.0f, W1X[i * BATCH_SIZE + j]);
            }
        }
        for (int i=0; i < N; i++) {
            for (int j=0; j < BATCH_SIZE; j++) {
                Z[i * BATCH_SIZE + j] = 0;
                for (int k=0; k < N; k++) {
                    Z[i * BATCH_SIZE + j] += W2[i * N + k] * Y[k * BATCH_SIZE + j];
                }
            }
        }
        auto curr = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(curr - prev);
        std::cout << "[CPU] Time taken: " << duration.count() << " microseconds" << std::endl;
    }
    if (mode & 2) {
        // GPU based matrix multiplication
        auto prev = high_resolution_clock::now();
        float *W1GPU = nullptr;
        float *XGPU = nullptr;
        float *W1XGPU = nullptr;
        cudaMalloc(&W1GPU, sizeof(float) * N * N);
        cudaMalloc(&XGPU, sizeof(float) * N * BATCH_SIZE);
        cudaMalloc(&W1XGPU, sizeof(float) * N * BATCH_SIZE);
        cudaMemcpy(W1GPU, W1, sizeof(float) * N * N, cudaMemcpyDefault);
        cudaMemcpy(XGPU, X, sizeof(float) * N * BATCH_SIZE, cudaMemcpyDefault);
        cudaMemset(W1XGPU, 0, sizeof(float) * N * BATCH_SIZE);
        dim3 block(32,32);
        // need to get numblocks
        // We may need more than 32 blocks!
        // For Size 32768, smemsize explodes too much :(
        int numBlocks = min(N/32, 32);
        size_t sMemSize = ((N/numBlocks) * TILE_SIZE + TILE_SIZE * BATCH_SIZE) * sizeof(float);
        if (sMemSize > 69632) {
            // Need to recalculate numBlocks and sMemSize
            numBlocks = numBlocks * (N / 16384);
            sMemSize = 69632;
        }
        dim3 grid(1, numBlocks, 1);
        cudaFuncSetAttribute(matmul, cudaFuncAttributeMaxDynamicSharedMemorySize, sMemSize);
        matmul<<<grid , block, sMemSize>>>(W1GPU,XGPU,W1XGPU,N, numBlocks);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        }
        cudaDeviceSynchronize();
        dim3 grid2(1, N/32, 1);
        float* YGPU = XGPU; // We can reuse hehe
        reLU<<<grid2, block>>>(W1XGPU, YGPU, N);
        cudaDeviceSynchronize();
        float* W2GPU = W1GPU;
        cudaMemcpy(W2GPU, W2, sizeof(float) * N * N, cudaMemcpyDefault);
        float* ZGPU = W1XGPU;
        cudaMemset(ZGPU, 0, sizeof(float) * N * BATCH_SIZE);
        matmul<<<grid, block, sMemSize>>>(W2GPU, YGPU, ZGPU, N, numBlocks);
        cudaDeviceSynchronize();
        cudaMemcpy(ZCpy, ZGPU, sizeof(float) * N * BATCH_SIZE, cudaMemcpyDefault);
        auto curr = high_resolution_clock::now();
        if (mode & 1) {
            for (int i=0; i < N; i++) {
                for (int j=0; j < BATCH_SIZE; j++) {
                    if (std::fabs(ZCpy[i * BATCH_SIZE + j] - Z[i * BATCH_SIZE + j])/Z[i * BATCH_SIZE+ j] > EPSILON) {
                        std::cout << "Incorrect result of matmul at indices: (" << i << ", " << j << ")" << std::endl;
                        std::cout << "Expected: " << Z[i * BATCH_SIZE + j] << ", Got: " << ZCpy[i * BATCH_SIZE + j] << std::endl;
                        return 1;
                    }
                }
            }
        }
        cudaFree(W1GPU);
        cudaFree(XGPU);
        cudaFree(W1XGPU);
        auto duration = duration_cast<microseconds>(curr - prev);
        std::cout << "[GPU] Time Taken: " << duration.count() << " microseconds" << std::endl;
    }
    if (mode & 4) {
        auto prev = high_resolution_clock::now();
        float *W1GPU = nullptr;
        float *XGPU = nullptr;
        float *W1XGPU = nullptr;
        float *YGPU = nullptr;
        float *ZGPU = nullptr;
        float *W2GPU = nullptr;
        cudaMalloc(&W1GPU, sizeof(float) * N * N);
        cudaMalloc(&XGPU, sizeof(float) * N * BATCH_SIZE);
        cudaMalloc(&W1XGPU, sizeof(float) * N * BATCH_SIZE);
        cudaMalloc(&YGPU, sizeof(float) * N * BATCH_SIZE);
        cudaMalloc(&ZGPU, sizeof(float) * N * BATCH_SIZE);
        cudaMalloc(&W2GPU, sizeof(float) * N * N);
        cudaMemcpy(W1GPU, W1, sizeof(float) * N * N, cudaMemcpyDefault);
        cudaMemcpy(W2GPU, W2, sizeof(float) * N * N, cudaMemcpyDefault);
        cudaMemcpy(XGPU, X, sizeof(float) * N * BATCH_SIZE, cudaMemcpyDefault);
        cudaMemset(W1XGPU, 0, sizeof(float) * N * BATCH_SIZE);
        cudaMemset(YGPU, 0, sizeof(float) * N * BATCH_SIZE);
        cudaMemset(ZGPU, 0, sizeof(float) * N * BATCH_SIZE);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA1 Error: " << cudaGetErrorString(err) << std::endl;
        }
        dim3 block(8, 32);
        // need to get numblocks
        int numBlocks = min(N/32, 32);
        size_t sMemSize = ((N/numBlocks) * TILE_SIZE + TILE_SIZE * (BATCH_SIZE/NUM_STREAMS)) * sizeof(float);
        if (sMemSize > 69632) {
            // Need to recalculate numBlocks and sMemSize
            numBlocks = numBlocks * (N / 16384);
            sMemSize = 69632;
        }
        dim3 grid(1, numBlocks, 1);
        cudaFuncSetAttribute(matmulTwo, cudaFuncAttributeMaxDynamicSharedMemorySize, sMemSize);
        cudaStream_t streams[NUM_STREAMS];
        for (int i=0; i < NUM_STREAMS; i++) {
            cudaStreamCreate(&streams[i]);
        }
        for (int i=0; i < NUM_STREAMS; i++) {
            matmulTwo<<<grid , block, sMemSize, streams[i]>>>(W1GPU,XGPU,W1XGPU,N, numBlocks, i);
            dim3 grid2(1, N/32, 1);
            reLUTwo<<<grid2, block, 0, streams[i]>>>(W1XGPU, YGPU, N, i);
            matmulTwo<<<grid, block, sMemSize, streams[i]>>>(W2GPU, YGPU, ZGPU, N, numBlocks, i);
        }
        cudaDeviceSynchronize();
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA2 Error: " << cudaGetErrorString(err) << std::endl;
        }
        cudaMemcpy(ZCpy, ZGPU, sizeof(float) * N * BATCH_SIZE, cudaMemcpyDefault);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA3 Error: " << cudaGetErrorString(err) << std::endl;
        }
        auto curr = high_resolution_clock::now();
        if (mode & 1) {
            for (int i=0; i < N; i++) {
                for (int j=0; j < BATCH_SIZE; j++) {
                    if (std::fabs(ZCpy[i * BATCH_SIZE + j] - Z[i * BATCH_SIZE + j])/Z[i * BATCH_SIZE+ j] > EPSILON) {
                        std::cout << "Incorrect result of matmul at indices: (" << i << ", " << j << ")" << std::endl;
                        std::cout << "Expected: " << Z[i * BATCH_SIZE + j] << ", Got: " << ZCpy[i * BATCH_SIZE + j] << std::endl;
                        return 1;
                    }
                }
            }
        }
        cudaFree(W1GPU);
        cudaFree(XGPU);
        cudaFree(W1XGPU);
        cudaFree(W2GPU);
        cudaFree(YGPU);
        cudaFree(ZGPU);
        auto duration = duration_cast<microseconds>(curr - prev);
        std::cout << "[GPU] Time Taken: " << duration.count() << " microseconds" << std::endl;
    }
    delete []W1;
    delete []X;
    delete []W2;
    delete []W1X;
    delete []Y;
    delete []Z;
    delete []ZCpy;
}