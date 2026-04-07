#include <iostream>
#include <bits/stdc++.h>
#include <chrono>
#include <cuda_runtime.h>
#include <cuda/cmath>
#include <cublas_v2.h>
using namespace std::chrono;

#define N 1024
#define IN 512
#define L 4
const int H[L-1] = {2048, 2048, 2048};
#define OUT 512
#define EPS 0.001

#define IDX(Y,X,N) ((((Y) * (N)) + (X)))

#define TILE_SIZE 32
// #define DEBUG_PRINT

__global__ void matmul(float* A, float* B, float* C, int P, int Q, int R, bool aTrans=false, bool bTrans=false) {
    // A is a PxQ matrix
    // B is a QxR matrix
    // C is a PxR matrix, C = A x B
    // Given: Each of P,Q,R are multiples of 32
    // we will use simple tiling
    // 32x32 threads per block
    __shared__ float tileA[TILE_SIZE * TILE_SIZE];
    __shared__ float tileB[TILE_SIZE * TILE_SIZE];
    float localSum = 0.0f;
    for (int tileId = 0; tileId < (Q/TILE_SIZE); ++tileId) {
        if (!aTrans) {
            tileA[IDX(threadIdx.y, threadIdx.x, TILE_SIZE)] = A[IDX(blockIdx.y * blockDim.y + threadIdx.y, tileId * TILE_SIZE + threadIdx.x, Q)];
        }
        else {
            tileA[IDX(threadIdx.y, threadIdx.x, TILE_SIZE)] = A[IDX(tileId * TILE_SIZE + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y, P)]; // Think of A as a Q x P matrix now 
        }
        if (!bTrans) {
            tileB[IDX(threadIdx.y, threadIdx.x, TILE_SIZE)] = B[IDX(tileId * TILE_SIZE + threadIdx.y, blockIdx.x * blockDim.x + threadIdx.x, R)];
        }
        else {
            tileB[IDX(threadIdx.y, threadIdx.x, TILE_SIZE)] = B[IDX( blockIdx.x * blockDim.x + threadIdx.x, tileId * TILE_SIZE + threadIdx.y, Q)];
        }
        __syncthreads();
        for (int i=0; i < TILE_SIZE; i++) {
            localSum += tileA[IDX(threadIdx.y, i, TILE_SIZE)] * tileB[IDX(i, threadIdx.x, TILE_SIZE)];
        }
        __syncthreads();
    } 
    C[IDX(blockIdx.y * blockDim.y + threadIdx.y, blockIdx.x * blockDim.x + threadIdx.x, R)] = localSum;
}

__global__ void matAdd(float *A, float* B, float *C, int numCols) {
    C[IDX(blockIdx.y * blockDim.y + threadIdx.y, blockIdx.x * blockDim.x + threadIdx.x, numCols)] = A[IDX(blockIdx.y * blockDim.y + threadIdx.y, blockIdx.x * blockDim.x + threadIdx.x, numCols)] + B[IDX(blockIdx.y * blockDim.y + threadIdx.y, blockIdx.x * blockDim.x + threadIdx.x, numCols)];
}

__global__ void bpInit(float *A, float* B, float *C, int numCols, int numRows) {
    C[IDX(blockIdx.y * blockDim.y + threadIdx.y, blockIdx.x * blockDim.x + threadIdx.x, numCols)] = 2 * (A[IDX(blockIdx.y * blockDim.y + threadIdx.y, blockIdx.x * blockDim.x + threadIdx.x, numCols)] - B[IDX(blockIdx.y * blockDim.y + threadIdx.y, blockIdx.x * blockDim.x + threadIdx.x, numCols)]);
    C[IDX(blockIdx.y * blockDim.y + threadIdx.y, blockIdx.x * blockDim.x + threadIdx.x, numCols)] = C[IDX(blockIdx.y * blockDim.y + threadIdx.y, blockIdx.x * blockDim.x + threadIdx.x, numCols)]/numRows;
    C[IDX(blockIdx.y * blockDim.y + threadIdx.y, blockIdx.x * blockDim.x + threadIdx.x, numCols)] = C[IDX(blockIdx.y * blockDim.y + threadIdx.y, blockIdx.x * blockDim.x + threadIdx.x, numCols)]/numCols;
}

__global__ void matVecAdd(float *A, float* B, float *C, int numCols) {
    // Be careful!
    C[IDX(blockIdx.y * blockDim.y + threadIdx.y, blockIdx.x * blockDim.x + threadIdx.x, numCols)] = A[IDX(blockIdx.y * blockDim.y + threadIdx.y, blockIdx.x * blockDim.x + threadIdx.x, numCols)] + B[blockIdx.x * blockDim.x + threadIdx.x];
}

__global__ void matVecSub(float *A, float* B, float *C, int numCols) {
    // Be careful!
    C[IDX(blockIdx.y * blockDim.y + threadIdx.y, blockIdx.x * blockDim.x + threadIdx.x, numCols)] = A[IDX(blockIdx.y * blockDim.y + threadIdx.y, blockIdx.x * blockDim.x + threadIdx.x, numCols)] - B[blockIdx.x * blockDim.x + threadIdx.x];
}

__global__ void biasGradientGPU(float *dB, float *dZ, int outDim, int batchSize) {
    // Assumption: 1D grid
    dB[blockIdx.x * blockDim.x + threadIdx.x] = 0;
    for (int i=0; i < batchSize; i++) {
        dB[blockIdx.x * blockDim.x + threadIdx.x] += dZ[IDX(i, blockIdx.x * blockDim.x + threadIdx.x, outDim)];
    }
}

__global__ void reLU(float *A, float *B, int numCols) {
    // 32 x 32 threads per block
    B[IDX(blockIdx.y * blockDim.y + threadIdx.y, blockIdx.x * blockDim.x + threadIdx.x, numCols)] = fmaxf(A[IDX(blockIdx.y * blockDim.y + threadIdx.y, blockIdx.x * blockDim.x + threadIdx.x, numCols)], 0.0f);
}

__global__ void reLUGrad(float *dZI, float *dAI, float *ZI, int batchSize, int outDim) {
    int index = IDX(blockIdx.y * blockDim.y + threadIdx.y, blockIdx.x * blockDim.x + threadIdx.x, outDim);
    dZI[index] = dAI[index] * ((ZI[index] > EPS ? 1 : 0));
}

void cpuMatMul(float *A, float *B, float *C, int P, int Q, int R, bool aTrans = false, bool bTrans = false) {
    for (int i=0; i < P; i++) {
        for (int j=0; j < R; j++) {
            C[i * R + j] = 0;
            for (int k=0; k < Q; k++) {
                if (!aTrans && !bTrans) {
                    C[i * R + j] += A[i * Q + k] * B[k * R + j];
                }
                else if (aTrans && !bTrans) {
                    C[i * R + j] += A[k * P + i] * B[k * R + j];    
                }
                else if (bTrans && !aTrans) {
                    C[i * R + j] += A[i * Q + k] * B[j * Q + k];
                }
                else {
                    C[i * R + j] += A[k * P + i] * B[j * Q + k];
                }
            }
        }
    }
}

void fpLayerGPU(float *input, float *Z, float *A, float *weight, float *bias, int P, int Q, int R, bool doReLU = true) {
    // Z = input * weight + bias
    // A = ReLU(Z)
    // if (input == nullptr || Z == nullptr || A == nullptr || weight == nullptr || bias == nullptr) {
    //     std::cerr << "ono" << std::endl;
    //     exit(1);
    // }
    dim3 block(32,32);
    dim3 grid(((R + block.x - 1)/block.x), ((P + block.y - 1)/block.y));
    matmul<<<grid, block>>>(input, weight, Z, P, Q, R);
    cudaDeviceSynchronize();
    matVecAdd<<<grid, block>>>(Z, bias, Z, R);
    cudaDeviceSynchronize();
    if (doReLU) {
        reLU<<<grid, block>>>(Z, A, R);
        cudaDeviceSynchronize();
    }
}

void fpLayerCPU(float *input, float *Z, float *A, float *weight, float *bias, int P, int Q, int R, bool doReLU = true) {
    for (int i=0; i < P; i++) {
        for (int j=0; j < R; j++) {
            // Standard way: Bias is added to the column index 'j'
            float sum = bias[j]; 
            for (int k=0; k < Q; k++) {
                sum += (input[i * Q + k] * weight[k * R + j]);
            }
            Z[i * R + j] = sum;
            if (doReLU) {
                A[i * R + j] = fmaxf(sum, 0.0f);
            }
        }
    }
}

bool checkMatch(float *m1, float *m2, int numRows, int numCols, std::string caller) {
    for (int i=0; i < numRows; i++) {
        for (int j=0; j < numCols; j++) {
            if (m1[i * numCols + j] != 0.0f) {
                if (fabs(m1[i * numCols + j]) > EPS) {
                    if ((fabs(m2[i * numCols + j] - m1[i * numCols + j])/fabs(m1[i * numCols + j])) > EPS) {
                        std::cout << "Mismatch in function: " << caller << std::endl;
                        std::cout << "Indices: (" << i << ", " << j << ")" << std::endl;
                        std::cout << "Expected: " << m1[i * numCols + j] << " Got: " << m2[i * numCols + j] << std::endl;
                        return false;
                    }
                }
                else {
                    if ((fabs(m2[i * numCols + j]) > EPS)) {
                        std::cout << "Mismatch in function: " << caller << std::endl;
                        std::cout << "Indices: (" << i << ", " << j << ")" << std::endl;
                        std::cout << "Expected: " << m1[i * numCols + j] << " Got: " << m2[i * numCols + j] << std::endl;
                        return false;
                    }
                }
            }
            else {
                if (m2[i * numCols + j] != 0.0f) {
                    std::cout << "Mismatch in function: " << caller << std::endl;
                    std::cout << "Indices: (" << i << ", " << j << ")" << std::endl;
                    std::cout << "Expected: " << m1[i * numCols + j] << " Got: " << m2[i * numCols + j] << std::endl;
                    return false;
                }
            }
        }
    }
    return true;
}

float getLoss(float *m1, float *m2, int numRows, int numCols) {
    float sum = 0.0f;
    for (int i=0; i < numRows; i++) {
        for (int j=0; j < numCols; j++) {
            float diff = (m1[i * numCols + j] - m2[i * numCols + j]);
            diff = diff * diff;
            diff = diff/numRows;
            diff = diff/numCols;
            sum += diff;
        }
    }
    return sum;
}

void initBPCPU(float *Z, float *Y, float *dZ, int numRows, int numCols) {
    for (int i=0; i < numRows; i++) {
        for (int j=0; j < numCols; j++) {
            dZ[i * numCols + j] = 2 * (Z[i * numCols + j] - Y[i * numCols + j]);
            dZ[i * numCols + j] = dZ[i * numCols + j]/numRows;
            dZ[i * numCols + j] = dZ[i * numCols + j]/numCols;
        }
    }
}

void lastLayerBPCPU(float *dZ, float *A, float *W, float *dW, float *dB, float *dA, int batchSize, int outDim, int inDim) {
    cpuMatMul(A, dZ, dW, inDim, batchSize, outDim, true, false);
    for (int j=0; j < outDim; j++) {
        dB[j] = 0;
        for (int i=0; i < batchSize; i++) {
            dB[j] += dZ[i * outDim + j];
        }
    }
    if (dA != nullptr) {
        cpuMatMul(dZ, W, dA, batchSize, outDim, inDim, false, true);
    }
}

void lastLayerBPGPU(float *dZ, float *A, float *W, float *dW, float *dB, float *dA, int batchSize, int outDim, int inDim) {
    // First up, calculate dW = A^TdZ
    // dW -> inDim, outDim. A -> batchSize * inDim, dZ -> batchSize * outDim
    dim3 block(32, 32);
    dim3 grid1(((outDim + block.x - 1)/block.x), ((inDim + block.y - 1)/block.y));
    matmul<<<grid1, block>>>(A, dZ, dW, inDim, batchSize, outDim, true, false);
    // Next bias gradient
    dim3 block2(32,1);
    dim3 grid2(((outDim + block2.x - 1)/block2.x), 1);
    biasGradientGPU<<<grid2, block2>>>(dB, dZ, outDim, batchSize);
    // Finally, dA = dZW^T
    if (dA != nullptr) {
        dim3 grid3(((inDim + block.x - 1)/block.x), ((batchSize + block.y - 1)/block.y));
        matmul<<<grid3, block>>>(dZ, W, dA, batchSize, outDim, inDim, false, true);
    }
}

void bpLayerICPU(float *dAI, float *dZI, float *dWI, float *dBI, float *dAIMinus1, float *ZI, float *AIMinus1, float *WI, int batchSize, int outDim, int inDim) {
    // Given: dAI, ZI, AIMinus1, WI
    // To compute: dZI, dWI, dBI, dAIMinus1
    // dZI, ZI, dAI -> batchSize * outDim
    // dAIMinus1, AIMinus1 -> batchSize * inDim
    // dWI, WI -> inDim * outDim
    // dBI -> outDim * 1
    // first, dZI = dAI * (ZI > 0)
    for (int i=0; i < batchSize; i++) {
        for (int j=0; j < outDim; j++) {
            dZI[i * outDim + j] = dAI[i * outDim + j] * ((ZI[i * outDim + j] > EPS ? 1 : 0));
        }
    }
    lastLayerBPCPU(dZI, AIMinus1, WI, dWI, dBI, dAIMinus1, batchSize, outDim, inDim);
}

void bpLayerIGPU(float *dAI, float *dZI, float *dWI, float *dBI, float *dAIMinus1, float *ZI, float *AIMinus1, float *WI, int batchSize, int outDim, int inDim) {
    dim3 block(32, 32);
    dim3 grid(((outDim + block.x - 1)/block.x), ((batchSize + block.y - 1)/block.y));
    reLUGrad<<<grid, block>>>(dZI, dAI, ZI, batchSize, outDim);
    lastLayerBPGPU(dZI, AIMinus1, WI, dWI, dBI, dAIMinus1, batchSize, outDim, inDim);
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "Usage: ./partA <Mode: 0 -> FP-only 1 -> Transpose matmul testing>" << std::endl;
        return 1;       
    }   
    // Common part is setting up input, output, weight
    // input -> N X IN
    // output -> N X OUT
    // WEIGHTS: W1 -> IN X H1, W2 -> H1 X H2, W3 -> H2 x H3, W4 -> H3 X OUT
    // BIASES: B1 -> H1, B2 -> H2, B3 -> H3, B4 -> OUT
    // BEGIN: CPU Memory Allocation
    #ifdef DEBUG_PRINT
    std::cout << "Begin: CPU Memory Allocation" << std::endl;
    #endif
    float *input = new float[N * IN];
    float *output = new float[N * OUT];
    float *W[L];
    float *B[L];
    W[0] = new float[IN * H[0]];
    W[L-1] = new float[H[L-2] * OUT];
    B[0] = new float[H[0]];
    B[L-1] = new float[OUT];
    for (int i=1; i < L-1; i++) {
        W[i] = new float[H[i] * H[i-1]];
        B[i] = new float[H[i]];
    }
    // DONE: CPU Memory Allocation 

    // BEGIN: GPU Memory Allocation
    #ifdef DEBUG_PRINT
    std::cout << "Begin: GPU Memory Allocation" << std::endl;
    #endif
    float *inputGPU = nullptr;
    float *outputGPU = nullptr;
    float *WGPU[L];
    float *BGPU[L];
    for (int i=0; i < L; i++) {
        WGPU[i] = BGPU[i] = nullptr;
    }
    cudaMalloc(&inputGPU, N * IN * sizeof(float));
    cudaMalloc(&outputGPU, N * OUT * sizeof(float));
    cudaMalloc(&WGPU[0],IN * H[0] * sizeof(float));
    cudaMalloc(&WGPU[L - 1], H[L-2] * OUT * sizeof(float));
    cudaMalloc(&BGPU[0], H[0] * sizeof(float));
    cudaMalloc(&BGPU[L-1], OUT * sizeof(float));
    for (int i=1; i < L-1; i++) {
        cudaMalloc(&WGPU[i], H[i] * H[i-1] * sizeof(float));
        cudaMalloc(&BGPU[i], H[i] * sizeof(float));
    }
    // DONE: GPU Memory Allocation
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA1 Error: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }

    // BEGIN: CPU initialization
    #ifdef DEBUG_PRINT
    std::cout << "Begin: CPU Initialization" << std::endl;
    #endif
    for (int i=0; i < N; i++) {
        for (int j=0; j < IN; j++) {
            input[i * IN + j] = (((float)(rand()))/((float)RAND_MAX));
        }
        for (int j=0; j < OUT; j++) {
            output[i * OUT + j] = (((float)(rand()))/((float)RAND_MAX));
        }
    }
    for (int l=0; l < L; l++) {
        if (l == 0) {
            for (int i = 0; i < H[0]; i++) {
                for (int j=0; j < IN; j++) {
                    W[l][j * H[0] + i] = (((float)(rand()))/((float)RAND_MAX));
                }
                B[l][i] = (((float)(rand()))/((float)RAND_MAX));
            }
        }
        else if (l < L - 1) {
            for (int i=0; i < H[l]; i++) {
                for (int j=0; j < H[l - 1]; j++) {
                    W[l][j * H[l] + i] = (((float)(rand()))/((float)RAND_MAX));
                }
                B[l][i] = (((float)(rand()))/((float)RAND_MAX));
            }
        }
        else {
            for (int i=0; i < OUT; i++) {
                for (int j=0; j < H[L-2]; j++) {
                    W[l][j * OUT + i] = (((float)(rand()))/((float)RAND_MAX));
                }
                B[l][i] = (((float)(rand()))/((float)RAND_MAX));
            }
        }
    }
    // DONE: CPU initialization

    // BEGIN: COPY TO GPU
    #ifdef DEBUG_PRINT
    std::cout << "Begin: Copy to GPU" << std::endl;
    #endif
    cudaMemcpy(inputGPU, input, N * IN * sizeof(float), cudaMemcpyDefault);
    cudaMemcpy(outputGPU, output, N * OUT * sizeof(float), cudaMemcpyDefault);
    cudaMemcpy(WGPU[0], W[0], H[0] * IN * sizeof(float), cudaMemcpyDefault);
    cudaMemcpy(WGPU[L-1], W[L-1], H[L-2] * OUT * sizeof(float), cudaMemcpyDefault);
    cudaMemcpy(BGPU[0], B[0], H[0] * sizeof(float), cudaMemcpyDefault);
    cudaMemcpy(BGPU[L-1], B[L-1], OUT * sizeof(float), cudaMemcpyDefault);
    for (int i=1; i < L-1; i++) {
        cudaMemcpy(WGPU[i], W[i], H[i-1] * H[i] * sizeof(float), cudaMemcpyDefault);
        cudaMemcpy(BGPU[i], B[i], H[i] * sizeof(float), cudaMemcpyDefault);
    }
    // DONE: COPY TO GPU

    #ifdef DEBUG_PRINT
    std::cout << "Begin: Actual computation" << std::endl;
    #endif
    int mode = std::stoi(argv[1]);
    if (mode == 0) {
        #ifdef DEBUG_PRINT
        float *Z[L-1];
        float *ZCpy[L-1];
        float *A[L-1];
        float *ACpy[L-1];
        for (int i=0; i < L-1; i++) {
            Z[i] = new float[N * H[i]];
            ZCpy[i] = new float[N * H[i]];
            A[i] = new float[N * H[i]];
            ACpy[i] = new float[N * H[i]];
        }
        #endif
        float *ZGPU[L-1]; 
        float *AGPU[L-1];
        for (int i=0; i < L-1; i++) {
            cudaMalloc(&ZGPU[i], N * H[i] * sizeof(float));
            cudaMalloc(&AGPU[i], N * H[i] * sizeof(float));
        }
        #ifdef DEBUG_PRINT
        float *pred = new float[N * OUT];
        float *predCpy = new float[N * OUT];
        #endif
        float *predGPU = nullptr;
        cudaMalloc(&predGPU, N * OUT * sizeof(float));
        fpLayerGPU(inputGPU, ZGPU[0], AGPU[0], WGPU[0], BGPU[0], N, IN, H[0]);
        #ifdef DEBUG_PRINT
        fpLayerCPU(input, Z[0], A[0], W[0], B[0], N, IN, H[0]);
        cudaMemcpy(ZCpy[0],ZGPU[0], N * H[0] * sizeof(float), cudaMemcpyDefault);
        cudaMemcpy(ACpy[0],AGPU[0], N * H[0] * sizeof(float), cudaMemcpyDefault);
        if (!checkMatch(Z[0], ZCpy[0], N, H[0], "Z0")) {exit(1);}
        if (!checkMatch(A[0], ACpy[0], N, H[0], "A0")) {exit(1);}
        #endif

        for (int i=1; i < L-1; i++) {
            fpLayerGPU(AGPU[i-1],ZGPU[i],AGPU[i],WGPU[i],BGPU[i],N,H[i-1],H[i]);
            #ifdef DEBUG_PRINT
            fpLayerCPU(A[i-1],Z[i],A[i],W[i],B[i],N,H[i-1],H[i]);
            cudaMemcpy(ZCpy[i],ZGPU[i], N * H[i] * sizeof(float), cudaMemcpyDefault);
            cudaMemcpy(ACpy[i],AGPU[i], N * H[i] * sizeof(float), cudaMemcpyDefault);
            if (!checkMatch(Z[i], ZCpy[i], N, H[i], "Z0")) {exit(1);}
            if (!checkMatch(A[i], ACpy[i], N, H[i], "A0")) {exit(1);}
            std::cout << "Layer " << i << " done" << std::endl;
            #endif
        } 
        fpLayerGPU(AGPU[L-2], predGPU, nullptr, WGPU[L-1], BGPU[L-1],N, H[L-2], OUT, false);
        #ifdef DEBUG_PRINT
        fpLayerCPU(A[L-2], pred, nullptr, W[L-1],B[L-1],N, H[L-2],OUT, false);
        cudaMemcpy(predCpy, predGPU, N * OUT * sizeof(float), cudaMemcpyDefault);
        if (!checkMatch(pred, predCpy, N, OUT, "Final")) {exit(1);}
        #endif
        #ifdef DEBUG_PRINT
        for (int i=0; i < L-1; i++) {
            delete []Z[i];
            delete []ZCpy[i];
            delete []A[i];
            delete []ACpy[i];
        }
        delete []pred;
        delete []predCpy;
        #endif
        for (int i=0; i < L-1; i++) {
            cudaFree(AGPU[i]);
            cudaFree(ZGPU[i]);
        }
        cudaFree(predGPU);
        std::cout << "Forward pass correct" << std::endl;
    }
    else if (mode == 1) {
        #ifndef DEBUG_PRINT
        assert("Only run this mode with DP on!" && false);
        #endif
        // test ATB, ABT, ATBT
        float *A = new float[N * IN];
        float *B1 = new float[IN * H[0]]; // AB1
        float *B2 = new float[H[0] * IN]; // AB2T
        float *B3 = new float[N * H[0]]; // ATB3
        float *B4 = new float[H[0] * N]; // ATB4T 
        float *C1 = new float[N * H[0]];
        float *C2 = new float[N * H[0]];
        float *C3 = new float[IN * H[0]];
        float *C4 = new float[IN * H[0]];
        float *BGPU[4] = {nullptr,nullptr,nullptr,nullptr};
        float *CGPU[4] = {nullptr,nullptr,nullptr,nullptr};
        float *AGPU = nullptr;
        float *Cres[4];
        Cres[0] = new float[N * H[0]];
        Cres[1] = new float[N * H[0]];
        Cres[2] = new float[IN * H[0]];
        Cres[3] = new float[IN * H[0]];
        cudaMalloc(&AGPU, N * IN * sizeof(float));
        cudaMalloc(&BGPU[0], IN * H[0] * sizeof(float));
        cudaMalloc(&BGPU[1], IN * H[0] * sizeof(float));
        cudaMalloc(&BGPU[2], N * H[0] * sizeof(float));
        cudaMalloc(&BGPU[3], N * H[0] * sizeof(float));
        cudaMalloc(&CGPU[0], N * H[0] * sizeof(float));
        cudaMalloc(&CGPU[1], N * H[0] * sizeof(float));
        cudaMalloc(&CGPU[2], IN * H[0] * sizeof(float));
        cudaMalloc(&CGPU[3], IN * H[0] * sizeof(float));
        for (int i=0; i < N; i++) {
            for (int j=0; j < IN; j++) {
                A[i * IN + j] = (((float)(rand()))/((float)RAND_MAX));
            }
        }
        for (int i=0; i < IN; i++) {
            for (int j=0; j < H[0]; j++) {
                B1[i * H[0] + j] = (((float)(rand()))/((float)RAND_MAX));
            }
        }
        for (int i=0; i < H[0]; i++) {
            for (int j=0; j < IN; j++) {
                B2[i * IN + j] = (((float)(rand()))/((float)RAND_MAX));
            }
        }
        for (int i=0; i < N; i++) {
            for (int j=0; j < H[0]; j++) {
                B3[i * H[0] + j] = (((float)(rand()))/((float)RAND_MAX));
            }
        }
        for (int i=0; i < H[0]; i++) {
            for (int j=0; j < N; j++) {
                B3[i * N + j] = (((float)(rand()))/((float)RAND_MAX));
            }
        }
        cudaMemcpy(BGPU[0],B1,IN * H[0] * sizeof(float), cudaMemcpyDefault);
        cudaMemcpy(BGPU[1],B2,IN * H[0] * sizeof(float), cudaMemcpyDefault);
        cudaMemcpy(BGPU[2],B3,N * H[0] * sizeof(float), cudaMemcpyDefault);
        cudaMemcpy(BGPU[3],B4,N * H[0] * sizeof(float), cudaMemcpyDefault);
        cudaMemcpy(AGPU, A, N * IN * sizeof(float), cudaMemcpyDefault);
        cpuMatMul(A,B1,C1,N,IN,H[0],false,false);
        cpuMatMul(A,B2,C2,N,IN,H[0],false,true);
        cpuMatMul(A,B3,C3,IN,N,H[0],true,false);
        cpuMatMul(A,B4,C4,IN,N,H[0],true,true);
        dim3 block(32,32);
        dim3 grid1(((H[0] + block.x - 1)/block.x), ((N + block.y - 1)/block.y));
        dim3 grid2(((H[0] + block.x - 1)/block.x), ((IN + block.y - 1)/block.y));
        matmul<<<grid1, block>>>(AGPU, BGPU[0], CGPU[0], N, IN, H[0], false, false);
        matmul<<<grid1, block>>>(AGPU, BGPU[1], CGPU[1], N, IN, H[0], false, true);
        matmul<<<grid2, block>>>(AGPU, BGPU[2], CGPU[2], IN, N, H[0], true, false);
        matmul<<<grid2, block>>>(AGPU, BGPU[3], CGPU[3], IN, N, H[0], true, true);
        cudaMemcpy(Cres[0], CGPU[0], N * H[0] * sizeof(float), cudaMemcpyDefault);
        cudaMemcpy(Cres[1], CGPU[1], N * H[0] * sizeof(float), cudaMemcpyDefault);
        cudaMemcpy(Cres[2], CGPU[2], IN * H[0] * sizeof(float), cudaMemcpyDefault);
        cudaMemcpy(Cres[3], CGPU[3], IN * H[0] * sizeof(float), cudaMemcpyDefault);
        if (!checkMatch(Cres[0],C1,N,H[0],"NT,NT")) {exit(1);}
        if (!checkMatch(Cres[1],C2,N,H[0],"NT,T")) {exit(1);}
        if (!checkMatch(Cres[2],C3,IN,H[0],"T,NT")) {exit(1);}
        if (!checkMatch(Cres[3],C4,IN,H[0],"T,T")) {exit(1);}
        std::cout << "Transpose matmul working" << std::endl;
    }
    else if (mode == 2) {
        #ifdef DEBUG_PRINT
        float *Z[L-1];
        float *ZCpy[L-1];
        float *A[L-1];
        float *ACpy[L-1];
        for (int i=0; i < L-1; i++) {
            Z[i] = new float[N * H[i]];
            ZCpy[i] = new float[N * H[i]];
            A[i] = new float[N * H[i]];
            ACpy[i] = new float[N * H[i]];
        }
        #endif
        float *ZGPU[L-1]; 
        float *AGPU[L-1];
        for (int i=0; i < L-1; i++) {
            cudaMalloc(&ZGPU[i], N * H[i] * sizeof(float));
            cudaMalloc(&AGPU[i], N * H[i] * sizeof(float));
        }
        #ifdef DEBUG_PRINT
        float *pred = new float[N * OUT];
        float *predCpy = new float[N * OUT];
        #endif
        float *predGPU = nullptr;
        cudaMalloc(&predGPU, N * OUT * sizeof(float));
        fpLayerGPU(inputGPU, ZGPU[0], AGPU[0], WGPU[0], BGPU[0], N, IN, H[0]);
        #ifdef DEBUG_PRINT
        fpLayerCPU(input, Z[0], A[0], W[0], B[0], N, IN, H[0]);
        cudaMemcpy(ZCpy[0],ZGPU[0], N * H[0] * sizeof(float), cudaMemcpyDefault);
        cudaMemcpy(ACpy[0],AGPU[0], N * H[0] * sizeof(float), cudaMemcpyDefault);
        if (!checkMatch(Z[0], ZCpy[0], N, H[0], "Z0")) {exit(1);}
        if (!checkMatch(A[0], ACpy[0], N, H[0], "A0")) {exit(1);}
        #endif

        for (int i=1; i < L-1; i++) {
            fpLayerGPU(AGPU[i-1],ZGPU[i],AGPU[i],WGPU[i],BGPU[i],N,H[i-1],H[i]);
            #ifdef DEBUG_PRINT
            fpLayerCPU(A[i-1],Z[i],A[i],W[i],B[i],N,H[i-1],H[i]);
            cudaMemcpy(ZCpy[i],ZGPU[i], N * H[i] * sizeof(float), cudaMemcpyDefault);
            cudaMemcpy(ACpy[i],AGPU[i], N * H[i] * sizeof(float), cudaMemcpyDefault);
            if (!checkMatch(Z[i], ZCpy[i], N, H[i], "Z0")) {exit(1);}
            if (!checkMatch(A[i], ACpy[i], N, H[i], "A0")) {exit(1);}
            std::cout << "Layer " << i << " done" << std::endl;
            #endif
        } 
        fpLayerGPU(AGPU[L-2], predGPU, nullptr, WGPU[L-1], BGPU[L-1],N, H[L-2], OUT, false);
        #ifdef DEBUG_PRINT
        fpLayerCPU(A[L-2], pred, nullptr, W[L-1],B[L-1],N, H[L-2],OUT, false);
        cudaMemcpy(predCpy, predGPU, N * OUT * sizeof(float), cudaMemcpyDefault);
        if (!checkMatch(pred, predCpy, N, OUT, "Final")) {exit(1);}
        #endif
        std::cout << "Forward pass correct" << std::endl;

        // Now for backward pass
        #ifdef DEBUG_PRINT
        float *dW[L];
        float *dWCpy[L];
        float *dB[L];
        float *dBCpy[L];
        
        float *dZ[L-1];
        float *dZCpy[L-1];
        float *dA[L-1];
        float *dACpy[L-1];
        
        float *dpred;
        float *dpredCpy;
        #endif 

        float *dWGPU[L];
        float *dBGPU[L];
        float *dZGPU[L-1];
        float *dAGPU[L-1];
        float *dpredGPU;
        
        #ifdef DEBUG_PRINT
        dW[0] = new float[IN * H[0]];
        dWCpy[0] = new float[IN * H[0]];
        dW[L-1] = new float[H[L-2] * OUT];
        dWCpy[L-1] = new float[H[L-2] * OUT];
        dB[0] = new float[H[0]];
        dBCpy[0] = new float[H[0]];
        dB[L-1] = new float[OUT];
        dBCpy[L-1] = new float[OUT];
        #endif

        cudaMalloc(&dWGPU[0], IN * H[0] * sizeof(float));
        cudaMalloc(&dWGPU[L-1], H[L-2] * OUT * sizeof(float));
        cudaMalloc(&dBGPU[0], H[0] * sizeof(float));
        cudaMalloc(&dBGPU[L-1], OUT * sizeof(float));
        for (int i=1; i < L - 1; i++) {
            #ifdef DEBUG_PRINT
            dW[i] = new float[H[i-1] * H[i]];
            dWCpy[i] = new float[H[i-1] * H[i]];
            dB[i] = new float[H[i]];
            dBCpy[i] = new float[H[i]];
            #endif
            cudaMalloc(&dWGPU[i], H[i-1] * H[i] * sizeof(float));
            cudaMalloc(&dBGPU[i], H[i] * sizeof(float));
        }
        for (int i=0; i < L-1; i++) {
            #ifdef DEBUG_PRINT
            dZ[i] = new float[N * H[i]];
            dZCpy[i] = new float[N * H[i]];
            dA[i] = new float[N * H[i]];
            dACpy[i] = new float[N * H[i]];
            #endif
            cudaMalloc(&dZGPU[i], N * H[i] * sizeof(float));
            cudaMalloc(&dAGPU[i], N * H[i] * sizeof(float));
        }
        #ifdef DEBUG_PRINT
        dpred = new float[N * OUT];
        dpredCpy = new float[N * OUT];
        #endif
        cudaMalloc(&dpredGPU, N * OUT * sizeof(float));

        // CPU Gradient Descent
        #ifdef DEBUG_PRINT
        initBPCPU(pred, output, dpred, N, OUT);
        lastLayerBPCPU(dpred, A[L-2], W[L-1], dW[L-1], dB[L-1], dA[L-2],N, OUT, H[L-2]);
        for (int l = L - 2; l >= 0; l--) {
            if (l != 0) {
                bpLayerICPU(dA[l],dZ[l],dW[l],dB[l],dA[l-1],Z[l],A[l-1],W[l],N,H[l],H[l-1]);
            }
            else {
                bpLayerICPU(dA[l],dZ[l],dW[l],dB[l],nullptr,Z[l],input,W[l],N,H[l],IN);
            }
        }
        #endif

        // GPU Gradient Descent
        dim3 block(32, 32);
        dim3 grid1(((OUT + block.x - 1)/block.x), ((N + block.y - 1)/block.y));
        bpInit<<<grid1, block>>>(predGPU, outputGPU, dpredGPU, OUT, N);
        #ifdef DEBUG_PRINT
        cudaMemcpy(dpredCpy, dpredGPU, N * OUT * sizeof(float), cudaMemcpyDefault);
        if (!checkMatch(dpred, dpredCpy, N, OUT, "dPred")) {exit(1);}
        lastLayerBPGPU(dpredGPU, AGPU[L-2], WGPU[L-1],dWGPU[L-1],dBGPU[L-1],dAGPU[L-2],N, OUT, H[L-2]);
        cudaMemcpy(dWCpy[L-1], dWGPU[L-1], H[L-2] * OUT * sizeof(float), cudaMemcpyDefault);
        if (!checkMatch(dW[L-1],dWCpy[L-1],H[L-2], OUT, "dW[L-1]")) {exit(1);}
        cudaMemcpy(dBCpy[L-1], dBGPU[L-1], OUT * sizeof(float), cudaMemcpyDefault);
        if (!checkMatch(dB[L-1],dBCpy[L-1],1, OUT, "dB[L-1]")) {exit(1);}
        cudaMemcpy(dACpy[L-2], dAGPU[L-2], N * H[L-2] * sizeof(float), cudaMemcpyDefault);
        if (!checkMatch(dA[L-2],dACpy[L-2],N, H[L-2], "dA[L-2]")) {exit(1);}
        #endif
        for (int l = L-2; l >= 0; l--) {
            if (l != 0) {
                bpLayerIGPU(dAGPU[l], dZGPU[l], dWGPU[l], dBGPU[l], dAGPU[l-1], ZGPU[l], AGPU[l-1], WGPU[l], N, H[l], H[l-1]);
            }
            else {
                bpLayerIGPU(dAGPU[l], dZGPU[l], dWGPU[l], dBGPU[l],nullptr, ZGPU[l], inputGPU, WGPU[l], N, H[l], IN);
            }
            #ifdef DEBUG_PRINT
            cudaMemcpy(dZCpy[l], dZGPU[l], sizeof(float) * N * H[l], cudaMemcpyDefault);
            if (!checkMatch(dZ[l], dZCpy[l],N, H[l], "dZ[l]")) {exit(1);}
            if (l != 0) {
                cudaMemcpy(dWCpy[l], dWGPU[l], H[l-1] * H[l] * sizeof(float), cudaMemcpyDefault);
                if (!checkMatch(dW[l], dWCpy[l], H[l-1], H[l], "dW[l]")) {exit(1);}
            }
            else {
                cudaMemcpy(dWCpy[l], dWGPU[l], IN * H[l] * sizeof(float), cudaMemcpyDefault);
                if (!checkMatch(dW[l], dWCpy[l], IN, H[l], "dW[l]")) {exit(1);}
            }
            cudaMemcpy(dBCpy[l], dBGPU[l], H[l] * sizeof(float), cudaMemcpyDefault);
            if (!checkMatch(dB[l], dBCpy[l], H[l],1, "dB[l]")) {exit(1);}
            if (l != 0) {
                cudaMemcpy(dACpy[l-1], dAGPU[l-1], N * H[l-1] * sizeof(float), cudaMemcpyDefault);
                if (!checkMatch(dA[l-1], dACpy[l-1], N, H[l-1], "dA[l-1]")) {exit(1);}
            }
            #endif
        }
        std::cout << "Gradient calculation correct!" << std::endl;

        #ifdef DEBUG_PRINT
        for (int i=0; i < L-1; i++) {
            delete []Z[i];
            delete []ZCpy[i];
            delete []A[i];
            delete []ACpy[i];
        }
        delete []pred;
        delete []predCpy;
        #endif
        for (int i=0; i < L-1; i++) {
            cudaFree(AGPU[i]);
            cudaFree(ZGPU[i]);
        }
        cudaFree(predGPU);
        for (int i=0; i < L; i++) {
            #ifdef DEBUG_PRINT
            delete []dW[i];
            delete []dWCpy[i];
            delete []dB[i];
            delete []dBCpy[i];
            #endif
            cudaFree(dWGPU[i]);
            cudaFree(dBGPU[i]);
        }
        for (int i=0; i < L-1; i++) {
            #ifdef DEBUG_PRINT
            delete []dZ[i];
            delete []dZCpy[i];
            delete []dA[i];
            delete []dACpy[i];
            #endif
            cudaFree(dZGPU[i]);
            cudaFree(dAGPU[i]);
        }
        #ifdef DEBUG_PRINT
        delete []dpred;
        delete []dpredCpy;
        #endif
        cudaFree(dpredGPU);

    }
    delete []input;
    delete []output;
    for (int i=0; i < L; i++) {
        delete []W[i];
        delete []B[i];
    }
    cudaFree(inputGPU);
    cudaFree(outputGPU);
    for (int i=0; i < L; i++) {
        cudaFree(WGPU[i]);
        cudaFree(BGPU[i]);
    }

}