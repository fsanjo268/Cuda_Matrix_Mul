%%cu

/**
 * Matrix Multiplication: C = A * B.
 *
 * This file contains both device and host code to compute a matrix multiplication.
 * made by:
 * Fernando San José Domínguez
 * Mario Pereda Puyo
 */

#include <stdio.h>
#include <math.h>

#define MATRIX_DIM	 64
#define SEGMENT_SIZE 32

// --------------------
// Device Kernels
// --------------------
__global__ void transposeMatrix(float *d_data, int dim_x) {
	  int x = blockIdx.x * blockDim.x + threadIdx.x;//calculate id_x thread in grid 
    int y = blockIdx.y * blockDim.y + threadIdx.y;//calculate id_y thread in grid

    if (x < dim_x && y < dim_x) {
        int index_in = y * dim_x + x;//calculate position of the element to transpose
        int index_out = x * dim_x + y;//calculate the new position of the element in transpose matrix
        float temp = d_data[index_out];//save the element that will be remplace with the element to transpose first
        d_data[index_out] = d_data[index_in];//swap the element to transpose in old position in to the new one
        d_data[index_in] = temp;//set the value that we overwritte in the position of the element to traspose
    }
	  
}

__global__ void scalarProd(float *C, const float *A, const float *B, int nElem) {

	int tid = threadIdx.x + blockIdx.x * blockDim.x;//threads id in unidimensional grid
    
    if (tid < nElem) {
        C[tid] = A[tid] * B[tid];
    }
}

__global__ void vectorReduce(float *R, const float *C, int nElem) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;//thread id
    unsigned int i = blockIdx.x * blockDim.x  + threadIdx.x;//thread id in grid
    sdata[tid] = C[i] + C[i + blockDim.x];//set the values of C in sdata
    __syncthreads();

    // do reduction in shared mem
    for(unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset && (tid + offset) < nElem) {
            sdata[tid] += sdata[tid + offset];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if(tid == 0){
      R[blockIdx.x] = sdata[0];
    }
    __syncthreads();
}

__global__ void vectorReduceOneBlock(float *R, const float *C, int nElem) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x; //thread id
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;//thread id in grid

    //set the values of C in sdata
    sdata[tid] = (i < nElem) ? C[i] : 0.0f;
    sdata[tid + blockDim.x] = (i + blockDim.x < nElem) ? C[i + blockDim.x] : 0.0f;
    __syncthreads();

    // do reduction in shared mem in steps
    for (unsigned int offset = blockDim.x; offset > 0; offset >>= 1) {
        if (tid < offset && i + offset < nElem) {
            sdata[tid] += sdata[tid + offset];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) {
        R[blockIdx.x] = sdata[0];
    }
}

// ---------------------
// Host Utility Routines
// ---------------------
void matrixMul(const float *A, const float *B, float *C, const int n)
{
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			float acum = 0.0f;
			for (int k = 0; k < n; k++) {
				acum += A[i * n + k] * B[k * n + j];
			}
			C[i * n + j] = acum;
		}
	}
}

bool compareData(float *h_C, float *d_C, int n)
{
	double eps = 1.E-6;//must change that value because the matrix carries decimal errors in some values
	for (int i = 0; i < n * n; i++) {
		if (fabsf(h_C[i] - d_C[i]) > eps) {
			return false;
		}
	}
	return true;
}

float randFloat(float low, float high) {
	float t = (float) rand() / (float) RAND_MAX;
	return (1.0f - t) * low + (t * high);
}

// ------------
// Main Program
// ------------

int main (void) {
  // Matrix Dimensions
  int dim_x = MATRIX_DIM;
  int dim_y = dim_x;

  // Matrix Size
  int mat_size = dim_x * dim_y;

  // Block Dimension
  int block_dim = SEGMENT_SIZE;

  // Number of Blocks
   int n_block = (dim_x % block_dim == 0) ? (dim_x / block_dim) : (dim_x / block_dim) + 1;

  // Execution Configuration Parameters
  dim3 blocksPerGrid(n_block, n_block);
  dim3 threadsPerBlock(block_dim, block_dim);

  // Size Required to Store the Matrix
  size_t n_bytes = (mat_size * sizeof(float));

  // Allocate Pinned Host Memory
  float *h_A, *h_B, *h_C, *h_R;

  cudaMallocHost(&h_A, n_bytes);
  cudaMallocHost(&h_B, n_bytes);
  cudaMallocHost(&h_C, n_bytes);
  cudaMallocHost(&h_R, n_bytes);

  // Initialize Host Data
  srand(123);

  // Generating input data on CPU
  for (int i=0; i < mat_size; i++) {
		h_A[i] = randFloat(0.0f, 1.0f);
		h_B[i] = randFloat(0.0f, 1.0f);
	}

  // Compute Reference Matrix Multiplication
  matrixMul(h_A, h_B, h_R, dim_x);

  // CUDA Streams
	cudaStream_t stream;

  // Create Stream
	cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

  // Performance Data
  float kernel_time, kernel_bandwidth;

  // Allocate Device Memory
  float *d_A, *d_B, *d_C, *d_D;

  cudaMalloc(&d_A, n_bytes);
  cudaMalloc(&d_B, n_bytes);
  cudaMalloc(&d_C, n_bytes);
  cudaMalloc(&d_D, n_bytes);

  // CUDA Events
  cudaEvent_t start, stop;

  // Start Time Measurement
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, stream);

  // Copy Host Data to Device
  cudaMemcpyAsync(d_A, h_A, n_bytes, cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_B, h_B, n_bytes, cudaMemcpyHostToDevice, stream);

  cudaStreamSynchronize(stream);

  transposeMatrix<<<blocksPerGrid, threadsPerBlock>>>(d_B, dim_x);

  cudaMemcpy(h_A, d_B, n_bytes, cudaMemcpyDeviceToHost);

  //now we have one d_B per block that used the previous kernel
  for (int i = 0; i < dim_y; i++) {
      for (int j = 0; j < dim_x; j++) {

            // Obtain a scalar vector from the product of columns of d_A and lines of d_B and store it in d_C
            scalarProd<<<n_block, block_dim>>>(d_C + i * dim_x, d_A + i * dim_x, d_B + j * dim_x, dim_x);
            

            cudaStreamSynchronize(stream);

            if(block_dim!=dim_x ){
            // Stores in i line and j column of d_D the reduced vector obtained from line i in d_C 
            vectorReduce<<<n_block, block_dim, block_dim*sizeof(float)>>>(d_D + i * dim_x + j, d_C + i * dim_x, dim_x);
            
            }else{
                // Stores in i line and j column of d_D the reduced vector obtained from line i in d_C 
                vectorReduceOneBlock<<<n_block, block_dim, (block_dim*2)*sizeof(float)>>>(d_D + i * dim_x + j, d_C + i * dim_x, dim_x);
                
            }
         
        }
  }


  cudaDeviceSynchronize();

  cudaStreamSynchronize(stream);

  // Copy Device Data to Host
  cudaMemcpyAsync(h_A, d_C, n_bytes, cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(h_C, d_D, n_bytes, cudaMemcpyDeviceToHost, stream);

  // End Time Measurement
  cudaEventRecord(stop, stream);
  cudaEventSynchronize(stop);

  cudaEventElapsedTime(&kernel_time, start, stop);
 
  bool res = compareData(h_C, h_R, dim_x);

  if (res == true) {
      // Report Effective Bandwidth
      kernel_bandwidth = (2.0f * 1000.0f * n_bytes) / (1024 * 1024 * 1024);
      kernel_bandwidth /= kernel_time;

      printf("Throughput = %.4f GB/s, Time = %.5f ms, Size = %u fp32 elements, \n",
              kernel_bandwidth, kernel_time, (dim_x * dim_y));
  }


  // Free Host Memory
  cudaFreeHost(h_A);
  cudaFreeHost(h_B);
  cudaFreeHost(h_C);
  cudaFreeHost(h_R);

  // Free Device Memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaFree(d_D);

  // Destroy Events
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  
  // Destroy Stream
	cudaStreamDestroy(stream);

  if (res == false) {
      printf("Test Failed!\n");
      exit(EXIT_FAILURE);
  }
  printf("Test Passed\n");
  exit(EXIT_SUCCESS);
}
   
