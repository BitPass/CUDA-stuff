/* 
Simples Possible Cuda-like Cuda Program 
Vector Add 
For clarity, no error checking is added
*/

#include <stdio.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < numElements) {
		C[i] = A[i] + B[i];
	}
}

int main(void)
{
	int numElements = 50000;     // Print the vector length to be used, and compute its size
	size_t size = numElements * sizeof(float);
	printf("[Vector addition of %d elements]\n", numElements);
	float *h_A = (float *)malloc(size);     // Allocate the host input vector A
	float *h_B = (float *)malloc(size);     // Allocate the host input vector B
	float *h_C = (float *)malloc(size);     // Allocate the host output vector C
	for (int i = 0; i < numElements; ++i) /* Initialize the host input vectors*/ {
		h_A[i] = rand() / (float)RAND_MAX;
		h_B[i] = rand() / (float)RAND_MAX;
	}

	printf("%f\n", h_A[1]);
	printf("%f\n", h_B[1]);

	float *d_A = NULL; // Allocate the device input vector A
	cudaMalloc((void **)&d_A, size);
	float *d_B = NULL;// Allocate the device input vector B
	cudaMalloc((void **)&d_B, size);
	float *d_C = NULL; // Allocate the device output vector C
	cudaMalloc((void **)&d_C, size);
	printf("Copy input data from the host memory to the CUDA device\n"); /* Copy the host input vectors A and B in host memory to the device input vectors in device memory */
	cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

	int threadsPerBlock = 256;
	int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
	printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
	vectorAdd <<<blocksPerGrid, threadsPerBlock >>>(d_A, d_B, d_C, numElements); // Launch the Vector Add CUDA Kernel
	printf("Copy output data from the CUDA device to the host memory\n");
	cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost); /* Copy the device result vector in device memory to the host result vector in host memory.*/

	printf("%f\n", h_C[1]);

	cudaFree(d_A); // Free device global memory
	cudaFree(d_B);
	cudaFree(d_C);
	free(h_A); // Free host memory
	free(h_B);
	free(h_C);
	printf("Done\n");
	return 0;
}

