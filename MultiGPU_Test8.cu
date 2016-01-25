#include "Utilities.cuh"
#include "InputOutput.cuh"

#define BLOCKSIZE 128

/*******************/
/* KERNEL FUNCTION */
/*******************/
template<class T>
__global__ void kernelFunction(T * __restrict__ d_data, const unsigned int NperGPU) {

	const int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid < NperGPU) for (int k = 0; k < 1000; k++) d_data[tid] = d_data[tid] * d_data[tid];

}

/******************/
/* PLAN STRUCTURE */
/******************/
// --- Async
template<class T>
struct plan {
	T 				*d_data;
};

/*********************/
/* SVD PLAN CREATION */
/*********************/
template<class T>
void createPlan(plan<T>& plan, unsigned int NperGPU, unsigned int gpuID) {

    // --- Device allocation
    gpuErrchk(cudaSetDevice(gpuID));
	gpuErrchk(cudaMalloc(&(plan.d_data), NperGPU * sizeof(T)));
}

/********/
/* MAIN */
/********/
int main() {

    const int numGPUs 	= 4;
    const int NperGPU 	= 500000;
    const int N			= NperGPU * numGPUs;

    plan<double> plan[numGPUs];
    for (int k = 0; k < numGPUs; k++) createPlan(plan[k], NperGPU, k);

    // --- "Breadth-first" approach - async
    double *inputMatrices;   gpuErrchk(cudaMallocHost(&inputMatrices, N * sizeof(double)));
	for (int k = 0; k < numGPUs; k++) {
 		gpuErrchk(cudaSetDevice(k));
 		gpuErrchk(cudaMemcpyAsync(plan[k].d_data, inputMatrices + k * NperGPU, NperGPU * sizeof(double), cudaMemcpyHostToDevice));
 	}

 	for (int k = 0; k < numGPUs; k++) {
 		gpuErrchk(cudaSetDevice(k));
 		kernelFunction<<<iDivUp(NperGPU, BLOCKSIZE), BLOCKSIZE>>>(plan[k].d_data, NperGPU);
 	}

 	for (int k = 0; k < numGPUs; k++) {
 		gpuErrchk(cudaSetDevice(k));
 		gpuErrchk(cudaMemcpyAsync(inputMatrices + k * NperGPU, plan[k].d_data, NperGPU * sizeof(double), cudaMemcpyDeviceToHost));
 	}

    gpuErrchk(cudaDeviceReset());
}

