#include <stdio.h>
#include <iostream>

#include <omp.h>

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
	T 				*h_data;
};

/*********************/
/* SVD PLAN CREATION */
/*********************/
template<class T>
void createPlan(plan<T>& plan, unsigned int NperGPU, unsigned int gpuID) {

    // --- Device allocation
    gpuErrchk(cudaSetDevice(gpuID));
	gpuErrchk(cudaMalloc(&(plan.d_data), NperGPU * sizeof(T)));
	gpuErrchk(cudaMallocHost((void **)&plan.h_data, NperGPU * sizeof(T)));
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

    float time[numGPUs];
    cudaEvent_t start[numGPUs], stop[numGPUs];

	int maxNumProcessors = omp_get_max_threads();
	std::cout << "Maximum number of CPU threads = " << maxNumProcessors << std::endl;

    // --- "Breadth-first" approach - async
//	omp_set_num_threads(numGPUs);
//	#pragma omp parallel
//	{
//		unsigned int k = omp_get_thread_num();
//		gpuErrchk(cudaSetDevice(k));
// 	    cudaEventCreateWithFlags(&start[k], cudaEventBlockingSync);
// 	    cudaEventCreateWithFlags(&stop[k], cudaEventBlockingSync);
// 	    cudaEventRecord(start[k], 0);
// 		gpuErrchk(cudaMemcpyAsync(plan[k].d_data, plan[k].h_data, NperGPU * sizeof(double), cudaMemcpyHostToDevice));
// 		cudaEventRecord(stop[k], 0);
// 		cudaEventSynchronize(stop[k]);
// 		cudaEventElapsedTime(&time[k], start[k], stop[k]);
// 		printf("Thread nr. %i; Elapsed time:  %3.1f ms \n", k, time[k]);
// 	}

    // --- "Breadth-first" approach - async
// 	for (int k = 0; k < numGPUs; k++) {
// 		gpuErrchk(cudaSetDevice(k));
// 	    cudaEventCreateWithFlags(&start[k], cudaEventBlockingSync);
// 	    cudaEventCreateWithFlags(&stop[k], cudaEventBlockingSync);
// 	    cudaEventRecord(start[k], 0);
// 		gpuErrchk(cudaMemcpyAsync(plan[k].d_data, plan[k].h_data, NperGPU * sizeof(double), cudaMemcpyHostToDevice));
// 		cudaEventRecord(stop[k], 0);
// 		cudaEventSynchronize(stop[k]);
// 		cudaEventElapsedTime(&time[k], start[k], stop[k]);
// 	}
//	for (int k = 0; k < numGPUs; k++) printf("Elapsed time:  %3.1f ms \n", time[k]);

    // --- "Breadth-first" approach - async
 	for (int k = 0; k < numGPUs; k++) {
 		gpuErrchk(cudaSetDevice(k));
 		gpuErrchk(cudaMemcpyAsync(plan[k].d_data, plan[k].h_data, NperGPU * sizeof(double), cudaMemcpyHostToDevice));
 	}

//	for (int k = 0; k < numGPUs; k++) {
// 		gpuErrchk(cudaSetDevice(k));
// 	    cudaEventCreateWithFlags(&start[k], cudaEventBlockingSync);
// 	    cudaEventCreateWithFlags(&stop[k], cudaEventBlockingSync);
// 	    cudaEventRecord(start[k], 0);
// 		kernelFunction<<<iDivUp(NperGPU, BLOCKSIZE), BLOCKSIZE>>>(plan[k].d_data, NperGPU);
// 		cudaEventRecord(stop[k], 0);
// 		cudaEventSynchronize(stop[k]);
// 		cudaEventElapsedTime(&time[k], start[k], stop[k]);
// 	}
// 	for (int k = 0; k < numGPUs; k++) printf("Elapsed time:  %3.1f ms \n", time[k]);

// 	omp_set_num_threads(numGPUs);
// 	#pragma omp parallel
// 	{
// 		unsigned int k = omp_get_thread_num();
// 		gpuErrchk(cudaSetDevice(k));
// 	    cudaEventCreateWithFlags(&start[k], cudaEventBlockingSync);
// 	    cudaEventCreateWithFlags(&stop[k], cudaEventBlockingSync);
// 	    cudaEventRecord(start[k], 0);
// 		kernelFunction<<<iDivUp(NperGPU, BLOCKSIZE), BLOCKSIZE>>>(plan[k].d_data, NperGPU);
// 		cudaEventRecord(stop[k], 0);
// 		cudaEventSynchronize(stop[k]);
// 		cudaEventElapsedTime(&time[k], start[k], stop[k]);
// 		printf("Thread nr. %i; Elapsed time:  %3.1f ms \n", k, time[k]);
// 	}

 	for (int k = 0; k < numGPUs; k++) {
 		gpuErrchk(cudaSetDevice(k));
 		kernelFunction<<<iDivUp(NperGPU, BLOCKSIZE), BLOCKSIZE>>>(plan[k].d_data, NperGPU);
 	}

 	for (int k = 0; k < numGPUs; k++) {
 		gpuErrchk(cudaSetDevice(k));
 		gpuErrchk(cudaMemcpyAsync(plan[k].h_data, plan[k].d_data, NperGPU * sizeof(double), cudaMemcpyDeviceToHost));
 	}

    gpuErrchk(cudaDeviceReset());
}

