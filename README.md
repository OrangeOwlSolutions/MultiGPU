# MultiGPU
See [Concurrency in CUDA multi-GPU executions](http://stackoverflow.com/questions/11673154/concurrency-in-cuda-multi-gpu-executions/35010019#35010019).
- ```MultiGPU_Test1.cu```: Breadth-first - synchronous copy;
- ```MultiGPU_Test2.cu```: Depth-first   - synchronous copy;
- ```MultiGPU_Test3.cu```: Depth-first   - asynchronous copy with streams;
- ```MultiGPU_Test4.cu```: Depth-first   - asynchronous copy no   streams;
- ```MultiGPU_Test5.cu```: Depth-first   - asynchronous copy no   streams unique host vector;
- ```MultiGPU_Test6.cu```: Breadth-first - asynchronous copy with streams;
- ```MultiGPU_Test7.cu```: Breadth-first - asynchronous copy no   streams;
- ```MultiGPU_Test8.cu```: Breadth-first - asynchronous copy no   streams unique host vector;
