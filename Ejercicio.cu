#include <iostream>
#include <cuda_runtime.h>
#include <iomanip>
#include <chrono>
#include <thread>
#include <vector>
#include <numeric>
#include "utils.h"

using namespace std::chrono;
#define THREADS 512
#define BLOCKS	min(32, ((SIZE / THREADS) + 1))
#define SIZE 1000000000 

const int TESTS = 10; 

double total_par_time = 0;
long resultado_total = 0;


__global__ void count_evens_parallel(int *array, unsigned long long *acum){
    int index = threadIdx.x + (blockIdx.x * blockDim.x);

    while (index < SIZE) {
        if (array[index] % 2 == 0){
            atomicAdd(acum, 1);
        }
        index += (blockDim.x * gridDim.x);
    }
}   
    
int main(){
    int *h_array;       
    int *d_array;      
    unsigned long long *d_acum;    
    unsigned long long h_acum;         

    high_resolution_clock::time_point startTime, endTime;
    double ms_par;

    h_array = new int[SIZE];
    fill_array(h_array, SIZE);
    display_array("Arreglo: ", h_array);

    cudaMalloc((void**) &d_array, SIZE * sizeof(int));
    cudaMalloc((void**) &d_acum, sizeof(long));

    cudaMemcpy(d_array, h_array, SIZE * sizeof(int), cudaMemcpyHostToDevice);

    std::cout << "Version paralela CUDA..." << std::endl;

    for(int i = 0; i < TESTS; i++){
        cudaMemset(d_acum, 0, sizeof(long));

        startTime = high_resolution_clock::now();

        count_evens_parallel<<<BLOCKS, THREADS>>>(d_array, d_acum);

        endTime = high_resolution_clock::now();

        cudaMemcpy(&h_acum, d_acum, sizeof(long), cudaMemcpyDeviceToHost);

        ms_par = duration<double, std::milli>(endTime - startTime).count();
        total_par_time += ms_par;

        std::cout << "Total pares (CUDA): " << h_acum 
                  << "  Prueba " << i + 1 << " -> " << ms_par << " ms" << std::endl;
    }

    // 4. Liberar memoria
    cudaFree(d_array);
    cudaFree(d_acum);
    delete[] h_array;

    return 0;
}