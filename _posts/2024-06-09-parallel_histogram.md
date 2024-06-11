---
layout: post
title: Parallel Histogram
date: 2024-06-09
category: CUDA
tags:
- cuda
redirect_from:
- /cuda/2024/06/09/parallel_histogram/
- /parallel_histogram.html
---

### **Introduction**
The aim of the blog posts is to introduce parallel histogram pattern, where each output elements can be updated by any thread. Therefore, we should co-ordinate among threads as they update the output value. In this blog post we will read the introduction about the uses of atomic operations to serialize the updates of each element. Then we will study about privatization, an optmization technique. Let's dig in!

This blog post is written while reading the
ninth chapter, Parallel Histogram: An Introduction
to atomic operations ad privatization,
of the incredible book "Programming Massively Parallel
Processors: A Hands-on Approach<sup>[1](#link1)</sup>"
by [Wen-mei W. Hwu](https://scholar.google.com/citations?user=ohjQPx8AAAAJ&hl=en),
[David B. Kirk](https://scholar.google.com/citations?user=fMbArPwAAAAJ&hl=en),
and [Izzat El Hajj](https://scholar.google.com/citations?user=_VVw504AAAAJ&hl=en).

### **Background**

### **Atomic operations and a basic histogram kernel**
To parallelize histogram computation launch same number of threads as the number of data such that each thread has one input element. Each threads reads the assigned input and increment the appropritate counter. When the multiple threads increases the same elements it is known as output interference. Programmers should handle these. The operations perform read, write and modify. If any undesirable outcomes are caused, it is known as *read-modify-write race condition*; when two or more threads competes to atain same results. 

An atomic operation on a memory location is an operation that performs a read-modify-write sequence on the memory location in such a way that no other read-modify-write sequence to the location can overlap with it. Read, modify, and write cannot be divided further therefore named as atomic operation. Various atomic operations are addition, subtraction, increment, decrement, minimum, maximum, logical and, and logical or. To perform atomic add operation in CUDA:
```cuda
// intrinsic function
int atomicAdd(int* address, int val);
```
Note that intrinsic function are compiled into hardware atomic operation instruction. All major compilers support them.

A CUDA kernel performing a parallel histogam computation is given below:
```cuda
__global__ void histo_kernel(char* data, unsigned int length, unsigned int* histo) {
    // thread index calculation
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < length) {
        int alphabet_position = data[1] - 'a';
        if (alphabet_position >= 0 && alphabet_position < 26) {
            atomicAdd(&(histo[alphabet_position/4]), 1);
        } 
    }
}
```

### **Latency and throughput of atomic operations**
Note that it should be very clear that the many DRAM accesses results in high memory access throughput. This sometimes break down when atomic operations update the same memory location. This can be resolved by starting a new read-modify-write sequence once the previous read-modify-write-sequence is completed. Only one atomic operation execute at the same memory location at a time. This duration is approximately the latency of a memory load and latency of a memory store. The length of these time section is the minimum amount of time dedicated for each atomic operation. To improve the throughput of atomic operations we could reduce the access latency to the heavily contended locations. Cache memories are primary tools to reduce memory access latency.

### **Privatization**
Privatization is the process to replicate output data into private copies so that each subset of thread can update its private copy. Main benefit is it has low latency and it increases the throughput. But private copies need to be merged into the original data structure after the computation completes. Therefore, privatization is done for a group of threads rather than an individual threads.
```cuda
// creates a private copy of histogram to every block
// these private copy will be cached in the L2 cache memory
__global__ void histo_private_kernel(char *data, unsigned int length, unsigned int *histo) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

    if (i < length) {
        int alphabet_position = data[i] - 'a';
        if (alphabet_position >= 0 && alphabet_position < 26) {
            atomicAdd(&(histo[blockIdx.x*NUM_BINS + alphabet_position/4]), 1);
        }
    }

    // commit the values in private copy into the block
    if (blockIdx.x > 0) {
        __syncthreads();
        // responsible for creating one or more histogram bins
        for (unsigned int bin=threadIdx.x; bin<NUM_BINS; bin += blockDim.x) {
            unsigned int binValue = histo[blockIdx.x*NUM_BINS + bin];
            if (binValue > 0) {
                atomicAdd(&(histo[bin]), binValue);
            }
        }
    }
}
```
Using shared memory allocation:
```cuda
// creates a private copy of histogram to every block
// these private copy will be cached in the L2 cache memory
__global__ void histo_private_kernel(char *data, unsigned int length, unsigned int *histo) {

    // privatized bins
    __shared__ unsigned int histo_s[NUM_BINS];
    for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
        histo_s[bin] = 0u;
    }
    __syncthreads();

    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < length) {
        int alphabet_position = data[i] - 'a';
        if (alphabet_position >= 0 && alphabet_position < 26) {
            atomicAdd(&(histo_s[alphabet_position/4]), 1);
        }
    }

    // commit the values in private copy into the block
    __syncthreads();
    // responsible for creating one or more histogram bins
    for (unsigned int bin=threadIdx.x; bin<NUM_BINS; bin += blockDim.x) {
        // read private bin value
        unsigned int binValue = histo_s[bin];
        if (binValue > 0) {
            atomicAdd(&(histo[bin]), binValue);
        }
    }
}
```

### **Coarsening**
As discussed above shared memory reduces the latency of each atomic operation in privatized histogram. But their is an extra overhead while making the private copy to the public copy. This is done once per thread block. This overhead is worth paying when we execute threads blocks in parallel. But when the number of thread blocks that are launched exceeds the number of that can be executed by the hardware, there is unecessay privatization overhead. This overhead can be reduced via thread coarsening, i.e., by reducing the number of private copies made, by reducing the number of blocks such that each thread process multiple input elements. We'll read two ways to assign multiple input elements to a thread:
- Contiguous Partitioning: sequential access pattern by each thread makes good use of cache lines.
- Interleaved Partitioning

#### **Contiguous Partitioning**
```cuda
__global__  void histo_private_kernel(char* data, unsigned int length, unsigned int* histo) {
    // initialize private bins
    __shared__ unsigned int histo_s[NUM_BINS];
    for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
        histo_s[binIdx] = 0u;
    }
    __syncthreads();

    // histogram
    // Contiguous partitioning
    // CFACTOR is a coarsening factor
    unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
    for (unsigned int i = tid*CFACTOR; i < min((tid+1)*CFACTOR, length); i++) {
        int alphabet_position = data[i] - 'a';
        if (alphabet_position >= 0 && alphabet_position < 26) {
            atomicAdd(&(histo_s[alphabet_position/4]), 1);
        }
    }

    __syncthreads();
    // commit to global memory
    for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
        unsigned int binValue = histo_s[binIdx];
        if (binValue > 0) {
            atomicAdd(&(histo[binIdx]), binValue);
        }
    }
}
```

#### **Interleaved Partitioning**
```cuda
__global__  void histo_private_kernel(char* data, unsigned int length, unsigned int* histo) {
    // initialize private bins
    __shared__ unsigned int histo_s[NUM_BINS];
    for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
        histo_s[binIdx] = 0u;
    }
    __syncthreads();

    // histogram
    // Interleaved partitioning
    unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
    for (unsigned int i = tid; i < length; i += blockDim.x*gridDim.x) {
        int alphabet_position = data[i] - 'a';
        if (alphabet_position >= 0 && alphabet_position < 26) {
            atomicAdd(&(histo_s[alphabet_position/4]), 1);
        }
    }

    __syncthreads();
    // commit to global memory
    for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
        unsigned int binValue = histo_s[binIdx];
        if (binValue > 0) {
            atomicAdd(&(histo[binIdx]), binValue);
        }
    }
}
```

### **Aggregation**
