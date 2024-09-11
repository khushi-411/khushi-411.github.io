---
layout: post
title: Reduction
date: 2024-06-25
category: CUDA
tags:
- cuda
redirect_from:
- /cuda/2024/06/25/reduction/
- /reduction.html
---

### **Introduction**
A reduction is a method of deriving a single value from the array of value. Example, sumation of an array. A parallel reduction is an technique of co-ordinating parallel threads to produce right results. Reduction can be defined for mathematical operations like addition, subtraction, min, max, multiplication etc. This blog posts will start with reduction trees, a simple reduction kernel, minimizing control divergence, minimizing memory divergence, minimizing global memory accesses, hierarchical reduction, and thread coarsening for reduced overhead. Let's dig in!

This blog post is written while reading the
tenth chapter, Reduction,
of the incredible book "Programming Massively Parallel
Processors: A Hands-on Approach<sup>[1](#link1)</sup>"
by [Wen-mei W. Hwu](https://scholar.google.com/citations?user=ohjQPx8AAAAJ&hl=en),
[David B. Kirk](https://scholar.google.com/citations?user=fMbArPwAAAAJ&hl=en),
and [Izzat El Hajj](https://scholar.google.com/citations?user=_VVw504AAAAJ&hl=en).

### **Reduction Trees**
A reduction tree is actually a parallel reduction pattern whose leaves are original input elements and whose root is the final result. A reduction tree is not a tree data structure. Here, the edges shares the information between the operations performed. For N input values, tt takes log<sub>2</sub>N steps to complete the reduction process. The operator should be associate for the conversion from the sequential reduction to the reduction tree. We will also need to rearrange the operation while writing code hence the operator should hold the commutative property. Here's the example of a typical parallel sum reduction tree:

<img alt="Reduction Trees" src="/assets/CUDA/redution_max.png" class="center" >

### **A Simple Reduction Kernel**
We will implement parallel sum reduction tree such that reduction is performed within a single block. If the input block size if $N$, we will call a kernel and launch a grid with one block of $1/2N$ threads since each thread adds two elements. In the next subsequent step, half of the thread will drop off, now $1/4N$ threads will participate. This thread will go on, until only one thread is remaining to produce the total sum.

The figure below shows the assignment of the threads to the input array locations and progress of execution over time.

<img alt="A Simple Reduction Kernel" src="/assets/CUDA/simple_reduction.png" class="center" >

```cuda
__global__ void simpleSumReductionKernel(float* input, float* output) {
    // note: it jumps 2
    unsigned int i = 2*threadIdx.x;
    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
        if (threadIdx.x % stride == 2) {
            input[i] += input[i + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        *output = input[0];
    }
}
```

### **Minimizing Control Divergence**
In the above method, the management of active and inactive threads in each iteration results in higher control divergence because only half of the threads is used in the process and half of them is wasted in the subsequent steps. This leads to reduce in execution resource utilization efficiency. Hence, we need a better way to assign threads to the input array locations. In the above method, the distance between active threads increase over time, hence increases the level of control divergence. There is a better way to do this. As the time progresses, instead of increasing the distance between threads (strides), we should decrease them (as shown in the figure below). This will increase the efficiency and will improve the reduced resource consumption.

<img alt="Minimizing Control Divergence" src="/assets/CUDA/reduction_control_divergence.png" class="center" >

```cuda
__global__ void ConvergentSumReductionKernel(float* input, float* output) {
    unsigned int i = threadIdx.x;
    for (unsigned int stride = blockDim.x; stride >= 1; stride /= 2) {
        if (threadIdx.x < stride) {
            input[i] += input[i + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        *output = input[0];
    }
}
```

### **Minimizing Global Memory Accesses**
Following the above discussions, we can further improve the performance by using shared memory accesses instead of global memory. We'll keep the partial sum results in the shared memory as shown in the figure and code below.

<img alt="Minimizing Memory Divergence" src="/assets/CUDA/reduction_shared_mem.png" class="center" >

Note that for an $N$ element input array, there will be just $N + 1$ global memory accesses. With coalescing, there will be $N/32$ global memory accesses.

```cuda
__global__ void SharedMemorySumReductionKernel(float* input) {
    __shared__ float input_s[BLOCK_DIM];
    unsigned int t = threadIdx.x;
    input_s[t] = input[t] + input[t + BLOCK_DIM];
    for (unsigned int stride = blockDim.x/2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (threadIdx.x < stride) {
            input_s[t] += input_s[t + stride];
        }
    }
    if (threadIdx.x == 0) {
        *output = input_s[0];
    }
}
```

### **Hierarchical Reduction for Arbitrary Input Length**
All the kernels above perform reduction in a single block because we perform `__syncthreads()` operations which is limited within the block scope. This limits the level of parallelism to $1024$ threads on current hardware. Things get slower when the input size increases. To resolve this we partition the input elements into different segements such that each segment is the size of the block. Then all blocks independently execute the reduction tree and accumulate their results to the final output using an atomic add operation.

<img alt="Segmented multiblock reduction" src="/assets/CUDA/reduction_multiblock.png" class="center" >

```cuda
__global__ SegmentedSumReductionKernel(float* input, float* output) {
    __shared__ float input_s[BLOCK_DIM];
    // each block process 2*blockDim.x elements
    // starting location of the segment to be processed by the block
    unsigned int segment = 2*blockDim.x*blockIdx.x;
    unsigned int i = segment + threadIdx.x;
    unsigned int t = threadIdx.x;
    input_s[t] = input[i] + input[i + BLOCK_DIM];
    for (unsigned int stride = blockDim.x/2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (t < stride) {
            input_s[t] += input_s[t + stride];
        }
    }
    if (t == 0) {
        atomicAdd(output, input_s[0]);
    }
}
```

### **Thread Coarsening for Reduced Overhead**
Until now to parallelize reduction, we have actually paid heacy price to distribute the work accross multiple thread blocks. This process increases the hardware under-utilization as more wraps starts becoming idle and final wrap experience more control divergence. Hence, we'll serialize the threads blocks manually so that the hardware resources are not spend here. We'll use thread coarsening technique to do that and start by assigning more elements to each thread block.

<img alt="Thread Coarsening in reduction" src="/assets/CUDA/reduction_thread_coarsening" class="center" >

The kernel code for implementing reduction with thread coarsening for the multiblock segmented kernel is given below.

```cuda
__global__ CoarsenedSumReductionKernel(float* input, float* output) {
    __shared__ float input_s[BLOCK_DIM];
    unsigned int segment = COARSE_FACTOR*2*blockDim.x*blockIdx.x;
    unsigned int i = segment + threadIdx.x;
    unsigned int t = threadIdx.x;
    float sum = input[i];
    for (unsigned int tile = 1; tile < COARSE_FACTOR*2; ++tile) {
        sum += input[i + tile*BLOCK_DIM];
    }
    input_s[t] = sum;
    for (unsigned int stride = blockDim.x/2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (t < stride) {
            input_s[t] += input_s[t + stride];
        }
    }
    if (t == 0) {
        atomicAdd(output, input_s[0]);
    }
}
```

### **Where to Go Next?**
I hope you enjoyed reading this blog post!
If you have any questions or suggestions, please feel
free to drop a comment or reach out to me. I'd love to hear from you!

This post is part of an ongoing series on CUDA programming.
I plan to continue this series and keep things exciting.
Check out the rest of my CUDA blog series:
1. [Introduction to Parallel Programming and CPU-GPU Architectures](https://khushi-411.github.io/gpu_intro/)
2. [Multidimensional Grids and Data](https://khushi-411.github.io/multidim_grids_and_data/)
3. [Compute Architecture and Scheduling](https://khushi-411.github.io/compute_architecture_and_scheduling/)
4. [Memory Architecture and Data Locality](https://khushi-411.github.io/memory_architecture_and_data_locality/)
5. [Performance Considerations](https://khushi-411.github.io/performance_considerations/)
6. [Convolution](https://khushi-411.github.io/convolution/)
7. [Stencil](https://khushi-411.github.io/stencil/)
8. [Parallel Histogram](https://khushi-411.github.io/parallel_histogram/)
9. [Reduction](https://khushi-411.github.io/reduction/)

Stay tuned for more!

### **Resources & References**
<a id="link1">1</a>. Wen-mei W. Hwu, David B. Kirk, Izzat El Hajj, [Programming Massively Parallel Processors: A Hands-on Approach](https://www.amazon.in/Programming-Massively-Parallel-Processors-Hands/dp/0323912311), 4th edition, United States: Katey Birtcher; 2022 \
<a id="link2">2</a>. [Lecture 9 Reductions](https://www.youtube.com/watch?v=09wntC6BT5o) by [Mark Saroufim](https://x.com/marksaroufim); Mar 2024

