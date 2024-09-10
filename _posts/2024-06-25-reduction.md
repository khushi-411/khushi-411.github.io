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
We will implement parallel sum reduction tree such that reduction is performed within a single block. If the input block size if N, we will call a kernel and launch a grid with one block of $1/2N$ threads since each thread adds two elements. In the next subsequent step, half of the thread will drop off, now $1/4N$ threads will participate. This thread will go on, until only one thread is remaining to produce the total sum.

The figure below shows the assignment of the threads to the input array locations and progress of execution over time.

<img alt="A Simple Reduction Kernel" src="/assets/CUDA/simple_redution.png" class="center" >

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

<img alt="Minimizing Control Divergence" src="/assets/CUDA/redution_control_divergence.png" class="center" >

### **Minimizing Memory Divergence**

### **Minimizing Global Memory Accesses**

<img alt="Minimizing Memory Divergence" src="/assets/CUDA/redution_shared_mem.png" class="center" >

### **Hierarchical Reduction for Arbitrary Input Length**

<img alt="Segmented multiblock reduction" src="/assets/CUDA/redution_multiblock.png" class="center" >

### **Thread Coarsening for Reduced Overhead**

<img alt="Thread Coarsening in reduction" src="/assets/CUDA/redution_thread_coarsening" class="center" >

### **Resources & References**
<a id="link1">1</a>. Wen-mei W. Hwu, David B. Kirk, Izzat El Hajj, [Programming Massively Parallel Processors: A Hands-on Approach](https://www.amazon.in/Programming-Massively-Parallel-Processors-Hands/dp/0323912311), 4th edition, United States: Katey Birtcher; 2022
