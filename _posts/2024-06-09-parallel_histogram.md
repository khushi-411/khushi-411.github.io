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
