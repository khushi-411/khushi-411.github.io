---
layout: post
title: Convolution
date: 2024-05-22
category: CUDA
tags:
- cuda
redirect_from:
- /cuda/2024/05/22/convolution/
- /convolution.html
---

### **Introduction**
Convolution is an array operation used in
various forms in signal processing, digital recording,
image/video processing, and computer vision.
Each output element is calculated independently
as a weighted sum of the corresponding input elements
and surrounding input elements. The weights used in
calculating the weighted sum are defined as a filter
array known as a convolution kernel. However,
processing multiple data inputs is challenging,
so it becomes a sophisticated use case for tiling
methods and input data staging methods.

This blog post is written while reading the
third chapter, Compute Architecture and Scheduling,
of the incredible book "Programming Massively Parallel
Processors: A Hands-on Approach<sup>[1](#link1)</sup>"
by [Wen-mei W. Hwu](https://scholar.google.com/citations?user=ohjQPx8AAAAJ&hl=en),
[David B. Kirk](https://scholar.google.com/citations?user=fMbArPwAAAAJ&hl=en),
and [Izzat El Hajj](https://scholar.google.com/citations?user=_VVw504AAAAJ&hl=en).

### **Background**
A convolution on 1D data is known as 1D convolution,
it is defined as an input data array of n
elements [x<sub>0</sub>, x<sub>1</sub>,...,x<sub>n-1</sub>]
and a filter array of 2r+1 elements
[f<sub>0</sub>, f<sub>1</sub>,...,f<sub>2r</sub>]
and returns an output data array y. The size of the
filter is an odd number, so it is symmetric
around the element to be calculated.

<p align="center">
  <img alt="1D Convolution" src="/assets/CUDA/conv1d_formula.png" class="center" >
</p>

A convolution on 2D arrays is known as 2D convolution.
Let N be the 2-dimensional array, and f be the
filter of dimension (2r<sub>x</sub>+1) in the
x-direction and (2r<sub>y</sub>+1) in the y-direction.
To output convoluted kernel is given by:

<p align="center">
  <img alt="2D Convolution" src="/assets/CUDA/conv2d_formula.png" class="center" >
</p>

For the elements in the input array that do
exist for the calculation with the filter arrays,
we create an imaginary element, either 0 or
any other values; these cells with missing
elements are known as **ghost cells**.
These boundary conditions affect the efficiency of the tiling.

### **Parallel Convolution: a basic algorithm**
```cuda
__global__ void convolution_2d_basic_kernel(float* N, float* F, float* P,
    int r, int width, int height) {
    
    // To calculate the output element indices
    int outCol = blockIdx.x*blockDim.x + threadIdx.x;
    int outRow = blockIdx.y*blockDim.y + threadIdx.y;

    // Register variable
    float Pvalue = 0.0f;
    for (int fRow = 0; fRow < 2*r+1; fRow++) {
        for (int fCol = 0; fCol < 2*r+1; fCol++) {
            inRow = outRow - r + fRow;
            inCol = outCol - r + fCol;

            // There will be control flow divergence
            // It'll depend on the width and height of the
            //     input array and radius of the filter
            // To handle ghost cells
            if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
                Pvalue += F[fRow][fCol]*N[inRow*width + inCol];
            }   
        }   
    }   
    // Release Pvalue to the output element
    P[outRow][outCol] = Pvalue;
}
```

### **Constant Memory and Caching**
Characteristics for an array to be implemented in the constant memory:
- The constant memory size is very small, ~ 64 KB.
In order for an array to reside here, its size
should be small. The filter array discussed
above holds up a very small size (7 or less).
- The values of the variables should not change
while executing in the kernel. For example, the
filter values in the convolution kernel above do not change during execution.
- All threads should execute the elements of the filter array.

To declare any variable or array in constant memory,
apply `__constant__` to tell the compiler.
The only difference is that while writing this kernel,
we'll not pass the filter array through a pointer
as passed above; instead, the filter array will
be accessed as a global variable.

Since constant memory variables are not modified
during kernel execution, we don't need to support
writes into threads when caching them to the SM.
Hence, this saves us from memory and power consumption.
The variables/arrays stored in a form of cache here are
highly effective and are known as *constant cache*.
The constant cache provides a huge bandwidth to satisfy
the data needs of these threads. This implies that no
DRAM bandwidth is spent on accesses to the filter
elements. It also implies that access to input
elements also benefits from caching.

### **Tiled Convolution with Halo cells**

### **Resources & References**
<a id="link1">1</a>. Wen-mei W. Hwu, David B. Kirk, Izzat El Hajj, [Programming Massively Parallel Processors: A Hands-on Approach](https://www.amazon.in/Programming-Massively-Parallel-Processors-Hands/dp/0323912311), 4th edition, United States: Katey Birtcher; 2022
