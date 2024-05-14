---
layout: post
title: Multidimensional Grid and Data
date: 2024-05-13
category: CUDA
tags:
- cuda
redirect_from:
- /cuda/2024/05/13/multidim_grids_and_data/
- /multidim_grids_and_data.html
---

## Multidimensional Grid and Data

### **Introduction**
Hi there! The blog posts aim to share how the blocks and threads
are organized and how they are used to process multidimensional data.
We'll start with the basics of thread organization and then move our
focus to its applications. We'll demonstrate color-to-gray scale conversion,
blurring images and a
naive implementation of matrix multiplication in CUDA.

This blog post is written while reading the third chapter,
Multidimensional Grids and Data, of the incredible book
"Programming Massively Parallel Processors:
A Hands-on Approach<sup>[1](#link1)</sup>" by [Wen-mei W. Hwu](https://scholar.google.com/citations?user=ohjQPx8AAAAJ&hl=en), [David B. Kirk](https://scholar.google.com/citations?user=fMbArPwAAAAJ&hl=en), and [Izzat El Hajj](https://scholar.google.com/citations?user=_VVw504AAAAJ&hl=en).

### Multidimensional Grid Organization
In CUDA, computation is performed in a three-level hierarchy.
As soon as the user launches the kernel in device memory, a grid is
executed. A grid is a 3D array of blocks. A block is a 3D array of
threads. A thread is a simplified view of how a processor executes
a sequential program in a modern computer.

Some of the CUDA built-in variables are `blockIdx` (block index),
`threadIdx` (thread index), `gridDim` (dimensions of the grid, aka,
depicts the number of blocks in a grid), and `blockDim` (dimensions of
the block, aka, depicts the number of threads in a block).
To launch a kernel, we need to provide two parameters, as shown below:

```cuda
// dim3 (built-in variable) is a type
// dimGrid and dimBlock are host code variables
dim3 dimGrid(32, 1, 1);
dim3 dimBlock(128, 1, 1);

kernelLaunch<<<dimGrid, dimBlock>>>(...);
```

<img alt="Multidimensional Grids and Blocks" src="/assets/CUDA/kernel_launch.png" class="center" >

#### Notes:
- If we want to work on 1D grids and blocks, we can call kernels with
integer values instead of declaring them explicitly.
- Allowed values of `gridDim.x` range from $1$ to $2^{31} - 1$.
And those of `gridDim.y` and `gridDim.z` range from $1$ to $2^6 - 1$
- The total size of the block in the current CUDA system is limited to $1024$ threads.
- All threads in a block have the same `blockIdx` values,
and all blocks in a grid have the same `gridIdx` values.
- Range of `blockIdx.x`: $0$ to $gridDim.x - 1$

### Color to Gray Scale Conversion
Images are a 2-dimensional array of pixels. To find the co-ordinate
of a thread assigned to process the pixel is given by:
```cuda
Vertical (row co-ordinate) = blockIdx.y * blockDim.y + threadIdx.y;
Horizontal (column co-ordinate) = blockIdx.x * blockDim.x + threadIdx.x;
```
In C/CUDA, the information on the number of columns/rows in
dynamically allocated arrays is not known at compile time. As a result,
programmers need to explicitly linearize, aka flatten, a dynamically allocated
2D or a 3D array into a linear array in the current CUDA C.
This is due to the flat memory space in modern computers. There are two
ways we can linearize multidimensional arrays: row-major layout and
column-major layout. Below is the diagram showing how to linearize the
row-major 2-dimensional array:

<img alt="Linearize multi-dimensional arrays" src="/assets/CUDA/linearize.png" class="center" >

### **Resources & References**
<a id="link1">1</a>. Wen-mei W. Hwu, David B. Kirk, Izzat El Hajj, [Programming Massively Parallel Processors: A Hands-on Approach](https://www.amazon.in/Programming-Massively-Parallel-Processors-Hands/dp/0323912311), 4th edition, United States: Katey Birtcher; 2022
