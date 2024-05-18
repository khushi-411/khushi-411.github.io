---
layout: post
title: Memory Architecture and Data Locality
date: 2024-05-18
category: CUDA
tags:
- cuda
redirect_from:
- /cuda/2024/05/18/memory_architecture_and_data_locality/
- /memory_architecture_and_data_locality.html
---

### **Introduction**
Hi there! The blog posts aim to study the GPUs' on-chip
memory architecture and how to organize and position data
for efficient thread access. Until now, we executed our
programs all from global memory access (off-chip DRAM),
which leads to delays and traffic and negatively affects
performance. In this blog post, we will study ways to
tolerate these long-latency operations, i.e. it'll
introduce various techniques to reduce global memory access.

This blog post is written while reading the fifth chapter,
Memory Architecture and Data Locality of the amazing book
"Programming Massively Parallel Processors: A Hands-on
Approach<sup>[1](#link1)</sup>" by
[Wen-mei W. Hwu](https://scholar.google.com/citations?user=ohjQPx8AAAAJ&hl=en),
[David B. Kirk](https://scholar.google.com/citations?user=fMbArPwAAAAJ&hl=en),
and [Izzat El Hajj](https://scholar.google.com/citations?user=_VVw504AAAAJ&hl=en).

### **Importance of memory access Efficiency**
For matrix multiplication, in every iteration in a
for-loop, two global memory accesses are performed,
one for floating-point multiplication and the other
for its addition. This ratio of floating-point
operations (FLOPS) to bytes (B) accessed from the
global memory is called **arithmetic intensity**
or **computational intensity**. The execution of the
matrix multiplication kernel is limited by the
rate at which data can be delivered from the memory
to the GPU cores. Such executions are referred
to as **memory-bound** programs.

For example, the Ampere GPUs have a peek memory
bandwidth of 1555 GB/sec, and the number of
GFLOPS performed is 19,500. Therefore, the
computational intensity will be 19,500 / 1555 = 12.5
Operations per byte. That means that to
fully utilize every 4 bytes of memory,
50 operations must be performed.

Hence, the programmer must determine the
relative case of the program's compute-bound and
memory-bound scenarios to write a performant model.

### **CUDA Memory Types**
From the figure above, here is a short description
of the CUDA memory types:
- **Global Memory**: It is read and written by
the hosts and the device.
- **Constant Memory**: It is read and written by the
hosts but only read by the device (short latency and high bandwidth).
- **Local Memory**: It is placed in the global memory;
it can be read and written but not shared across the threads.
Each thread has its own section of private local memory
that cannot be shared or allocated to the registers.
- **Shared Memory**: This is on-chip memory with high-speed access.
It is allocated to thread blocks, so access to all the
threads is across the blocks; that's how threads interact with each other.
- **Registers**: These are on-chip memory with
high-speed access and are allocated to individual threads.
Kernel functions use these registers to hold private variables and data.

### **Resources & References**
<a id="link1">1</a>. Wen-mei W. Hwu, David B. Kirk, Izzat El Hajj, [Programming Massively Parallel Processors: A Hands-on Approach](https://www.amazon.in/Programming-Massively-Parallel-Processors-Hands/dp/0323912311), 4th edition, United States: Katey Birtcher; 2022
