---
layout: post
title: Performance Considerations
date: 2024-05-20
category: CUDA
tags:
- cuda
redirect_from:
- /cuda/2024/05/20/performance_considerations/
- /performance_considerations.html
---

### **Introduction**
To achieve high-performance computing, we need to
manage parallel code alongside the given hardware
resources. In this part of the blog post, we'll
read about the off-chip memory architecture and
discuss memory coalescing, memory latency hiding,
and thread coalescing (this depends on the different
aspects of the architecture). Lastly, we'll study
the common checklist of optimization techniques
for different types of parallel patterns.

This blog post is written while reading the
sixth chapter, Performance Considerations of
the fantastic book "Programming Massively
Parallel Processors: A Hands-on Approach<sup>[1](#link1)</sup>"
by [Wen-mei W. Hwu](https://scholar.google.com/citations?user=ohjQPx8AAAAJ&hl=en),
[David B. Kirk](https://scholar.google.com/citations?user=fMbArPwAAAAJ&hl=en),
and [Izzat El Hajj](https://scholar.google.com/citations?user=_VVw504AAAAJ&hl=en).

### **Memory Coalescing**
Memory coalescing is a technique used to
move data efficiently from global memory
to shared memory. It is used alongside the
tiling techniques. When accessing the DRAM
location, the range of consecutive locations
that are along with the requested locations
is accessed. These are known *DRAM bursts*.
If any applications need any focused use of
data, DRAM accesses them directly and transfers
them at high speed, compared to some random
access from the sequence.

In modern DRAMs, we know that the matrices
are linearized while accessing their elements.
When all threads in a wrap access consecutive
memory locations, the hardware coalesces, i.e.,
the hardware combines all these accesses into a
combined access. Such access allows DRAM to deliver
data as a burst.

For instance, in a row-major layout matrix,
access to the input elements is already
coalesced because consecutive threads will
have access to consecutive elements of the column,
as shown below:

<img alt="Coalesced Access" src="/assets/CUDA/coalesced_mem.png" class="center" >

The next image, in a column-major layout matrix below,
shows when the consecutive threads access the consecutive
columns. The logical view of the matrix shows that
it's not favourable for coalescing. In the physical view,
we are accessing consecutive elements, but they are
not consecutive in memory because of the column-major layout.

<img alt="Uncoalesced Access" src="/assets/CUDA/uncoalesced_mem.png" class="center" >

To optimize performance in cases where we cannot
naturally achieve memory coalescing, we can
rearrange how threads can be mapped to the data or
rearrange the data layout itself. Another way is
to transfer the data between the global memory and
shared memory in a coalesced manner and carry an
unfavourable access pattern in the shared memory for faster access latency.

An optimization technique for a matrix-matrix
multiplication when the second input matrix
is in the column-major layout is known as **corner turning**.
To solve this problem, where consecutive threads
load nonconsecutive locations in the memory, resulting
in uncoalesced memory accesses, we assign consecutive threads
to load consecutive elements of the matrix (figure shown below).
This ensures memory accesses are coalesced.

Matrix multiplication without corner turning.
<img alt="Uncoalesced" src="/assets/CUDA/uncoalesced.png" class="center" >

Applying corner turning to coalesce accesses to matrix B (column-major layout).
<img alt="Coalesced" src="/assets/CUDA/coalesced.png" class="center" >

### **Hiding Memory Latency**
DRAM systems have two levels of parallel organizations:
banks and channels. Each channel is a memory controller
that connects DRAM banks to the processor. A bus connects
the banks to the channels. A bus's data transfer bandwidth
is determined by its width and clock frequency.
The image below shows the data transfer timing
when a single bank and when two (or multiple) banks are connected
to a channel and how to hide its latency.

<img alt="Banking improves data transfers." src="/assets/CUDA/banking.png" class="center" >

### **Resources & References**
<a id="link1">1</a>. Wen-mei W. Hwu, David B. Kirk, Izzat El Hajj, [Programming Massively Parallel Processors: A Hands-on Approach](https://www.amazon.in/Programming-Massively-Parallel-Processors-Hands/dp/0323912311), 4th edition, United States: Katey Birtcher; 2022
