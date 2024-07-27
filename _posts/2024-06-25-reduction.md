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

### **Resources & References**
<a id="link1">1</a>. Wen-mei W. Hwu, David B. Kirk, Izzat El Hajj, [Programming Massively Parallel Processors: A Hands-on Approach](https://www.amazon.in/Programming-Massively-Parallel-Processors-Hands/dp/0323912311), 4th edition, United States: Katey Birtcher; 2022
