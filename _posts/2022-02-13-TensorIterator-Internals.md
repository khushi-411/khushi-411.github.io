---
layout: post
title: TensorIterator-Internals
date: 2022-02-13
category: pytorch
tags: 
- pytorch
redirect_from:
- /pytorch/2022/02/13/TensorIterator-Internals/
- /TensorIterator-Internals.html
---

Hi, Guys! Excited about the post? So, I am!
Before that, let's discuss the prerequisite knowledge you'll
need and what all things you are gonna learn. This post will dive
into details about the `TensorIterator Internals`. You will know how
the working of these C++ classes are helpful in 
[PyTorch](https://github.com/pytorch/pytorch).
You will learn how to get into the codebase.
You will get to know for what sake, TensorIterators are implemented,
the benefits of using it, and a lot more.
Note that the concept of `TensorIterator` is inspired by
[NumPy's](https://github.com/numpy/numpy) Array Iterator, `NpyIter`.

Ahead of that, I want to thank [Edward Z. Yang](https://github.com/ezyang)
for his podcast on [TensorIterator Internals](https://pytorch-dev-podcast.simplecast.com/episodes/tensoriterator),
[Sameer Deshmukh](https://github.com/v0dro) for writing a blog on [PyTorch TensorIterator Internals](https://labs.quansight.org/blog/2020/04/pytorch-tensoriterator-internals/index.html),
[Kurt Mohler](https://github.com/kurtamohler) for discussing the blog about the [PyTorch TensorIterator Internals - 2021 Update](https://labs.quansight.org/blog/2021/04/pytorch-tensoriterator-internals-update/index.html),
[Kshitij Kalambarkar](https://github.com/kshitij12345) for writing a blog about [JITerator Internals](https://kshitij12345.github.io/pytorch,/cuda/2022/01/17/Jiterator.html)
discussing the GPU support with TensorIterator. Thanks to you all!

The contents of this blog post are highly inspired by their work and PyTorch's
article on [How to use TensorIterator?](https://github.com/pytorch/pytorch/wiki/How-to-use-TensorIterator).
All the code snippets used in this blog post are available
in my GitHub repository named [TensorIterator-Internals](https://github.com/khushi-411/TensorIterator-Examples).

Let's start with an example.
We are given two vectors and the task is to add them.
The conventional way we will be following is: (This might be the coolest way to add :o)

```cpp
for (auto it1 = x.begin(), it2 = d.begin();
     it1 != x.end() && it2 != d.end(); 
     ++it1, ++it2)
{
    x1.push_back(*it1 + (*it2));
    x2.push_back(*it1 + (*it2));
}
```
<p align='right'>Credits: <a href="https://stackoverflow.com/questions/57048288">Stack Overflow</a></p>

But what if we want to enhance their speed or perform the same 
operations with a number of devices with different configs 
or if we're going to add two vectors of different sizes,
or if we want to achieve an output of a particular pre-defined type.
How to avoid memory overlaps? 
Ain't it fascinating to attain all the properties in one? 
Here comes the role of TensorIterator.

## **Basics of TensorIterator and its Config**
TensorIterator is the most fundamental class of PyTorch. In order to register
any operator in PyTorch, guys, you need to deal with it. The following snippet
demonstrates the basic construction of the TensorIterator. We first have to
build it with the required configuration. We only need to add an entry for the
input and the output entries. TensorIterator's infrastructure deals with the
rest of the story about the input/output shapes, data types, and memory overlaps.
Remember, once you enter a configuration, you can't change it because
TensorIterator is immutable.

The code snippet demonstrates the build of Binary Operations in PyTorch.

```cpp
#define BINARY_OP_CONFIG()
  TensorIteratorConfig()
    .set_check_mem_overlap(true)
    .allow_cpu_scalars(true)
    .promote_inputs_to_common_dtype(true)
    .cast_common_dtype_to_outputs(true)
    .enforce_safe_casting_to_output(true)

void TensorIteratorBase::build_binary_op(const TensorBase& out, const TensorBase& a, const TensorBase& b) {
  build(BINARY_OP_CONFIG()
      .add_owned_output(out)
      .add_owned_input(a)
      .add_owned_input(b));
}
```

If you want to learn more about building operations in TensorIterator checkout [here](https://github.com/pytorch/pytorch/blob/81a1330b913aeb5760e5e5fb15072748fdf85dc2/aten/src/ATen/TensorIterator.cpp#L1450).

## **Evolution of TensorIterator**
Before the introduction of TensorIterator, `TH` implementations were used.
`TH` uses preprocessor macros to write type independent loops over tensors.
That made things complicated and difficult to understand. Also, the speed of
the operations was affected. TensorIterator solves the problem.
It was introduced to simplify PyTorch operations. It uses C++ templates for
implementation. In this post, we will see the evolution of TensorIterator. Let's start...

### **Simple Implementation**
At the very beginning, the contributors of PyTorch introduced the naive
approach of writing an operator. This method is no longer in use, but I
think it is good to start. Here, we first define the iterator configuration
using `TensorIterator::Builder` that calculates the shapes and types of the variants,
then apply loop implementation that helps decide the operator's dispatch.

In the given example, we allocate `ret` in order to call `ret.dont_resize_outputs()`.
What if we have different input/output types? Call `ret.dont_compute_common_dtype()`, simple?

```cpp
void add_kernel(at::Tensor& ret, const at::Tensor& a, const at::Tensor& b, Scalar alpha_scalar) {
    auto builder = at::TensorIterator::Builder();
    builder.add_output(ret);
    builder.add_input(a);
    builder.add_input(b);
    auto iter = builder.build();
    auto alpha = alpha_scalar.to<scalar_t>();
    at::native::binary_kernel(*iter,
      [=](scalar_t a, scalar_t b) -> scalar_t { return a + alpha * b; },
     );
}
```

### **Using Native**
This section talks about the methods illustrated by the developers in the
`aten/src/ATen/native/cpu/` directory. The files in this directory are
compiled multiple times with different instruction sets. Please make sure
the kernels in this directory are under an anonymous namespace.
It helps the functions to have an internal linkage.

This method helped the developers to achieve compile-time optimizations.
Here, the kernels have the same dispatch patterns but everything is
should be mentioned inside the kernel.

```cpp
void add_kernel(at::Tensor& ret, const at::Tensor& a, const at::Tensor& b, Scalar alpha_scalar) {
  auto builder = at::TensorIterator::Builder();
    builder.add_output(ret);
    builder.add_input(a);
    builder.add_input(b);
    auto iter = builder.build();

  AT_DISPATCH_ALL_TYPES(iter.type(), "add", [&]() {
    auto alpha = alpha_scalar.to<scalar_t>();
    at::native::binary_kernel(iter,
      [=](scalar_t a, scalar_t b) -> scalar_t { return a + alpha * b; },
     );
  });
}
```

### **Vectorization**
The code demonstrates the explicit vectorization in order to achieve better performance optimization.

```cpp
void add_kernel(TensorIterator& iter, Scalar alpha_scalar) {
  AT_DISPATCH_ALL_TYPES(iter.type(), "add", [&]() {
    auto alpha = alpha_scalar.to<scalar_t>();
    auto alpha_vec = Vec256<scalar_t>(alpha);
    binary_kernel_vec(iter,
      [=](scalar_t a, scalar_t b) -> scalar_t { return a + alpha * b; },
      [=](Vec256<scalar_t> a, Vec256<scalar_t> b) {
        return vec256::fmadd(b, alpha_vec, a);
      });
  });
}
```

## **Properties of TensorIterator**

```cpp
#define UNARY_OP_CONFIG()
  TensorIteratorConfig()
    .set_check_mem_overlap(true)
    .cast_common_dtype_to_outputs(false)
    .enforce_safe_casting_to_output(false)
    .check_all_same_dtype(true)
```

The example above displays the TensorIterator Config for Unary Operations.
The TensorIterator checks the memory overlaps of the operands, validates
the output data types, and checks if all the input operands are of the same data type.
To build the `unary_op` we use:

```cpp
void TensorIteratorBase::build_unary_op(const TensorBase& out, const TensorBase& a) {
  build(UNARY_OP_CONFIG()
      .add_owned_output(out)
      .add_owned_input(a));
}
```

The properties of TensorIterator are as follows:
1. **` Broadcasting`**: What if the size of input and output are different
and we want to perform arithmetic operations? Here comes the role of
TensorIterator's property of Broadcasting. We only need dimensions to be aligned,
and one of the aligned dimensions should be one. If you want to know more about it,
check out the PyTorch documentation on [Broadcasting semantics](https://pytorch.org/docs/stable/notes/broadcasting.html).
2. **` Type Promotion`**: What if we want to add two vectors of different
data types? What if we want to get an output of some desired type?
TensorIterator supports Type Promotion. Check out more about
it in the [documentation](https://pytorch.org/docs/stable/tensor_attributes.html#type-promotion-doc).
3. **` Memory Overlaps`**: What if the output buffer alias the input?
What if the buffer is stridden? What if multiple things point to the same
memory location? Isn't it interesting to avoid these illegal memory overlaps?
TensorIterator prevents it!
4. **` Dimension Coalescing`**
5. **` Parallelization`**: TensorIterator supports running multiple iterations in
parallel. Iterations are classified into two categories pointwise iterations and
reduction iterations. Pointwise iterations can parallelize data along any dimension,
while reduction operations can parallelize only on specified dimensions.
We will learn more about it in the next section.
6. **` Vectorization`**: Don't you think with so many properties in one TensorItearator's
performance may affect? How to diminish it? TensorIterator supports vectorization,
as illustrated in the above section. It helps in the speedy manipulation of data.
It avoids explicit looping, indexing, etc. More on vectorization
can be found at [Wikipedia](https://en.m.wikipedia.org/wiki/Vectorization).

## **Iterations Using TensorIterator**

### **Iteration Details**
The simplest iteration operation is performed using the `for_each` function.
It has two overloads (`loop1d_t` and `loop2d_t`) as mentioned below. 

#### **Source Code [ `for_each` ]**
`for_each` implicitly parallelizes the input tensors if the input size
is greater than the number of elements. It implements `serial_for_each`
if the input size is less than or equal to the `GRAIN_SIZE`.
If you want to explicitly run your program in serial you can use `serial_for_each` differently.

```cpp
void for_each(loop1d_t loop, int64_t grain_size = at::internal::GRAIN_SIZE) {
  for_each(loop_2d_from_1d(loop), grain_size);
  }

void for_each(loop2d_t loop, int64_t grain_size = at::internal::GRAIN_SIZE);

/// Parallel algorithm to check that input should not
/// split smaller than GRAIN_SIZE
void parallel_reduce(loop2d_t loop);

void serial_for_each(loop1d_t loop, Range range) {
  serial_for_each(loop_2d_from_1d(loop), range);
  }

void serial_for_each(loop2d_t loop, Range range) const;
```
<p align='right'><a href="https://github.com/pytorch/pytorch/blob/33b7e6ff239ef674ff3cf012b3a280405fae07b9/aten/src/ATen/TensorIterator.h#L346">xref</a></p>

```cpp
void TensorIteratorBase::for_each(loop2d_t loop, int64_t grain_size) {
  /// pointer to the reference for total number of elements
  /// in the input tensor
  int64_t numel = this->numel();
  if (numel == 0) {
    /// if tensor size = 0; return
    return;
  } else if (numel < grain_size || at::get_num_threads() == 1) {
    /// If number of elements < `grain_size` or number of threads is 1,
    ///  run operations serially
    return serial_for_each(loop, {0, numel});
  } else {
    /// Run operations parallely
    at::parallel_for(0, numel, grain_size, [&](int64_t begin, int64_t end) {
      serial_for_each(loop, {begin, end});
    });
  }
}
```

This snippet is a simple demonstration of the application of `for_each`.
We first build the TensorIterator (the configuration mentioned below).
Then we create a `copy_loop` function that copies input data to the output.
In the examples below, the arguments signify the following:
- **` char** data`**: This argument denotes the `char*` pointer to the input tensors.
- **` const int64_t* strides`**: It is an array containing strides of the tensor. We add this stride to the pointer to move to the next element in the tensor.
- **` int64_t n`**: Size of the dimension we are iterating.

```cpp
at::TensorIteratorConfig iter_config;
  iter_config.add_output(out)
      .add_input(a)

      /// call if output was already allocated
      .resize_outputs(false)

      /// call if inputs/outputs have different types
      .check_all_same_dtype(false);

  auto iter = iter_config.build();

  /// Copies data from input into output.
  auto copy_loop = [](char **data, const int64_t *strides, int64_t n) {
    auto *out_data = data[0];
    auto *in_data = data[1];

    /// adding strides to reach the next element
    for (int64_t i = 0; i < n; i++) {
      /// casting to floating dtype
      *reinterpret_cast<float *>(out_data) =
          *reinterpret_cast<float *>(in_data);
      out_data += strides[0];
      in_data += strides[1];
    }
};

iter.for_each(copy_loop);
```

### **Using Kernels for Iterations**
Kernel methods are used in TensorIterator to perform simple pointwise operations
onto the input tensors. Two types of kernels are implemented:
`cpu_kernel()`, to explicitly perform operations on `cpu`, and the `gpu_kernel()`,
to explicitly perform operations on `gpu`.

The below-mentioned example demonstrates writing a CPU and a GPU kernel
to add two operations element-wise. While writing kernels we don't need
to worry about data types, memory overlaps, etc.

```cpp
auto iter = TensorIteratorConfig()
   .add_output(output)
   .add_input(input)
   .build()

/// CPU kernel
cpu_kernel(iter, [](float a, float b) {
   return a + b;
 });

/// GPU kernel 
gpu_kernel(iter, []GPU_LAMBDA(float a, float b) -> float {
   return a + b;
});
```

## **Conclusion**
Enjoyed? I did a lot! This post was just a gentle introduction to the `TensorIterator`.
If you want to learn more about it, you can dive deeper into the PyTorch codebase.
Apart from this, TensorIterator might be slow at times. If you want to know more about it,
you can visit [here](https://dev-discuss.pytorch.org/t/comparing-the-performance-of-0-4-1-and-master/136).

Thank you so much for being with me throughout this post. I love feedback.
Let me know your opinions and questions in the [Issue Tracker](https://github.com/khushi-411/TensorIterator-Internals/issues) of my
GitHub Repository: [TensorIterator-Internals](https://github.com/khushi-411/TensorIterator-Internals).Thanks!
