---
layout: post
title: C++ Sanitizers
date: 2022-12-26
category: cpp
tags:
- cpp
redirect_from:
- /cpp/2022/12/26/cpp-sanitizers/
- /cpp-sanitizers.html
---

Hi, there! I recently learned about C++ Sanitizers. That's an amazing concept.
C++ Sanitizers are recently added in Clang and GCC compilers for runtime code analysis.
Sanitizers detect errors that are usually not detected during compile time.
In this blog, I'll share some basic details about Sanitizers in C++.
I'll cover AddressSanitizer, MemorySanitizer, and UndefinedBehaviorSanitizer.

### AddressSanitzer
Also called as ASan. It checks for memory addresses that are out of bounds in C++.
It finds heap-, stack-, global- buffer overflow, initialization order bugs, memory leaks,
uses after return, scope, etc. Let's take an example:
```cpp
#include <iostream>
#include <string>

int main() {
    const char *names[] = {"khushi", "simmi"};
    std::string last_arg = names[3];
    return last_arg.size();
}
```
Compile the code using the following GCC command:
```cpp
g++ address.cpp -Wall -Werror
```
And then execute using `./a.out`. Or try running the code using `clang`. Use the following command:
```cpp
clang address.cpp -Wall -Werror
```
The code runs fine for both compilers even if we turn on the warnings argument (talking about GCC). Let's add another check in GCC, compile using:
```cpp
g++ address.cpp -Wall -Werror -fsanitize=address
```
Or using `clang`:
```cpp
clang address.cpp -Wall -Werror -fsanitize=address
```
The program is aborted on running showing the following error. This behavior is interesting! It shows that there are some stack-buffer-overflow errors in our code.
```cpp
=================================================================
==12011==ERROR: AddressSanitizer: stack-buffer-overflow on address 0x7fffcde3fc58 at pc 0x556fd8e6c4c3 bp 0x7fffcde3fbf0 sp 0x7fffcde3fbe0
READ of size 8 at 0x7fffcde3fc58 thread T0
    #0 0x556fd8e6c4c2 in main (/home/khushi/Documents/cpp_practice/a.out+0x14c2)
    #1 0x7f6cf87cb082 in __libc_start_main ../csu/libc-start.c:308
    #2 0x556fd8e6c2cd in _start (/home/khushi/Documents/cpp_practice/a.out+0x12cd)

Address 0x7fffcde3fc58 is located in stack of thread T0 at offset 88 in frame
    #0 0x556fd8e6c398 in main (/home/khushi/Documents/cpp_practice/a.out+0x1398)

  This frame has 3 object(s):
    [48, 49) '<unknown>'
    [64, 80) 'names' (line 5)
    [96, 128) 'last_arg' (line 6) <== Memory access at offset 88 underflows this variable
HINT: this may be a false positive if your program uses some custom stack unwind mechanism, swapcontext or vfork
      (longjmp and C++ exceptions *are* supported)
SUMMARY: AddressSanitizer: stack-buffer-overflow (/home/khushi/Documents/cpp_practice/a.out+0x14c2) in main
Shadow bytes around the buggy address:
  0x100079bbff30: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
  0x100079bbff40: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
  0x100079bbff50: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
  0x100079bbff60: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
  0x100079bbff70: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
=>0x100079bbff80: f1 f1 f1 f1 f1 f1 01 f2 00 00 f2[f2]00 00 00 00
  0x100079bbff90: f3 f3 f3 f3 00 00 00 00 00 00 00 00 00 00 00 00
  0x100079bbffa0: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
  0x100079bbffb0: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
  0x100079bbffc0: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
  0x100079bbffd0: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
Shadow byte legend (one shadow byte represents 8 application bytes):
  Addressable:           00
  Partially addressable: 01 02 03 04 05 06 07 
  Heap left redzone:       fa
  Freed heap region:       fd
  Stack left redzone:      f1
  Stack mid redzone:       f2
  Stack right redzone:     f3
  Stack after return:      f5
  Stack use after scope:   f8
  Global redzone:          f9
  Global init order:       f6
  Poisoned by user:        f7
  Container overflow:      fc
  Array cookie:            ac
  Intra object redzone:    bb
  ASan internal:           fe
  Left alloca redzone:     ca
  Right alloca redzone:    cb
  Shadow gap:              cc
==12011==ABORTING
```
***Note 1***: You can get another check by adding `-g` in the command line for more description.
***Note 2***: Clang will throw an error even if you don't turn on the warnings check-in command.

I recommend the viewers change the check-in commands in GCC and Clang to see the
difference between the outputs of both compilers.

### MemorySanitizer
Also known as MSan. It is used to detect uninitialized memory in C/C++.
It usually occurs when stack- or heap-allocated memory is read before it is written.
Let's take an example:
```cpp
#include <iostream>

void set_val(bool &b, const int val) {
    if (val > 1) {
        b = false;
    }
}

int main(const int argc, const char *[]) {
    bool b;
    set_val(b, argc);
    if (b) {
        std::cout << "value set\n";
    }
}
```
Compile the code using the following GCC command:
```cpp
g++ memory.cpp -Wall -Wextra
```
Then execute the code using `./a.out`. Or try compiling using clang:
```cpp
clang memory.cpp -Wall -Wextra
```
The code again runs completely fine. Let's try adding another check using the clang compiler. Use:
```
clang memory.cpp -Wall -Wextra -fsanitize=memory
```
***Note 3***: Currently, gcc does not support the MemorySanitizer.
Coming back to the result part, we caught a similar issue as in AddressSanitizer.
As shown below, clang throws a linker error, and the program gets terminated.
```cpp
/usr/bin/ld: /tmp/2-a58743.o: in function `main':
2.cpp:(.text+0x1c0): undefined reference to `std::cout'
/usr/bin/ld: 2.cpp:(.text+0x1d3): undefined reference to `std::basic_ostream<char, std::char_traits<char> >& std::operator<< <std::char_traits<char> >(std::basic_ostream<char, std::char_traits<char> >&, char const*)'
/usr/bin/ld: /tmp/2-a58743.o: in function `__cxx_global_var_init':
2.cpp:(.text.startup+0xa): undefined reference to `std::ios_base::Init::~Init()'
/usr/bin/ld: 2.cpp:(.text.startup+0x2f): undefined reference to `std::ios_base::Init::Init()'
clang: error: linker command failed with exit code 1 (use -v to see invocation)
```
***Note 4***: To track the location of initialized values we used in code, use:
```cpp
clang memory.cpp -Wall -Wextra -fsanitize-memory-track-origins
```
***Note 5***: To get any stack traces, add `-fno-omit-frame-pointer`.
TODO: read about Valgrind and benchmark it.

### UndefinedBehaviourSanitizer
Also known as UBSan. This sanitizer is used to detect undefined behavior.
It modifies the program at compile time to catch the errors.
Examples of UndefinedBehaviourSanilizers are signed integer overflow, conversion
to or from floating-point types, which usually cause overflow, etc. Let's take an example:
```cpp
#include <iostream>
#include <cmath>

int main() {
    float x{-NAN};
    std::cout << x << "\n";
    std::cout << "abs:" << ::abs(x) << "\n";
    return 0;
}
```
Compile the code using:
```cpp
g++ ubcheck.cpp -Wall -Werror
```
And run using `./a.out`. It runs fine and produces the following output:
```cpp
-nan
abs:-2147483648
```
Let's try something exciting! Compile using the following command:
```cpp
g++ ubcheck.cpp -Wconversion -lubsan -fsanitize=undefined -g
```
Oops, it raises the following warning. The warning says the value is changed unknowingly
while converting floating point numbers to int. Interesting!
```cpp
ubcheck.cpp: In function ‘int main()’:
ubcheck.cpp:7:34: warning: conversion from ‘float’ to ‘int’ may change value [-Wfloat-conversion]
    7 |     std::cout << "abs:" << ::abs(x) << "\n";
      |                                  ^
```

### Closing Remarks & Acknowledgement
This post taught us the essential details about Address, Memory, and UndefinedBehaviour Sanitizers in C++.
Hope you enjoyed reading the blog post.

Before ending this post, I want to devote my vote of appreciation to
Janson Turner for his informative [video on C++ Sanitizers](https://www.youtube.com/watch?v=MB6NPkB4YVs),
Google team for descriptive [repository on GitHub](https://github.com/google/sanitizers) about sanitizers,
and Kshitij Kalambarkar & Sanchit Jain for explanatory comments on [Github PR](https://github.com/pytorch/pytorch/commit/dfd2edc025b284abc6972bdcfaa9f4f7b8808036).
Thanks to you all! It was fun learning!
