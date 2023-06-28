---
layout: post
title: Lambdas from Scratch
date: 2023-06-27
category: C++
tags:
- cpp
redirect_from:
- /cpp/2023/06/27/cpp_lambdas/
- /cpp_lambdas.html
---

Hi, Guys! In this post, we will go through the concepts of
Lambda expression in C++. That's a pretty interesting thing.

Before starting my blog, I want to present my gratitude
to Arthur O'Dwyer for his fantastic talk on
[Back to Basics: Lambdas from Scratch - CppCon 2019](https://www.youtube.com/watch?v=3jCOwajNch0&t=1s).
Thanks so much! The content of the blog posts is entirely inspired by his talk.

Let's start with the fundamental implementation of the
C++ program to add 1 to the number. We will write the code something like this:
```cpp
int plus1(int x) {
    return x+1;
}
```
The next thing C++ added to the library is `function overloading`.
For example, we can see the same function name is passed,
but both functions have different data types.
```cpp
int plus1(int x) {
    return x+1;
}
double plus1(double x) {
    return x+1;
}
```
The drawback is we have to write the same body twice. So, as the
solution to this problem, the cpp developers introduced the concept of `templates`.
The above code can be written as:
```cpp
template <typename T>
T plus1(T x) {
    return x+1;
}
auto y = plus1(42);
auto z = plus1(3.14);
```
These templates make the type deduction by themselves.
If we call `plus1(42)`, it instantiates an integer type; else,
if we call `plus1(3.14)`, it instantiates a floating type by itself.
The code generated from both methods (templated and non-templated)
will be the same. [Click to see](https://godbolt.org/z/E9W3ao4re).


C++ can also give the ability to call the function;
We can also access class members using/creating methods.
```cpp
class Plus {
    int value;
public:
    // constructor
    Plus(int v);
    // member function, with const object
    int plusme (int x) const {
        return x + value;
    }
};
```
How does the computer know that we have to call the **`plusme`**
method if we call the `Plus` constructor? This is known as `static typing`.
To get rid of pointers, we use this.

***Note:*** There's no difference between `template <typename T>` and
`template <class T>`! See [ref](https://stackoverflow.com/questions/5307036).

**TODO**: template template parameter. [Stack](https://stackoverflow.com/questions/213761)

C++ offers the concept of Operator Overloading to us.
```cpp
class Plus {
    int value;
  public:
  Plus(int v);
  
  int operator() (int x) const {
    return x + value;
  }
};
auto plus = Plus(1);
assert(plus(42) == 43);
```
**Lambdas reduce the boilerplate!**
```cpp
auto plus = [value=1](int x) { return x + value; };
```
Hence, this closure is without garbage collection, heap allocation,
runtime polymorphism, etc.
```cpp
bool contains_title(const std::vector<Book>& shelf, std::string title) {
    auto has_title_t = [t=title](const Book& b) {
        return b.title() == t;
    };
    return v.end() != 
        std::find_if(v.begin(), v.end(), has_title_t);
}
```
~**TODO**: What is passed in the parameters block? here: `()`~:
parameters that the lambda function will accept.

### Capturing by Reference
In the above example, [t=title], will make a copy of the string.
It's called a `copy constructor`.

What if we don't want to make a copy? We pass the reference value.
That is, we can capture a pointer. Code:
```cpp
auto has_title_t = [pt=&title](const Book& b) {
    return b.title() == *pt;
};
```
Here, we capture the pointer to the string. Therefore, we have to dereference it (`*pt`).

Adding syntactic sugar to lambda:
```cpp
auto has_title_t = [&t=title](const Book& b) {
    return b.title() == t;
};
```
Here, `t` is a reference to the title. We didn't make the copy.

**Dangling Pointer**: When a pointer is pointing at the memory
address of a variable, but after some time, that variable is
deleted from that memory location while the pointer is still pointing to it,
then such a pointer is known as a dangling pointer.

Capturing `by move`
```cpp
auto has_title_t = [t=std::move(title)](const Book& b) {
    return b.title() == t;
};
```
`[t=std::move(title)]` creates a `std::string t` and calls the move constructor to initialize it.

### Shorthands of lambda
- [t=title] () { decltype(title) ... use(t); }
- [title] () { decltype(title) ... use(title); }
- [&t=title] () { use(t); }
- [&t=title] () { use(title); }
- To capture only what is needed-
    - [=] () { use(title); } : Capture only what is needed
    - [&] () { use(title); } : Capture only what is needed, by reference  // Most useful
    - Globals and statics are not captured
 
### More features on lambda
1. lambda, which doesn't capture anything. It does not capture
any variable inside it; it operates solely on parameters and does not rely on external states.
```cpp
template<class T> void fn(T t);
fn(+[] (int x) { return x+1; });
```
2. Default-constructible (if capture less) (introduced in C++ 20).
It can be created without any capture or parameter list.
Allows to create an instance of lambda without providing any arguments.
It is also called **polymorphic lambdas** or **stateless lambdas**.
```cpp
auto lam = []() { return 1; };
decltype(lam) copy;
```
3. Lambdas are constexpr by default, but not noexcept by default.
In C++20, stateless lambdas without captures are implicitly constexpr.
Note that constexpr lambdas must have a non-empty function body, and
return type deduction is not allowed.
```cpp
static_assert(lam() == 1);
static_assert(not noexcept(lam()));
```
4. Lambdas may have local state
```cpp
int count = 0;
auto increment = [&count]() { count++; };
// increase count by 1
increment();
// increase count by 1
increment();
// output: count: 2
std::cout << "count: " << count << std::endl;
```

### How to mutate Lambdas?

Pre-lambda mutable state (wrong!)
```cpp
auto counter = []() { static int i; return ++i; };  
```
This behaves the same as the following class type:
```cpp
class Counter {
    // no captured data members
public:
    int operator() const {
        static int i; return ++i;
    }
};
```
In all instances, the same operator is above.

Pre-lambda mutable state (wrong!)
```cpp
[i=0]() { return ++i; };
```
We get a compiler error for the above code.

We don't write const in lambda, but the `operator()` becomes
`const` by default. In this case, we don't want `const` since we are
incrementing the `i` value. Therefore the compiler throws a compiler error.
Now, how to remove the const? Therefore we need to `mutate`. 
```cpp
[i=0]() mutable { return ++i; };
```
- `mutable` does not affect the constness of the data members themselves.
- It just affects the const qualification of the lambda type's operator.

### Generic Lambdas

**Lambdas + Templates = Generic Lambdas**

Consider the following class member function templates
```cpp
class Plus {
    int value;
public:
    Plus(int v): value(v) {}
    template <class T>
    auto operator() (T x) const {
        return x + value;
    }
};
auto plus = Plus(1);
assert(plus(42) == 43);
```
Generic lambdas reduce boilerplate. This can be written as:
```cpp
auto plus = [value=1](auto x) { return x + value; };
```
`auto` has something to do with type deduction... type inference...
This `auto` is different from auto in a normal case. This is actually a template.

#### Naming the parameter type in generic lambdas
- How to name template `T` in generic lambda? replace it with `auto`?
Consider the following code:
```cpp
auto plus = [](auto... args) {
    return sum(args...);
};
auto times = [](auto&&... args) {
    return product(std::forward<???>(args)...);
};
```
Possible solutions will be:
```cpp
auto one = [](auto&&... args) {
    return product(std::forward<decltype(args)>(args)...);
};
auto two = []<class... Ts>(Ts&&... args) {
    return product(std::forward<Ts>(args)...);
};
auto genericLambda = [](auto x) {
    using T = decltype(x);
    // code here
};
```

### Variadic Lambdas

Variadic function templates
```cpp
class Plus {
    int value;
public:
    Plus(int v);
    template <class... As>
    auto operator()(As... as) {
        return sum(as..., value);
    }
};
```
Objects which I can call with any number of arguments.

Variadic lambdas reduce boilerplate
```cpp
auto plus = [value=1](auto... as) {
    return sum(as..., value);
};
assert(plus(42, 3.14, 1) == 47.14);
```

### Using `this` as the lambda objects

We can explicitly access the lambda as `this->t`.
This works the same way on lambda. Not really!
```cpp
class Widget {
    void work(int);

    void synchronous_foo(int x) {
        this->work(x);
    }

    void asynchronous_foo(int x) {
        fire_and_forge([=]() {
            this->work(x);
        });
    }
};
```
If `this` here means the lambda object, we couldn't compile it.
We can capture `this` in lambdas in the following ways:
- [=] () { this->work(); }
- [this] () { this->work(); }
    - equivalent to [ptr=this] () { ptr->work(); }
- [&] () { this->work(); }
    - equivalent to [ptr=this] () { ptr->work(); }
- [*this] () { this->work(x); }
    - equivalent to [obj=*this] () { obj.work(); }
- Capture `*this by move` has no shorthand.
    - [obj=std::move(*this)] () { obj.work(x); }

### std::function with lambdas

**TODO** the below topics with examples
- **`std::function`** or lambdas? Both are of nearly the same kind.
- How do you write functions that accept lambdas as arguments?
  - STL way
- How to pass lambda across an ABI boundary?

Lambdas may or may non be copyable
```cpp
std::unique_ptr<int> prop;
auto [p = std::move(prop)] () {};
auto lamb2 = std::move(lamb);  // ok
auto lamb3 = lamb;  // error
```
There is no copy constructor for lambda. 

How do you think I could fix it? Place the lambda on the heap, then share
access to it from all the instances of the `std::function`.
Or use any function type.

---

Thank you very much for being with me throughout the posts.
If you want to share some cool stuff or have questions,
please share in the comment section!
