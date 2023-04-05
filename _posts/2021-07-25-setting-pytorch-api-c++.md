---
layout: post
title: Setting up Microsoft Visual Studio for PyTorch C++ API
date: 2021-07-24
category: PyTorch
tags: 
- pytorch
redirect_from:
- /pytorch/2021/07/25/setting-pytorch-api-c++/
- /setting-pytorch-api-c++.html
---

{% highlight python %}
{% endhighlight %}

<script type="text/javascript" async
    src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

Hi everyone! Today we are going to set up Microsoft Visual Studio in Windows for PyTorch C++ API. As per me its the easiest way to start up coding in PyTorch using C++. Major advantage of using Microsoft Visual Studio is that you can avoid playing around CMake files, and passing the paths via command line (though this can be done in CMake as well, but it still can be a hustle for some). So, let's dive into the installation process!

### Step 1: Download LibTorch.

We will download LibTorch from [here](https://pytorch.org/get-started/locally/) . Our first task is to select and download the right binary according to the system we are using. 
So basically PyTorch has three Build options: Stable version, Preview(Nighty) and LTS(1.8.1). We'll be downloading stable version of PyTorch Build. Stable version means that it is a latest tested and supported version in PyTorch. LibTorch in windows has two available versions to download: Release & Debug.
Debug versions are those with extra information in the binary which helps with debugging, while for release versions compilers are not obliged to keep this information and remove it to provide a leaner binary. I'll be downloading debug version because it would be helpful for me for development as debugging codes using these versions is more easier. If you want fast result and you are aiming to work at production level please go on with release version.  

![0](https://user-images.githubusercontent.com/62256509/124362052-6088bb00-dc50-11eb-8184-2f475bc4dcf6.png)

### Step 2: Set Environment Variables.

* We can set up environment variables in Windows from command prompt using `setx` or `set` command. `setx` command is used to set path of variables of system globally while `set` command is used to set path for current session. Now, Open your command prompt and type these commands. I'll be setting up paths globally, therefore using `setx` command

```bash
setx LIBTORCH_DEBUG_HOME C:\Users\91939\Downloads\libtorch-win-shared-with-deps-debug-1.8.1+cpu\libtorch
```

Here, "C:\Users\91939\Downloads\libtorch-win-shared-with-deps-debug-1.8.1+cpu\libtorch" is the path of extracted folder of LibTorch.

* Now, we have to build PyTorch C++ extensions in Visual Studio for this we have to include Python header files to our Visual Studio. I will be setting up paths of variable globally. Type this command in command prompt

```bash
setx PYTHON_HOME "C:\Program Files (x86)\Microsoft Visual Studio\Shared\Python37_64"
```

Here, "C:\Program Files (x86)\Microsoft Visual Studio\Shared\Python37_64" is the path of Python header files. Please note that the Python version can change depending on your environment, the path is expected to stay same except the version differences but still you should explicitly check if it's the correct one.

![1](https://user-images.githubusercontent.com/62256509/124362070-74ccb800-dc50-11eb-93cd-6c2fcf80b6f3.png)

### Step 3: Configuring Visual Studio Project.

* Create a new project in Microsoft Visual Studio. Be sure to set active solution configuration to `Debug` (as we downloaded debug version) and active solution platform to `x64`. Then create a simple C++ file just to check whether its working or not.
![2](https://user-images.githubusercontent.com/62256509/124362102-a2b1fc80-dc50-11eb-8998-33a33dd92679.png)

* Our first step is to add include paths to our projects. In **project properties** go to `C/C++ -> General -> Additional Include Directories` and add the following paths:

```bash
* C:\Users\91939\Downloads\libtorch-win-shared-with-deps-debug-1.8.1+cpu\libtorch\include
* C:\Users\91939\Downloads\libtorch-win-shared-with-deps-debug-1.8.1+cpu\libtorch\include\torch\csrc\api\include
* C:\Program Files (x86)\Microsoft Visual Studio\Shared\Python37_64\include
```
![4](https://user-images.githubusercontent.com/62256509/124362124-c2492500-dc50-11eb-82a0-9dbfa5259908.png)

* Now, under **Linker** go to `General -> Additional Library Directories` and add the following additional library directories.

```bash
* C:\Users\91939\Downloads\libtorch-win-shared-with-deps-debug-1.8.1+cpu\libtorch\lib
* C:\Program Files (x86)\Microsoft Visual Studio\Shared\Python37_64\libs
```
![5](https://user-images.githubusercontent.com/62256509/124362132-d12fd780-dc50-11eb-817b-5feb5cef40b2.png)

In the same (Linker file) go to `Input -> Additional Dependencies ` to add more libraries as follows:

```bash
torch.lib 
torch_cpu.lib 
c10.lib 
python37.lib
```
![6](https://user-images.githubusercontent.com/62256509/124362382-13a5e400-dc52-11eb-827c-5b8b7b9ca65d.png)

* Go to C/C++, under language change conformance mode to No. This setting is to prevent some errors [`std`: ambiguous symbol].
![7](https://user-images.githubusercontent.com/62256509/124362389-17d20180-dc52-11eb-98f0-cf1e03b1918c.png)

* Till here, we are almost done with our setup. The only thing left is to build our project. Hard copy all the `dll` commands from `C:\Users\91939\Downloads\libtorch-win-shared-with-deps-debug-1.8.1+cpu\libtorch\lib` to ` x64->Debug` of our project.  Since, we downloaded the debug version of LibTorch, hence our project mode is also set to Debug. This can change depending on the user's requirements. 

### Step 4: Run

Run the following code:

```bash
#include <torch/torch.h>
#include <iostream>

int main() {
	torch::Tensor tensor = torch::ones(5);
	std::cout << tensor << std::endl;
	return 0;
}
```
Now, build the project and run the code. We'll get:
![8](https://user-images.githubusercontent.com/62256509/124362418-4354ec00-dc52-11eb-8e6a-9b45f3565180.png)

### Acknowledgment

Special thanks to [Kshitij Kalambarkar](https://github.com/kshitij12345) and [Kushashwa Ravi Shrimali](https://github.com/krshrimali) for your valuable suggestions, feedback and for helping me. Cheers to you guys!

### References

* https://pytorch.org/get-started/locally/
* https://raminnabati.com/post/004_adv_pytorch_integrating_pytorch_cpp_frontend_in_visual_studio_on_windows/
