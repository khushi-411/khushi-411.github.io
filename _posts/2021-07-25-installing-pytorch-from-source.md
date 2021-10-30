---
layout: post
title: Installing PyTorch from Source
date: 2021-07-25
category: pytorch
tags: pytorch
redirect_from:
: /pytorch/2021/07/25/installing-pytorch-from-source/
: /installing-pytorch-from-source.html
---

{% highlight python %}
{% endhighlight %}

<script type="text/javascript" async
    src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>


Hi Everyone! In this blog post, we'll be discussing setting up a developer environment for PyTorch. Our goal is to build PyTorch from source on any of the Linux distributions. Presently, I'm working on the Ubuntu 20.04 distribution of Linux in WSL (Windows Subsystem Linux). You can install any other Ubuntu version of your choice, the procedure will be the same because all the Ubuntu versions are Debian-based. Enjoy the installation process! Yey!!  

### Step 1: Installing Anaconda

I prefer to install Anaconda first because things become easy to install. You can download Anaconda from [here](https://www.anaconda.com/products/individual#download-section). The main advantage to installing Anaconda is that besides controlled dependency versions, we will get a high-quality BLAS library (MKL) which is used to accelerate various math functions and applications. 

After downloading Anaconda3 we have to run the following command:

```bash
chmod +x Anaconda3-2021.05-Linux-x86_64.sh
```

Here, the `Anaconda3-2021.05-Linux-x86_64.sh` is the name of the downloaded Anaconda3 installer. `chmod` command is used to change the access mode of our file and `+x` is used to make files executable. Anaconda3 installer is downloaded in shell script file format, so we need to change them to exceutable files.

To install Anaconda, use the command:

```bash
./Anaconda3-2021.05-Linux-x86_64.sh
```

Now, confirm the license agreement and continue the following procedure. After that, I advise you to either refresh or restart your terminal. I will be restarting my shell using the `exec bash` command. In case you are using a different shell than `bash`, like `zsh` - you can do: `exec zsh`.

### Step 2: Installing PyTorch 

Our next task is to create a PyTorch development environment, run the following code:

```bash
conda create --name pytorch-dev python=3.8.10
```

`pytorch-dev` is just the name of the environment, I want to create. You can choose yours! We need to activate our development environment use:

```bash
conda activate pytorch-dev
```

Now, we have to install PyTorch from the source, use the following command:

```bash
conda install astunparse numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses
```

**Note:** Step 3, Step 4 and Step 5 are not mandatory, install only if your laptop has GPU with CUDA support. 

### Step 3: Install CUDA 

We will be compiling our code using CUDA, we will be installing the required toolkit from [here](https://developer.nvidia.com/cuda-downloads). I'll be downloading CUDA Toolkit 11.2.2 version. These options are selected as per my requirement:

![1](https://user-images.githubusercontent.com/62256509/126878857-9e3c6f00-5904-4b05-bff2-8e15b9c9eaa5.png)

Now, run the following commands to install CUDA on your machine.

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.2.2/local_installers/cuda-repo-wsl-ubuntu-11-2-local_11.2.2-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-11-2-local_11.2.2-1_amd64.deb
sudo apt-key add /var/cuda-repo-wsl-ubuntu-11-2-local/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda
```

For Linux:

```bash
conda install -c pytorch magma-cuda112
```

Before installing check the CUDA version you have installed, then install. 

### Step 4: Install cuDNN

This is NVIDIA's `CUDA Deep Neural Network Library` (cuDNN). It is a GPU accelerated library. The main advantage of using these libraries are:

* Accelerates deep learning frameworks.
* Provides better implementations for convolution, pooling, normalizations, and activation layers.

Visit [this](https://developer.nvidia.com/cudnn) link to download cuDNN. Check out which version you'll need for the installed CUDA. I installed `CUDA 11.2.2`, therefore I'll be installing the `cuDNN v8.2.1` version as it's compatible with that.

Now, we need to extract our `cuDNN` package installed, use the following command:

```bash
tar -xvzf cudnn-11.3-linux-x64-v8.2.1.32.tgz
```
I created a folder named `cudnn` and extracted the `cuDNN` file there.

### Step 5: Setting Paths

We can check whether CUDA is installed or not using `ls /usr/local/cuda` command. To set the path use:

```bash
export PATH=/usr/local/cuda/bin/:$PATH
```

Setting the path for `lib` folder of CUDA using:

```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

We can check whether the `CUDA compiler` works or not using the `nvcc` command.

Last step to go! We just need to copy all the `include` and `lib64` header files to the `CUDA` folder. First, we have to go to the file where we have saved `cudnn` files then run the following commands:

```bash
sudo cp -r cuda/include/cudnn* /usr/local/cuda/include
sudo cp -r cuda/lib64/libcudnn* /usr/local/cuda/lib64
```

### Step 6: Installing Pytorch

Now, it is ready to use, we just have to clone the PyTorch directory. We will run the following command:

```bash
git clone https://github.com/pytorch/pytorch
```

Let's update our submodule using:

```bash
git submodule sync
git submodule update --init --recursive
```

One more step to go!!! Install PyTorch on Linux by running the following command:

```bash
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
python3 setup.py develop
```

I'll be installing for development purposes, therefore used `develop` in the command line. Use can also use `install` instead of `develop` to install `setup.py`.

To check CMAKE_PREFIX_PATH, use: `echo $CMAKE_PREFIX_PATH` command.

Building PyTorch can take a long, so be patient! Hurrah!!! We succeeded in installing PyTorch from the source. 

### Acknowledgment

I'd love to thank [Kshitij Kalambarkar](https://github.com/kshitij12345) and [Kushashwa Ravi Shrimali](https://github.com/krshrimali) for their useful suggestions, feedback, and help. Cheers to you guys!

**References:**

* [https://github.com/pytorch/pytorch](https://github.com/pytorch/pytorch)
* [https://www.youtube.com/watch?v=AIkvOtHJZeo&list=PLFYhn53SnsgJI2jjjsV0bZu3tdfi361wv&index=1](https://www.youtube.com/watch?v=AIkvOtHJZeo&list=PLFYhn53SnsgJI2jjjsV0bZu3tdfi361wv&index=1)
