# Llama.cpp

## 项目简介

Llama.cpp 是基于 C++ 的大模型推理框架，其主要的目的是为了减少依赖以及最小化启动的配置，并且在各个平台上实现 SOTA 的性能表现。这也是为什么框架会选择 C++ 作为主要开发语言的原因。Llama.cpp 主要是基于作者自己开发的 GGML 库进行拓展，以实现一系列高性能的系统运行。

## 框架安装

Llama.cpp Python 是基于 llama.cpp 进行开发所实现的 Python 库，以实现 Python 语言的开发调用！

### Llama.cpp Python

1. 安装支持 CPU 的`llama-cpp-python`：`pip install llama-cpp-python`
2. 安装支持 GPU 的 `llama-cpp-python`：
   * 基础要求：Python、CMake、Git、Visual Studio (Desktop、Embeded)
   * 下载仓库：`git clone --recursive -j8 https://github.com/abetlen/llama-cpp-python.git`
   * 环境变量：`set FORCE_CMAKE=1 && set CMAKE_ARGS=-DGGML_CUDA=ON && set CMAKE_ARGS=-DCMAKE_CUDA_COMPILER=nvcc_path`
   * 配置文件：复制 Cuda 文件到 VS Build目录中！
   * 编译文件：到代码仓库中编译文件 `python -m pip install -e .`

注：hugging face 或者 kaggle下载模型的权重文件，目前推荐选用量化的级别为Q6_K的大模型; Nivida Cuda Version > 12.4

## 拓展内容

## Gemma.cpp

Gemma.cpp 是 Google 团队基于 Llama.cpp 的思想进行开发，针对其旗下的 Gemma 大模型实现的推理框架。

1. 通过Scoop安装CMake；
2. 通过 VS 2022 安装 Desktop Development 和 Embeded Development；

```powershell
winget install --id Kitware.CMake
winget install --id Microsoft.VisualStudio.2022.BuildTools --force --override "--passive --wait --add Microsoft.VisualStudio.Workload.VCTools;installRecommended --add Microsoft.VisualStudio.Component.VC.Llvm.Clang --add Microsoft.VisualStudio.Component.VC.Llvm.ClangToolset"
```

3. 在完成环境配置之后，基于上述环境直接编译即可：
````powershell
cmake --preset windows
cmake --build --preset windows -j (number of parallel threads)
````

### Gemma.cpp Python

对于Gemma.cpp，可以直接从源代码安装对应的 Python 库：
```powershell
git clone https://github.com/namtranase/gemma-cpp-python.git
cd gemma-cpp-python
pip install .
```

