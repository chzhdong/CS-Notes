---
title: Inference Tools
published: 2024-12-19
description: 'The Local Inference for LLMs with existing repositories'
image: ''
tags: ["Inference"]
category: 'Tool'
draft: false 
lang: 'zh_CN'
---

## 开源方案部署概述

本文主要基于已有的大模型、Github仓库、相关开源工具，实现大模型在本地、远程的部署，以及各类Prompt的工程实践和推理

## Windows

### Gemma.cpp

通过Scoop安装CMake；

通过 VS 2022 安装 Desktop Development 和 Embeded Development；

```powershell
winget install --id Kitware.CMake
winget install --id Microsoft.VisualStudio.2022.BuildTools --force --override "--passive --wait --add Microsoft.VisualStudio.Workload.VCTools;installRecommended --add Microsoft.VisualStudio.Component.VC.Llvm.Clang --add Microsoft.VisualStudio.Component.VC.Llvm.ClangToolset"
```

3. 在完成环境配置之后，基于上述环境直接编译即可：
````powershell
cmake --preset windows
cmake --build --preset windows -j (number of parallel threads)
````

### Gemma-cpp-python

1. 对于Gemma.cpp，可以直接从源代码安装对应的 Python 库：
```powershell
git clone https://github.com/namtranase/gemma-cpp-python.git
cd gemma-cpp-python
pip install .
```

### Llama-cpp-python

1. 安装支持 CPU 的`llama-cpp-python`：`pip install llama-cpp-python`
2. 安装支持 GPU 的 `llama-cpp-python`：
    * 基础要求：Python、CMake、Git、Visual Studio (Desktop、Embeded)
    * 下载仓库：`git clone --recursive -j8 https://github.com/abetlen/llama-cpp-python.git`
    * 环境变量：`set FORCE_CMAKE=1 && set CMAKE_ARGS=-DGGML_CUDA=ON && set CMAKE_ARGS=-DCMAKE_CUDA_COMPILER=nvcc_path`
    * 配置文件：复制 Cuda 文件到 VS Build目录中！
    * 编译文件：到代码仓库中编译文件 `python -m pip install -e .`

注：hugging face 或者 kaggle下载模型的权重文件，目前推荐选用量化的级别为Q6_K的大模型; Nivida Cuda Version > 12.4
