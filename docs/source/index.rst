===============
PyTorch Backend
===============

Overview
=========


Introduction
------------

The PyTorch Backend Project is an initiative aimed at simplifying the integration of various GPU 
devices with the PyTorch framework. This project provides a foundational layer that abstracts 
and standardizes the interaction between PyTorch and different types of GPU hardware. 
It allows developers to focus on specific implementations required for their particular 
GPU devices, without needing to delve into the complexities of the entire integration process.

Key Features
------------

1. **Unified Abstraction Layer**: The project offers a common interface for all GPU devices, 
ensuring that the core PyTorch codebase can interact seamlessly with a wide variety of hardware. 
This abstraction layer manages the intricacies of different GPU architectures and APIs.

2. **Base Level Implementation**: It includes base implementations for common GPU functionalities, 
such as memory management, kernel execution, and data transfer. These implementations 
serve as a foundation, reducing the amount of boilerplate code that developers need to write.

3. **Customizable Integration**: While the project provides comprehensive base-level support, 
it also offers hooks and interfaces for users to implement custom features and optimizations 
specific to their hardware. This flexibility is crucial for leveraging the unique capabilities 
of different GPU architectures.

4. **Scalability and Performance**: The project is designed with scalability in mind, 
supporting efficient operation across a wide range of GPU configurations, from single-device 
setups to large-scale multi-GPU systems. It aims to optimize performance by minimizing overhead 
and maximizing hardware utilization.

5. **Community and Support**: As part of the PyTorch ecosystem, the backend project benefits 
from a vibrant community of developers and researchers. This community contributes to 
continuous improvements, bug fixes, and the addition of new features.

Benefits to Users
---------------

1. **Ease of Integration**: By abstracting the low-level details, the project allows users to 
integrate new GPU devices with minimal effort. This is particularly beneficial for organizations 
that need to support diverse hardware environments.

2. **Focus on Innovation**: Developers can concentrate on implementing cutting-edge algorithms 
and optimizations, rather than dealing with hardware-specific intricacies.

3. **Consistent API**: The unified API ensures that code written for one type of GPU can be easily 
adapted to work with others, enhancing code portability and reuse.

Getting Started
===============

To start using the PyTorch Backend Project, users can refer to the comprehensive documentation 
provided. This includes detailed guides on setting up the environment, integrating new devices, 
and best practices for optimizing performance.

The PyTorch Backend Project is a critical component of the PyTorch ecosystem, enabling seamless 
GPU integration and accelerating the development of high-performance machine learning applications.


Project Structure
----------------


.. code-block:: shell

    .
    |-- codegen               //代码生成：包括前反向注册，反向实现，反向绑定，自定义算子路由，重定向路由等
    |   |-- autograd
    |   |   `-- templates     //TODO: will be moved into csrc/aten
    |   `-- templates         //TODO: will be moved into csrc/aten
    |-- csrc                  //与PyTorch相关的C++实现，不涉及具体后端的实现代码，理论上仅包含后端接口调用
    |   |-- api               //libtorch功能
    |   |-- aten              //代码生成：仅包含 wrap 以及 PyTorch算子注册；后续考虑把 Tensor & Storage & Serialization 移动到这里，因为这三个都是Tensor的相关逻辑
    |   |-- core              //设备无关代码：通用内存池，各种功能模块的基类
    |   |-- distributed       //分布式
    |   `-- npu               //npu的PyTorch相关实现，理论上仅包含接口调用（后续如果能抽象成功，将改名成通用后端：backend之类）
    |-- npu                   //与具体设备强相关，几乎不涉及PyTorch相关概念（除了ATen之外），为csrc提供底层API(后续计划移动到third_party中)
    |   |-- acl
    |   |   |-- include
    |   |   `-- libs
    |   |-- aten
    |   |   |-- common
    |   |   |-- mirror
    |   |   `-- ops
    |   |-- core
    |   |   |-- interface
    |   |   |-- register
    |   |   `-- sys_ctrl
    |   `-- framework
    |       |-- aoe
    |       |-- autograd
    |       |-- contiguous
    |       |-- interface
    |       `-- utils
    |-- third_party
    |   |-- googletest
    |   `-- op-plugin     //后续计划移动到third_party/npu中，但是意义不大，依然是本体项目的submodule
    `-- torch_npu         //Pytorch的Python接口实现
        |-- csrc          //Python&C++绑定
        |   |-- core      //通用能力，仅为Python提供
        |   `-- npu       //面相Python需要暴露的所有底层能力的Python绑定
        |-- meta          //meta算子注册（用来计算自定义算子的输出内存占用）
        |-- npu           //npu设备相关功能模块的封装
        |-- testing
        `-- utils         //各种patch


Modules you may concerned
-------------------------
Following modules you may want to switch to your deivce implementation in order to make it works.




Documents
=============

API Documents
-------------

`C++ API <./cpp_html/index.html>`_
