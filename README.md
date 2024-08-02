# PyTorch Backend

## Overview

### Introduction

The PyTorch Backend Project is an initiative aimed at simplifying the integration of various GPU
devices with the PyTorch framework. This project provides a foundational layer that abstracts
and standardizes the interaction between PyTorch and different types of GPU hardware.
It allows developers to focus on specific implementations required for their particular
GPU devices, without needing to delve into the complexities of the entire integration process.

### Key Features

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

### Benefits to Users

1. **Ease of Integration**: By abstracting the low-level details, the project allows users to
integrate new GPU devices with minimal effort. This is particularly beneficial for organizations
that need to support diverse hardware environments.

2. **Focus on Innovation**: Developers can concentrate on implementing cutting-edge algorithms
and optimizations, rather than dealing with hardware-specific intricacies.

3. **Consistent API**: The unified API ensures that code written for one type of GPU can be easily
adapted to work with others, enhancing code portability and reuse.

## Getting Started

To start using the PyTorch Backend Project, users can refer to the comprehensive documentation
provided. This includes detailed guides on setting up the environment, integrating new devices,
and best practices for optimizing performance.

The PyTorch Backend Project is a critical component of the PyTorch ecosystem, enabling seamless
GPU integration and accelerating the development of high-performance machine learning applications.

### Project Structure

```bash
    .
    |-- codegen               // Code generation: includes registration for forward and backward, backward implementation, backward binding, custom operator routing, reroute routing, etc.
    |   |-- autograd
    |   |   `-- templates
    |   `-- templates
    |-- csrc                  // C++ implementations related to PyTorch, not involving specific backend implementations, theoretically only includes backend interface calls
    |   |-- api               // libtorch functionalities
    |   |-- aten              // Code generation: includes only wrap and PyTorch operator registration; in the future, considering moving Tensor & Storage & Serialization here, as these three are related to Tensor logic
    |   |-- core              // Device-independent code: general memory pool, base classes for various functional modules
    |   |-- distributed       // Distributed
    |   `-- npu               // NPU-related implementations in PyTorch, theoretically only includes interface calls
    |-- npu                   // Strongly device-specific, barely involves PyTorch-related concepts (except ATen), provides low-level APIs for csrc
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
    |   `-- op-plugin         // Device-specific
    `-- torch_npu             // Python interface implementation for PyTorch
        |-- csrc              // Python & C++ binding
        |   |-- core          // General capabilities, only provided for Python
        |   `-- npu           // Python bindings for all low-level capabilities needed to be exposed to Python for NPU
        |-- meta              // Meta operator registration
        |-- npu               // Encapsulation of functional modules related to NPU devices
        |-- testing
        `-- utils
```

### Modules you may concerned

Following modules you may want to switch to your deivce specific implementation in order to make it works.


## Documents

### API Documents

[C++ API](https://cosdt.github.io/torch_npu/cpp_html/index.html)


## License

PyTorch Backend has a BSD-style license, as found in the [LICENSE](LICENSE) file.
