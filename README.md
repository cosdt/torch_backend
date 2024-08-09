# PyTorch Backend

## Project Objective ([Implementation of PyTorch RFC 37](https://github.com/pytorch/rfcs/pull/64)):

  - **Bridging and Integration**: Construct a device-agnostic layer that promotes a unified interface on the upper layer and ensures compatibility with various hardware on the lower layer, shielding PyTorch from direct awareness of multiple backends.
  - **Low cost Integration**: Provide device abstraction layer to accelerate new backend integration by only implementing few interfaces, offer comprehensive integration documentation, provide integrate implementations as a reference (CUDA/CPU/NPU) and general test cases and contract tests.
  - **Quality Assurance**: Maintain quality through CI/CD for the integration mechanism of third-party devices based on PrivateUse1.
  - **Mainstream Approach**: Promote the integration mechanism of third-party devices based on PrivateUse1 as the mainstream approach for integrating new backends into PyTorch in the future.

## Current Progress:

  - [x] **Runtime**: Completed components include Device, Stream, Event, Generator, Guard, and Allocator.
  - [x] **AMP**: Registration and API have been completed.
  - [x] **Operators**: Migrated NPU operator list and codegen. The next steps will involve operator simplification and codegen refactoring.

## Next Steps:

  - [ ] **Device-agnostic**: Complete the device-agnostic layer; organize specific device logic according to different device type (e.g., backends/cuda, backends/cpu, backends/...). Making it as submodule in the future.
  - [ ] **CodeGen**: Enhance and refactor codegen module, providing general and reusable code generation capabilities that cover official operators, custom operators, routing code, forward and backward binding, etc.
  - [ ] **Operators**: Simplify operators, implement all factory class operators (as operator implementation reference), as well as functional operators (for testing the functionality of the third-party device integration mechanism).
  - [ ] **Tests & Docs**: Complete general test case suites, the full module integration and API documentation.
  - [ ] **Live Demo**: Integrate CUDA/CPU into PyTorch based on this project and provide a full-process integration tutorial.


------------------

## Getting Started

To start using the PyTorch Backend Project, users can refer to the [comprehensive documentation](https://cosdt.github.io/torch_backend/)
provided. This includes detailed guides on setting up the environment, integrating new devices,
and best practices for optimizing performance.

### Project Structure

```bash
    .
    ├── backends
    │   ├── fake               // dummy backend: provide all weak symbols needed by csrc, we can run this demo without implementing all symbols in REAL Backend by this fake backend.
    │   ├── npu                // one of REAL Backend: provide API and Structure related witch specific Backends strongly
    │   ├── cuda               // one of REAL Backend: will be implemented later
    │   └── ...
    ├── cmake
    ├── codegen                // Code generation: includes registration for forward and backward, backward implementation, backward binding, custom operator routing, reroute routing, etc.
    │   ├── autograd
    │   │   └── templates      // General template
    │   └── templates
    ├── csrc                   // C++ implementations related to PyTorch, not involving specific backend implementations, theoretically only includes backend interface calls
    │   ├── api                // libtorch functionalities
    │   ├── aten               // Code generation: includes only wrap and PyTorch operator registration; in the future, considering moving Tensor & Storage & Serialization here, as these three are related to Tensor logic
    │   ├── backend            // General Implementation of PyTorch API
    │   ├── core               // Common Utils
    │   │   ├── allocator
    │   │   ├── generator
    │   │   └── guard
    │   └── distributed        // Distributed
    ├── docs                   // All docs: C++ API, Python API and E2E tutorials
    │   ├── cpp
    │   │   └── source
    │   └── source
    ├── test                   // General TestCase Sets: including C++ and python
    │   └── cpp
    │       ├── backend
    │       ├── common
    │       └── core
    ├── third_party
    │   └── googletest
    └── torch_backend          // Python interface implementation for PyTorch
        ├── backend
        ├── csrc               // Python & C++ binding
        │   ├── backend        // Python bindings for all low-level capabilities needed to be exposed to Python
        │   └── core           // General capabilities, only provided for Python
        └── meta               // Meta operator registration
```

## Documents

### API Documents

[C++ API](https://cosdt.github.io/torch_backend/cpp_html/index.html)

## License

PyTorch Backend has a BSD-style license, as found in the [LICENSE](LICENSE) file.
