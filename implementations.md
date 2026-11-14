# FORGE

## Technical Specification & Phase 01 Implementation Plan

**Version:** 0.1  
**Status:** Initial specification  
**Architecture:** C++ Inference Runtime + Go Serving Layer  
**Execution:** CPU-first  
**Implementation:** From scratch

---

# PART I — TECHNICAL SPECIFICATION

## 1. Project Vision

Forge is a CPU-based Large Language Model (LLM) inference engine written in C++ with a Go-based inference server.

Its purpose is to load pretrained language models, execute transformer computations, manage memory efficiently, and serve inference requests concurrently.

The project aims to implement the fundamental components of an inference engine without relying on existing inference runtimes such as:

- llama.cpp
- vLLM
- Ollama
- TensorRT-LLM
- Hugging Face Transformers
- PyTorch inference

We may study their architectures and engineering decisions, but the implementation of Forge's core runtime will be our own.

### 1.1 Primary objectives

1. Understand and implement tensor operations.
2. Build a CPU execution backend.
3. Implement transformer inference.
4. Implement attention and KV caching.
5. Support pretrained model loading.
6. Implement autoregressive token generation.
7. Build a concurrent inference server in Go.
8. Optimize memory usage and execution throughput.

### 1.2 Non-objectives

Forge will not initially:

- Train or fine-tune models.
- Support distributed inference.
- Support GPU execution.
- Implement multimodal models.
- Provide a web-based chat interface.
- Depend on Python for runtime execution.

Python may be used for independent numerical verification and test-data generation, but not for the inference runtime.

---

# 2. System Architecture

Forge will consist of two independent subsystems.

## 2.1 C++ Inference Runtime

The C++ runtime is responsible for everything involving model execution.

Responsibilities:

- Tensor allocation and memory management.
- Numerical operations.
- Model weight loading.
- Tokenization.
- Transformer execution.
- Attention computation.
- KV cache management.
- Sampling and token generation.
- CPU optimization.

## 2.2 Go Inference Server

The Go server is responsible for accepting and managing inference requests.

Responsibilities:

- HTTP API.
- Request validation.
- Request lifecycle management.
- Scheduling.
- Concurrent request management.
- Streaming generated tokens.
- Metrics and logging.

The Go server must never directly manipulate tensor memory.

## 2.3 Communication boundary

Initially, we will expose the C++ runtime through a C-compatible API.

Go will communicate with it through a narrow native interface, initially using cgo.

This avoids introducing a network protocol between two components running in the same process.

Later, we can evaluate whether a separate runtime process and IPC protocol would provide useful isolation.

## 2.4 High-level architecture

```text
                  CLIENT
                    |
                    |
             HTTP / SSE
                    |
                    v
          +-------------------+
          |    GO SERVER      |
          |                   |
          | Request Handler   |
          | Scheduler         |
          | Queue Management  |
          | Streaming          |
          +-------------------+
                    |
                    | C ABI / cgo
                    |
                    v
          +-------------------+
          | C++ RUNTIME       |
          |                   |
          | Model Manager     |
          | Tokenizer         |
          | Tensor Engine     |
          | Transformer       |
          | KV Cache          |
          | Sampler           |
          +-------------------+
                    |
                    v
          +-------------------+
          | CPU EXECUTION     |
          |                   |
          | Memory            |
          | SIMD              |
          | Multithreading    |
          +-------------------+
```

---

# 3. Technology Decisions

| Component              | Technology            |
| ---------------------- | --------------------- |
| Runtime                | C++20                 |
| Server                 | Go                    |
| Build system           | CMake                 |
| Compiler               | Clang / GCC           |
| Testing (C++)          | GoogleTest            |
| Testing (Go)           | Go testing            |
| Memory management      | RAII, smart pointers  |
| API                    | HTTP/JSON             |
| Streaming              | Server-Sent Events    |
| Initial numerical type | FP32                  |
| Model format (target)  | GGUF                  |
| Operating system       | macOS/Linux initially |
| CI                     | GitHub Actions        |

### Important decisions

**Why C++20?**

- Modern memory management.
- RAII and deterministic resource ownership.
- Templates and compile-time abstractions.
- Strong control over memory layout.
- Standard concurrency primitives.

**Why Go?**

- Lightweight concurrency using goroutines.
- Excellent HTTP support.
- Straightforward request lifecycle management.
- Good observability and operational tooling.

**Why GGUF eventually?**

GGUF provides a practical path to loading existing pretrained models without depending on a Python model-loading framework.

We will implement the necessary format reader ourselves.

---

# 4. Model Architecture

The first supported architecture will be a decoder-only Transformer based on the Llama architecture.

This gives us a well-defined target rather than attempting to support every model architecture immediately.

## 4.1 Transformer execution pipeline

```text
Input Text
    |
    v
Tokenizer
    |
    v
Token IDs
    |
    v
Token Embeddings
    |
    v
Transformer Blocks
    |
    +--> RMSNorm
    |
    +--> Self Attention
    |       |
    |       +--> Q, K, V projections
    |       +--> RoPE
    |       +--> Attention scores
    |       +--> Softmax
    |       +--> Weighted values
    |
    +--> Residual Connection
    |
    +--> Feed Forward Network
    |       |
    |       +--> Gate projection
    |       +--> Up projection
    |       +--> SiLU
    |       +--> Down projection
    |
    +--> Residual Connection
    |
    v
Final RMSNorm
    |
    v
Language Model Head
    |
    v
Logits
    |
    v
Sampling
    |
    v
Next Token
```

The transformer block will be implemented incrementally.

## 4.2 Mathematical operations required

The runtime must eventually support:

- Matrix multiplication.
- Vector addition.
- Element-wise multiplication.
- Scalar multiplication.
- RMSNorm.
- Softmax.
- SiLU activation.
- Rotary Positional Embeddings (RoPE).
- Embedding lookup.
- Attention computation.
- Residual connections.

## 4.3 Inference process

Forge will support the two fundamental stages of autoregressive inference.

**Prefill**

The model processes the complete input prompt and constructs the KV cache.

**Decode**

The model processes one new token at a time, reusing the cached keys and values from previous tokens.

We will optimize these stages separately because their computational characteristics differ significantly.

---

# 5. Repository Structure

The repository will be organized around clear module boundaries.

```text
forge/
|
+-- CMakeLists.txt
+-- README.md
+-- LICENSE
+-- .gitignore
|
+-- docs/
|   +-- architecture.md
|   +-- tensor-design.md
|   +-- inference-pipeline.md
|   +-- decisions/
|
+-- runtime/
|   +-- CMakeLists.txt
|   |
|   +-- include/
|   |   +-- forge/
|   |       +-- core/
|   |       +-- tensor/
|   |       +-- ops/
|   |       +-- memory/
|   |       +-- tokenizer/
|   |       +-- model/
|   |       +-- transformer/
|   |       +-- attention/
|   |       +-- inference/
|   |       +-- sampling/
|   |
|   +-- src/
|   |   +-- core/
|   |   +-- tensor/
|   |   +-- ops/
|   |   +-- memory/
|   |   +-- tokenizer/
|   |   +-- model/
|   |   +-- transformer/
|   |   +-- attention/
|   |   +-- inference/
|   |   +-- sampling/
|   |
|   +-- tests/
|   |   +-- tensor/
|   |   +-- ops/
|   |   +-- memory/
|   |
|   +-- benchmarks/
|
+-- server/
|   +-- cmd/
|   +-- internal/
|   |   +-- api/
|   |   +-- scheduler/
|   |   +-- runtime/
|   |   +-- streaming/
|   |   +-- metrics/
|   +-- go.mod
|   +-- tests/
|
+-- examples/
|
+-- scripts/
|
+-- benchmarks/
```

We will not implement every directory immediately. The structure represents the eventual architecture.

---

# PART II — DEVELOPMENT ROADMAP

## 6. Major Development Phases

| Phase | Deliverable                       | Language |
| ----- | --------------------------------- | -------- |
| 01    | Tensor Engine                     | C++      |
| 02    | Numerical Operations Engine       | C++      |
| 03    | Tokenizer                         | C++      |
| 04    | Transformer Building Blocks       | C++      |
| 05    | Attention Mechanism               | C++      |
| 06    | KV Cache                          | C++      |
| 07    | Complete Transformer Forward Pass | C++      |
| 08    | Autoregressive Generation         | C++      |
| 09    | Model Loading and Quantization    | C++      |
| 10    | Runtime C API                     | C++      |
| 11    | Inference Server                  | Go       |
| 12    | Request Scheduling and Batching   | Go       |
| 13    | Concurrent Inference              | Both     |
| 14    | Performance Optimization          | Both     |

Each phase must have:

1. A technical design.
2. Implementation tasks.
3. Unit tests.
4. Integration tests where applicable.
5. Benchmarks where applicable.
6. Documentation.
7. Acceptance criteria.

We will not mark a phase complete merely because its code compiles.

---

# PART III — PHASE 01 IMPLEMENTATION PLAN

# Phase 01: Tensor Engine

**Language:** C++20  
**Status:** Not started  
**Priority:** Critical

## 7. Objective

Build the foundational tensor abstraction that will be used throughout the inference runtime.

A tensor is the fundamental data structure used to represent:

- Model weights.
- Token embeddings.
- Query, Key and Value matrices.
- Attention scores.
- Activations.
- Intermediate transformer outputs.

Before implementing transformers, we need a reliable way to represent and manipulate multidimensional numerical data.

We will build this ourselves.

## 7.1 What is a tensor?

A tensor is a multidimensional numerical array.

Examples:

Scalar:

```text
5
```

Shape: `[]`

Vector:

```text
[1, 2, 3, 4]
```

Shape: `[4]`

Matrix:

```text
[
  [1, 2, 3],
  [4, 5, 6]
]
```

Shape: `[2, 3]`

Three-dimensional tensor:

```text
[
  [
    [1, 2],
    [3, 4]
  ],
  [
    [5, 6],
    [7, 8]
  ]
]
```

Shape: `[2, 2, 2]`

A tensor has three fundamental properties:

1. Shape.
2. Data storage.
3. Data type.

We will also introduce strides because they will become important when implementing efficient tensor views and memory access.

---

# 8. Tensor Design

## 8.1 Initial Tensor API

We will expose an API similar to:

```cpp
Tensor tensor({
    2, 3
});

tensor.set({0, 0}, 1.0f);
tensor.set({0, 1}, 2.0f);
tensor.set({0, 2}, 3.0f);

tensor.set({1, 0}, 4.0f);
tensor.set({1, 1}, 5.0f);
tensor.set({1, 2}, 6.0f);

float value = tensor.get({1, 2});
```

Expected result:

```text
value = 6.0
```

We will also support:

```cpp
tensor.shape();
tensor.strides();
tensor.numel();
tensor.rank();
tensor.data();
tensor.get_flat(0);
tensor.set_flat(0, 10.0f);
```

The exact API will be finalized during implementation.

## 8.2 Memory layout

We will initially use contiguous row-major memory layout.

For a matrix:

```text
[
  [1, 2, 3],
  [4, 5, 6]
]
```

Physical memory:

```text
[1, 2, 3, 4, 5, 6]
```

The address calculation for a two-dimensional tensor is:

```text
offset = row * columns + column
```

For an arbitrary N-dimensional contiguous tensor:

```text
offset = sum(index[i] * stride[i])
```

For a shape:

```text
[2, 3, 4]
```

The contiguous strides are:

```text
[12, 4, 1]
```

This means:

```text
offset = i * 12 + j * 4 + k
```

We will implement this general calculation rather than hardcoding two-dimensional indexing.

## 8.3 Data type

Phase 01 will support only:

```cpp
float
```

That means FP32.

Why?

- Straightforward numerical behavior.
- Easier debugging.
- Simple memory allocation.
- Suitable for validating numerical operations.

Later, we will introduce a data type abstraction for FP16, BF16, INT8 and quantized formats.

## 8.4 Memory ownership

The Tensor class must own its underlying memory safely.

Requirements:

- No manual memory management by consumers.
- RAII-based ownership.
- No dangling pointers.
- Predictable copy semantics.
- Move support.
- Clear behavior for empty tensors.

Initial implementation should use:

```cpp
std::vector<float>
```

We will not build a custom memory allocator in Phase 01.

Custom allocators and memory pools belong to later performance optimization work.

---

# 9. Phase 01 Modules

## Module 1: Shape

Responsible for representing tensor dimensions.

Expected interface:

```cpp
class Shape {
public:
    explicit Shape(std::vector<size_t> dimensions);

    size_t rank() const;
    size_t numel() const;

    const std::vector<size_t>& dims() const;
};
```

Responsibilities:

- Store dimensions.
- Calculate tensor element count.
- Validate dimensions.
- Detect integer overflow.
- Support shape comparisons.

## Module 2: Strides

Responsible for mapping multidimensional indices to flat memory offsets.

Expected interface:

```cpp
class Strides {
public:
    explicit Strides(const Shape& shape);

    size_t offset(
        const std::vector<size_t>& indices
    ) const;

    const std::vector<size_t>& values() const;
};
```

Responsibilities:

- Calculate contiguous strides.
- Convert multidimensional indices into offsets.
- Validate index rank.
- Validate bounds.

## Module 3: Storage

Responsible for owning the underlying tensor data.

Expected interface:

```cpp
class Storage {
public:
    explicit Storage(size_t size);

    float* data();
    const float* data() const;

    size_t size() const;
};
```

Responsibilities:

- Allocate memory.
- Own memory.
- Expose controlled data access.
- Support move semantics.
- Initialize elements deterministically.

## Module 4: Tensor

The main abstraction combining the previous modules.

Expected interface:

```cpp
class Tensor {
public:
    explicit Tensor(Shape shape);

    float get(
        const std::vector<size_t>& indices
    ) const;

    void set(
        const std::vector<size_t>& indices,
        float value
    );

    float* data();
    const float* data() const;

    size_t numel() const;
    size_t rank() const;

    const Shape& shape() const;
    const Strides& strides() const;
};
```

Responsibilities:

- Combine shape and storage.
- Provide multidimensional indexing.
- Provide flat indexing.
- Validate memory access.
- Provide basic tensor inspection.

---

# 10. Implementation Tasks

## Task 01 — Project Infrastructure

- [ ] Create repository.
- [ ] Configure CMake.
- [ ] Set C++20 standard.
- [ ] Configure compiler warnings.
- [ ] Integrate GoogleTest.
- [ ] Create initial directory structure.
- [ ] Add a minimal executable.
- [ ] Add CTest integration.

**Deliverable:** A clean C++ project that builds and runs tests.

## Task 02 — Shape Implementation

- [ ] Implement Shape class.
- [ ] Implement rank().
- [ ] Implement numel().
- [ ] Handle scalar shapes.
- [ ] Reject invalid dimensions.
- [ ] Add overflow protection.
- [ ] Implement shape equality.

**Deliverable:** Tested multidimensional shape representation.

## Task 03 — Stride Implementation

- [ ] Implement contiguous stride calculation.
- [ ] Implement multidimensional offset calculation.
- [ ] Validate index dimensions.
- [ ] Validate index bounds.
- [ ] Test 1D, 2D, 3D and higher-dimensional tensors.

**Deliverable:** Reliable multidimensional-to-flat indexing.

## Task 04 — Storage Implementation

- [ ] Implement Storage.
- [ ] Allocate FP32 memory.
- [ ] Implement RAII ownership.
- [ ] Implement move constructor.
- [ ] Implement move assignment.
- [ ] Define copy behavior.
- [ ] Test memory ownership.

**Deliverable:** Safe tensor memory management.

## Task 05 — Tensor Implementation

- [ ] Implement Tensor constructor.
- [ ] Implement get().
- [ ] Implement set().
- [ ] Implement flat indexing.
- [ ] Implement shape access.
- [ ] Implement stride access.
- [ ] Implement numel().
- [ ] Implement rank().
- [ ] Add tensor initialization utilities.

**Deliverable:** Fully functional FP32 tensor abstraction.

## Task 06 — Tensor Inspection

Implement a simple tensor printing mechanism.

Example:

```cpp
Tensor t({2, 3});

t.set({0, 0}, 1.0f);
t.set({0, 1}, 2.0f);
t.set({0, 2}, 3.0f);
t.set({1, 0}, 4.0f);
t.set({1, 1}, 5.0f);
t.set({1, 2}, 6.0f);

t.print();
```

Expected output:

```text
Tensor(
  shape=[2, 3],
  dtype=float32,
  data=[
    [1, 2, 3],
    [4, 5, 6]
  ]
)
```

## Task 07 — Testing and Validation

- [ ] Test scalar tensors.
- [ ] Test vector tensors.
- [ ] Test matrix tensors.
- [ ] Test 3D tensors.
- [ ] Test index calculation.
- [ ] Test out-of-range access.
- [ ] Test invalid dimensions.
- [ ] Test move semantics.
- [ ] Test zero-sized dimensions.
- [ ] Run AddressSanitizer.
- [ ] Run UndefinedBehaviorSanitizer.

---

# 11. Phase 01 Acceptance Criteria

Phase 01 is complete only when all the following conditions are met.

| Requirement                 | Expected result |
| --------------------------- | --------------- |
| 1D tensor                   | Works           |
| 2D tensor                   | Works           |
| 3D tensor                   | Works           |
| N-dimensional shape         | Works           |
| FP32 storage                | Works           |
| Contiguous row-major layout | Works           |
| Stride calculation          | Correct         |
| Multidimensional indexing   | Correct         |
| Flat indexing               | Correct         |
| Bounds checking             | Implemented     |
| Memory ownership            | RAII            |
| Move semantics              | Tested          |
| Invalid shape handling      | Tested          |
| Automated tests             | Passing         |
| Sanitizer checks            | Passing         |

## 11.1 Definition of Done

- All unit tests pass.
- No known memory-safety issues.
- Public interfaces are documented.
- CMake builds successfully.
- A working tensor demonstration exists.
- README contains setup and build instructions.
- Phase 01 design decisions are documented.

---

# 12. Phase 01 Technical Boundaries

We deliberately will NOT implement the following in Phase 01:

| Feature                                | Planned phase            |
| -------------------------------------- | ------------------------ |
| Matrix multiplication                  | Phase 02                 |
| Addition and multiplication operations | Phase 02                 |
| Broadcasting                           | Phase 02                 |
| Transpose                              | Phase 02                 |
| Tensor views                           | Future optimization      |
| SIMD                                   | Performance optimization |
| Multithreading                         | Performance optimization |
| GPU support                            | Future extension         |
| Custom memory allocator                | Future optimization      |
| FP16 / INT8                            | Quantization phase       |

This is important.

We should avoid turning the tensor implementation into a giant abstraction before we understand the requirements of the operations that will consume it.

---

# PART IV — ENGINEERING STANDARDS

## 13. Coding Standards

### C++

- C++20.
- RAII for resource ownership.
- Prefer const correctness.
- Avoid unnecessary raw owning pointers.
- Use explicit constructors where appropriate.
- Keep interfaces small.
- Use namespaces consistently.
- Avoid premature template abstractions.

### Testing

Every module should have independent unit tests.

We will prioritize:

1. Correctness.
2. Memory safety.
3. Numerical stability.
4. Performance.

Performance must not compromise correctness.

### Documentation

Every major module should have:

- Purpose.
- Public API.
- Internal design.
- Complexity considerations.
- Memory considerations.
- Tests.

## 14. Git Workflow

Use small, logically separated commits.

Suggested initial commit sequence:

```text
chore: initialize forge repository
build: configure cmake and c++20
test: integrate googletest
feat(tensor): implement shape abstraction
feat(tensor): implement contiguous strides
feat(memory): implement fp32 storage
feat(tensor): implement multidimensional indexing
test(tensor): add tensor correctness tests
docs: document phase 01 architecture
```

---

# PART V — LONG-TERM SUCCESS CRITERIA

## 15. What should Forge eventually demonstrate?

At completion, Forge should be able to:

1. Load a supported pretrained decoder-only Transformer.
2. Run inference entirely on the CPU.
3. Generate tokens autoregressively.
4. Reuse KV cache during decoding.
5. Handle multiple concurrent requests.
6. Stream generated text through an HTTP API.
7. Support memory-efficient model execution.
8. Provide measurable throughput and latency benchmarks.

We should eventually benchmark:

- Time to first token (TTFT).
- Time per output token (TPOT).
- Tokens per second.
- Peak RAM usage.
- Prompt processing throughput.
- Concurrent request throughput.

We will compare against established inference engines using the same model, hardware and workload.

The goal is not to claim superiority over mature engines. It is to understand the engineering trade-offs and progressively close the performance gap.

---

# IMMEDIATE NEXT MILESTONE

## Phase 01 — Task 01: Project Infrastructure

Our first coding milestone is intentionally small.

We will:

1. Initialize the Forge repository.
2. Configure CMake.
3. Set C++20.
4. Configure compiler warnings.
5. Integrate GoogleTest.
6. Create the first executable.
7. Run the first automated test.

**No Tensor implementation yet.**

We first establish a reliable engineering environment. Then we begin building the tensor abstraction one module at a time.

**End of Technical Specification v0.1**
