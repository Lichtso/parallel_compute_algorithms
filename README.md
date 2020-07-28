# Parallel Compute Algorithms
This is the prototyping repository for the [proposed addition](https://llvm.discourse.group/t/gpu-compute-basic-algorithms/1281) of parallel compute algorithms to [MLIR](https://mlir.llvm.org/).

## References
- [Single-pass Parallel Prefix Scan with Decoupled Look-back](https://research.nvidia.com/sites/default/files/pubs/2016-03_Single-pass-Parallel-Prefix/nvr-2016-002.pdf)

## First Iteration
This iteration is a brainstorming to find a set of interesting parallel compute algorithms and primitives.
The [slides](https://docs.google.com/presentation/d/1mHitJw8vcJBAtKMc0swCA5TXZeL384WtZA9YqsUMEjg/edit?usp=sharing) explain these by example.

## Second Iteration
The purpose of this iteration is to figure out the interfaces and composition of the operations.
In order to quickly prove the composition to be viable and do so in an easy to understand way, this iteration was implemented in Python.

There are the following files:
- utils: Helpers for maths and nested arrays
- primitives: The operations which are not composed of other operations
- algorithms: Operations based on the primitives and other algorithms
- tests: Unit tests for all operations
