# Infera

![C++ CI](https://github.com/brandonviaje/InferLite/actions/workflows/cpp-tests.yml/badge.svg)

lightweight inference engine for executing trained neural network models efficiently

goal: learn the infrastructure behind graph optimization, kernel fusion and all under the hood features of an inference engine.

by the end of this project my inference engine should: 

- Load the model
- Construct a graph representation of the model
- Topologically sort nodes
- Run inference with user inputs


should prioritize:
- throughput
- latency
- concurrency

## System Architecture

<img width="1000" height="700" alt="Inference Server" src="https://github.com/user-attachments/assets/50655ea3-88e2-40e3-b19d-f4f8e3b80f8a" />

