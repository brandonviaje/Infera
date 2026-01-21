# Infera

![C++ CI](https://github.com/brandonviaje/InferLite/actions/workflows/cpp-tests.yml/badge.svg)

lightweight inference engine for executing trained neural network models efficiently

goal: learn the infrastructure behind inference server

# ML Inference Server

A **Machine Learning Inference Server** is a specialized server that hosts trained machine learning models and provides real-time predictions (inference) for new data. It acts as the bridge between a deployed model and applications that need to use it, such as web apps, mobile apps, or IoT devices.

## Key Components

1. **Model Repository / Storage**
   - Stores trained models and their versions.
   - Can be local or cloud-based.
   
2. **Model Loader / Manager**
   - Loads models into memory when requested.
   - Handles multiple models and version control.
   
3. **Inference Engine / Runtime**
   - Executes the model to generate predictions.
   - Optimized for CPU, GPU, or TPU performance.
   
4. **API / Request Handler**
   - Accepts input from clients via REST or gRPC.
   - Returns predictions or results to clients.
   
5. **Scheduler / Batching**
   - Groups requests to maximize hardware utilization.
   - Improves throughput and efficiency.
   
6. **Monitoring / Logging**
   - Tracks latency, throughput, and errors.
   - Ensures the system is observable and maintainable.

## Workflow

1. A client sends input data to the inference server via the API.
2. The server’s scheduler batches requests if needed, and sends them to the inference engine.
3. The inference engine runs the loaded model and generates predictions.
4. The server returns the predictions to the client.
5. Monitoring tracks the performance and logs any issues.

> Essentially, the inference server makes a trained ML model **usable in real-time applications** without having to reload or retrain it each time.

## System Architecture

<img width="1000" height="700" alt="Inference Server" src="https://github.com/user-attachments/assets/50655ea3-88e2-40e3-b19d-f4f8e3b80f8a" />

# Acknowledgements

- [ONNX Documentation](https://onnx.ai/onnx/index.html)
- [What is an Inference Server](https://www.doubleword.ai/resources/what-is-an-inference-server-10-characteristics-of-an-effective-inference-server-for-generative-ai-deployments)
- [Build Your Own Inference Engine](https://michalpitr.substack.com/p/build-your-own-inference-engine-from)
- [Understanding Inference Engine](https://www.gmicloud.ai/glossary/inference-engine)


