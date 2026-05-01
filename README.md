# Tensorbit Core
Tensorbit-Core is a high-performance C++ library dedicated to Second-Order Sparsity Analysis. It is the first stage of the Tensorbit ecosystem pipeline, responsible for identifying and removing the redundant mathematical "bulk" of LLMs and Vision Transformers before they reach distillation or quantization.

## Role in the Tensorbit Ecosystem
In the Tensorbit ecosystem-based workflow, the core acts as a surgical center. While other tools focus on compression, Tensorbit-Core uses Hessian-based Sensitivity Analysis to physically alter the model architecture. It identifies load-bearing parameters and eliminates the noise, ensuring that the subsequent distillation and quantization stages are operating on the most efficient "intelligence skeleton" possible.

## Key Capabilities

## The P-D-Q Pipeline

## Tensorbit Model Zoo

## Tech Stack

## Usage: The `tb-prune` Utility

## Installation

## License
Licensed under the **Apache License 2.0**. Developed by **Tensorbit Labs**.
