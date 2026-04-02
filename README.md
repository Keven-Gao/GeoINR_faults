# GeoINR-faults

**GeoINR-faults** is an extension of the GeoINR framework, designed to explore and extend the capabilities of implicit neural representations for structural geology.

In addition to fault modeling, GeoINR-faults re-implements a substantial portion of the core functionality of the original GeoINR framework, providing a unified implementation for both standard GeoINR workflows and fault-related extensions.

## Relationship to GeoINR

GeoINR-faults is based on the methodology introduced in the original GeoINR project:

> Hillier, M., Wellmann, F., de Kemp, E., Schetselaar, E., Brodaric, B., & Bedard, K. (2023). GeoINR 1.0: an implicit neural representation network for three-dimensional geological modelling. Geoscientific Model Development Discussions, 2023, 1-40. https://doi.org/10.5194/gmd-16-6987-2023

This repository is developed as a research extension focusing on fault modeling.

**Note:**
- This is **not the official GeoINR repository**
- The implementation in this repository is developed independently

## Repository Structure

- `main_faults` branch: fault modeling implementation used in this fork
- `main` branch: baseline branch kept for compatibility
- `GeoINR_original` branch: original GeoINR codebase (forked)
- `NN_fault` branch: related fault modeling works in INRs (forked)

## Installation

We provide the latest release version of GeoINR-faults via PyPI package services. We highly recommend using PyPI,

`$ pip install geoinr-faults`

A clean environment is recommended. To accelerate computations for complex and large-scale models, installing the GPU-enabled version of PyTorch is advised. See the [Installation guide](docs/source/installation.rst) for details.

## Documentation

After installation, you can either check the notebook tutorials or go to the documentation for further information.

- [Notebook tutorials](examples/)
- [Documentation (online)](https://keven-gao.github.io/GeoINR_faults/)
- [Documentation build instructions](docs/README.md)
- Local built docs entry: `docs/build/html/index.html`

## License

This repository includes components from the original GeoINR project. All original GeoINR copyright and permission notices are retained in accordance with the MIT License. Additional code in this repository is subject to separate copyright notices.

## Citation

If you use this work, please cite:

- Hillier, M., Wellmann, F., de Kemp, E., Schetselaar, E., Brodaric, B., & Bedard, K. (2023). GeoINR 1.0: an implicit neural representation network for three-dimensional geological modelling. Geoscientific Model Development Discussions, 2023, 1-40. https://doi.org/10.5194/gmd-16-6987-2023
- Gao, K., & Wellmann, F. (2025). Fault representation in structural modelling with implicit neural representations. Computers & Geosciences, 199, 105911. https://doi.org/10.1016/j.cageo.2025.105911
- This repository (to be updated)
