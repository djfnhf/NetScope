# NetScope: A Benchmark for Pre-trained Network Traffic Models

NetScope is a research-oriented benchmark toolkit for network traffic analysis.
It provides a unified workflow for two major directions:

- Traffic Classification
- Traffic Generation

The project keeps each experimental script independent, while exposing a single launcher entry point for task discovery, execution, and validation.

## Why NetScope

- One command-line entry for all benchmark tasks
- Dimension-oriented task naming, aligned with paper sections
- Non-destructive organization: existing scripts remain where they are
- Easy to extend with new tasks by editing a single registry file

## Project Layout

- classify/: classification benchmarks and analysis tools
- generate/: generation quality and utility evaluation tools
- netscope/tasks.py: centralized task registry
- run_netscope.py: unified launcher

## Environment Setup

### 1) Python version

- Recommended: Python 3.11

### 2) Create and activate environment

Conda:

```bash
conda create -n netscope python=3.11 -y
conda activate netscope
```

### 3) Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4) Optional GPU install for PyTorch

If you need CUDA acceleration, install torch/torchvision following the official PyTorch instructions for your CUDA version.

### 5) External repositories needed by some tasks

Some classification scripts reference external repos by path (for example ET-BERT, TrafficFormer, YaTC). Make sure these repositories exist under your project root (or update paths in script config accordingly).

### 6) System dependency for dataset build pipeline

The flow/packet dataset build pipeline under classify/build uses SplitCap via mono:

- Install mono runtime
- Prepare SplitCap.exe
- Configure SPLITCAP_EXE path in classify/build/config.py

## Quick Start

### 1) Inspect available tasks

```bash
python run_netscope.py list
```

### 2) Search tasks by keyword

```bash
python run_netscope.py search robustness
```

### 3) Show full detail of one task

```bash
python run_netscope.py show cls.robustness.perturbation_builder
```

### 4) Run one task

Use -- to forward arguments to the target script.

```bash
python run_netscope.py run cls.dataset.flow_packet_builder -- --in-root datasets/cstnet_120 --workers 12
```

You can also run directly by task id:

```bash
python run_netscope.py cls.dataset.flow_packet_builder -- --in-root datasets/cstnet_120 --workers 12
```

### 5) Run a group of tasks

```bash
python run_netscope.py run-group cls.
```

```bash
python run_netscope.py run-group gen. -- --help
```

Continue on failures when running a group:

```bash
python run_netscope.py run-group cls. --continue-on-error
```

### 6) Validate task registry

```bash
python run_netscope.py doctor
```

Validate only one prefix:

```bash
python run_netscope.py doctor --prefix gen.
```

## Task ID Convention

Task IDs are semantic aliases:

- cls.*: classification evaluations
- gen.*: generation and preprocessing evaluations/utilities

Examples:

- cls.data_efficiency.subset_builder
- cls.cost.uer_inference_benchmark
- gen.correctness.protocol_compliance
- gen.preprocess.json_to_pcap

## Dimension Mapping

### Traffic Classification

- Dataset Construction
  - classify/build/main.py
  - classify/pcap2tsv.py
  - classify/pcap2png.py
- Data Efficiency
  - classify/gen.py
- Robustness
  - classify/robust.py
- Interpretability
  - classify/explain_tsv.py
  - classify/explain_png.py
- Computational Cost
  - classify/uer_classifier_inference_benchmark.py
  - classify/yatc_classifier_inference_benchmark.py

### Traffic Generation

- Distribution Fidelity
  - generate/pcap_similarity.py
- Protocol Correctness
  - generate/eval_protocal.py
- Downstream Utility
  - generate/pcap_merge_for_e2e.py
- Generation Diversity
  - generate/generation_diversity.py

### Preprocessing

- generate/preprocess_data/json_to_pcap.py
- generate/preprocess_data/pcap_ip_filter.py


## License Notice

- The original code in this project is licensed under the **MIT License**.
- This project includes the third-party tool **SplitCap**, developed by NETRESEC/Erik Hjelmvik, which is provided under **CC BY-ND 4.0**: https://www.netresec.com/?page=SplitCap
