📘 TrafficBench Dataset Builder – Full Pipeline README

This repository provides a high-throughput, fully modularized dataset construction pipeline for encrypted traffic analysis.
It supports:

Flow-level dataset construction (session split → filtering → fair sampling → stratified split)

Packet-level dataset construction derived from flow splits (packet split → filtering → target-based sampling → balanced test expansion)

Strict non-leakage guarantees: packets inherit their flow’s split; no cross-split mixing.

Deterministic & reproducible processing using unified random seeds.

The system is designed for research pipelines such as ET-BERT, TrafficFormer, NetGPT, YATC, and TrafficLLM.

🚀 Pipeline Overview
Raw PCAPs
   │
   │ (Flow Stage: SplitCap -s session)
   ▼
Flow Candidates
   │
   │ Filtering (min-bytes, min-pkts)
   │ Max-min Fair Sampling per Class
   │ Stratified 8:1:1 Flow Split
   ▼
OUT_FLOW/{train,valid,test}/<class>/*.pcap
   │
   │ (Packet Stage: SplitCap -s packets 1 per flow)
   ▼
Packet Candidates per Split
   │
   │ Filtering (TCP/UDP L3 length)
   │ Target-based sampling (strict split inheritance)
   │ Optional test expansion at class-cap
   ▼
OUT_PACKET/{train,valid,test}/<class>/*.pcap

📂 Directory Structure
trafficbench/
│
├── main.py                   # Entry point
├── config.py (optional)      # External config for cleaner overrides
│
├── flow_stage/               # Flow-level pipeline
│   ├── flow_session_split.py
│   ├── flow_filter.py
│   ├── flow_sampling.py
│   ├── flow_split.py
│   ├── flow_materialize.py
│   └── flow_pipeline.py
│
├── packet_stage/             # Packet-level pipeline
│   ├── pkt_from_flow.py
│   ├── pkt_filter.py
│   ├── pkt_target_calc.py
│   ├── pkt_sampling.py
│   ├── pkt_expand_test.py
│   ├── pkt_materialize.py
│   └── packet_pipeline.py
│
└── utils/
    ├── io_utils.py
    ├── pcap_utils.py
    ├── splitcap_utils.py
    ├── sampling.py
    └── hashing.py

Utility Modules
File	Purpose
io_utils.py	Directory creation, safe linking/copying, file size, manifest utilities
pcap_utils.py	Fast packet counting, L3 length extraction, first-packet info
splitcap_utils.py	Unified interface for SplitCap session/packet splitting
sampling.py	Max-min fair sampling for flows
hashing.py	Unique short-hash generation to avoid file conflicts
⚙️ Configuration

All configuration is stored in a CONFIG dictionary. Example:

CONFIG = {
    "IN_ROOT_BASE": "/data/datasets/cstnet_120",

    "SPLITCAP_EXE": "/path/to/SplitCap.exe",

    "WORKERS": 12,
    "FLOW_SPLIT_WORKERS": 12,
    "PACKET_SPLIT_WORKERS": 12,

    "FLOW_MIN_BYTES": 2048,
    "FLOW_MIN_PKTS": 3,
    "FLOW_CLASS_MIN": 10,
    "FLOW_CLASS_CAP": 500,
    "FLOW_SPLIT_RATIOS": (0.8, 0.1, 0.1),

    "TCP_MIN_L3": 144,
    "UDP_MIN_L3": 103,
    "PKT_CLASS_MIN": 100,
    "PKT_CLASS_CAP": 5000,
    "PKT_SPLIT_RATIOS": (0.8, 0.1, 0.1),

    "SEED": 42,
    "DO_FLOW_STAGE": True,
    "DO_PACKET_FROM_FLOW": True,
}

▶️ Running the Pipeline

From the root directory:

python3 main.py --in-root /data/datasets/cstnet_120


or for multiple datasets:

python3 main.py \
  --in-root "/data/datasets/cstnet_120,/data/datasets/CICIoT2023_7"


You may override any config key:

python3 main.py \
  --in-root /data/... \
  --workers 24 \
  --flow-class-cap 300 \
  --pkt-class-cap 8000

📑 Output Structure
1. Flow Stage Output
<IN_ROOT>_flow/
│
├── train/<class>/*.pcap
├── valid/<class>/*.pcap
├── test/<class>/*.pcap
│
├── lists/{train.txt, valid.txt, test.txt}
│
├── stats.json
└── manifest.jsonl

2. Packet Stage Output
<IN_ROOT>_packet/
│
├── train/<class>/*.pcap
├── valid/<class>/*.pcap
├── test/<class>/*.pcap
│
├── lists/{train.txt, valid.txt, test.txt}
│
├── stats.json
└── manifest.jsonl

🔐 Split Integrity & Non-Leakage Guarantees

The packet stage never mixes packets across flow splits.

Every flow has a fixed split assignment: train / valid / test

All packets belonging to that flow inherit the same split

No random cross-split sampling is allowed

Target calculation uses:

P_train / 0.8
P_valid / 0.1
P_test  / 0.1


Only classes that can satisfy the target under all three constraints are kept.

📈 Balanced Test Expansion

When all classes reach packet cap, the pipeline:

Computes overflow size for each class

Finds the minimum overflow count

Adds that many packet samples from each class to the test split

Ensures perfectly balanced expansion across classes

This improves evaluation robustness and fairness.

🧪 Performance Notes

Uses multiprocessing with configurable worker pools

Packet counting uses RawPcapReader for speed

Session splitting uses SplitCap + parallel workers

All sampling is seeded for full reproducibility

Hard-linking is used where possible to minimize I/O

🛠 Dependencies

Python ≥ 3.8

scapy

tqdm

SplitCap.exe (run via mono on Linux)

multiprocessing

Install base dependencies:

pip install scapy tqdm
sudo apt install mono-runtime
