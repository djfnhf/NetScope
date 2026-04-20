from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Task:
    task_id: str
    paper_dimension: str
    script_relpath: str
    summary: str

    @property
    def script_path(self) -> Path:
        return ROOT / self.script_relpath


TASKS: List[Task] = [
    Task(
        task_id="cls.dataset.flow_packet_builder",
        paper_dimension="Traffic Classification / Dataset Construction",
        script_relpath="classify/build/main.py",
        summary="Build flow-level and packet-level datasets from raw PCAP using the modular pipeline.",
    ),
    Task(
        task_id="cls.dataset.tsv_builder",
        paper_dimension="Traffic Classification / Dataset Construction",
        script_relpath="classify/pcap2tsv.py",
        summary="Convert packet data to TSV token sequences for text-style classifiers.",
    ),
    Task(
        task_id="cls.dataset.png_builder",
        paper_dimension="Traffic Classification / Dataset Construction",
        script_relpath="classify/pcap2png.py",
        summary="Convert packet data to 40x40 PNG representations for image-style classifiers.",
    ),
    Task(
        task_id="cls.data_efficiency.subset_builder",
        paper_dimension="Traffic Classification / Data Efficiency",
        script_relpath="classify/gen.py",
        summary="Generate stratified low-label train subsets (1/5/10/20/50/75%).",
    ),
    Task(
        task_id="cls.robustness.perturbation_builder",
        paper_dimension="Traffic Classification / Robustness",
        script_relpath="classify/robust.py",
        summary="Generate robustness stress-test sets with perturbation levels and combinations.",
    ),
    Task(
        task_id="cls.interpretability.tsv_saliency",
        paper_dimension="Traffic Classification / Interpretability",
        script_relpath="classify/explain_tsv.py",
        summary="Run saliency explanation on TSV/token models and export heatmaps.",
    ),
    Task(
        task_id="cls.interpretability.png_saliency",
        paper_dimension="Traffic Classification / Interpretability",
        script_relpath="classify/explain_png.py",
        summary="Run saliency explanation on PNG/image models and export heatmaps.",
    ),
    Task(
        task_id="cls.cost.uer_inference_benchmark",
        paper_dimension="Traffic Classification / Computational Cost",
        script_relpath="classify/uer_classifier_inference_benchmark.py",
        summary="Benchmark inference latency/throughput for UER-family classifiers.",
    ),
    Task(
        task_id="cls.cost.yatc_inference_benchmark",
        paper_dimension="Traffic Classification / Computational Cost",
        script_relpath="classify/yatc_classifier_inference_benchmark.py",
        summary="Benchmark inference latency/throughput for YaTC classifier.",
    ),
    Task(
        task_id="gen.fidelity.distribution_metrics",
        paper_dimension="Traffic Generation / Distribution Fidelity",
        script_relpath="generate/pcap_similarity.py",
        summary="Compute JSD/TVD distribution fidelity metrics between real and generated traffic.",
    ),
    Task(
        task_id="gen.correctness.protocol_compliance",
        paper_dimension="Traffic Generation / Protocol Correctness",
        script_relpath="generate/eval_protocal.py",
        summary="Run hierarchical protocol compliance checks for generated PCAP.",
    ),
    Task(
        task_id="gen.utility.downstream_merge",
        paper_dimension="Traffic Generation / Downstream Utility",
        script_relpath="generate/pcap_merge_for_e2e.py",
        summary="Prepare merged PCAP sets for downstream utility scenarios.",
    ),
    Task(
        task_id="gen.diversity.metrics",
        paper_dimension="Traffic Generation / Generation Diversity",
        script_relpath="generate/generation_diversity.py",
        summary="Compute entropy, coverage, and novelty diversity metrics.",
    ),
    Task(
        task_id="prep.json_to_pcap",
        paper_dimension="Preprocessing",
        script_relpath="preprocess_data/json_to_pcap.py",
        summary="Convert generated JSON traffic records to PCAP format.",
    ),
    Task(
        task_id="prep.ip_filter",
        paper_dimension="Preprocessing",
        script_relpath="preprocess_data/pcap_ip_filter.py",
        summary="Filter PCAP by source/destination IP conditions.",
    ),
]

TASK_INDEX: Dict[str, Task] = {t.task_id: t for t in TASKS}


def list_groups() -> List[str]:
    return sorted({t.task_id.split(".")[0] for t in TASKS})


def list_dimensions() -> List[str]:
    return sorted({t.paper_dimension for t in TASKS})
