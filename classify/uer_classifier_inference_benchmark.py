#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Inference benchmark for UER classifiers (ET-BERT, TrafficFormer, NetGPT)."""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
from argparse import Namespace
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

# Script directory used for default output path.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Benchmark-level keys accepted in JSON config.
_BENCH_JSON_KEYS = frozenset(
    {
        "backend",
        "project_root",
        "checkpoint",
        "infer_batch_size",
        "warmup_batches",
        "repeat_rounds",
        "timed_runs",
        "device",
        "output_json",
        "no_save_json",
        "labels_num",
        "seed",
        "config_json",
        "data_parallel",
        "gpu_ids",
    }
)


def _parse_gpu_ids_str(s: Optional[str]) -> Optional[List[int]]:
    if s is None or not str(s).strip():
        return None
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def _sync_all_cuda_devices() -> None:
    import torch

    if not torch.cuda.is_available():
        return
    for i in range(torch.cuda.device_count()):
        torch.cuda.synchronize(i)


def _maybe_wrap_data_parallel(model: Any, bench_args: Namespace) -> Tuple[Any, bool, List[int]]:
    """Wrap model with DataParallel for multi-GPU inference."""
    import torch
    import torch.nn as nn

    if not torch.cuda.is_available():
        return model, False, []
    if not getattr(bench_args, "data_parallel", False):
        return model, False, []
    n = torch.cuda.device_count()
    if n < 2:
        print(
            "[uer_classifier_inference_benchmark] --data_parallel is set but visible GPU < 2, skip DataParallel",
            file=sys.stderr,
        )
        return model, False, []
    parsed = _parse_gpu_ids_str(getattr(bench_args, "gpu_ids", None))
    if parsed is None:
        gpu_ids = list(range(n))
    else:
        gpu_ids = parsed
        for gid in gpu_ids:
            if gid < 0 or gid >= n:
                raise SystemExit(f"Invalid --gpu_ids: {gid} is outside visible range 0..{n - 1}")
    wrapped = nn.DataParallel(model, device_ids=gpu_ids)
    return wrapped, True, gpu_ids


def _strip_module_prefix(state: Dict[str, Any]) -> Dict[str, Any]:
    if not state:
        return state
    if any(k.startswith("module.") for k in state.keys()):
        return {k.replace("module.", "", 1): v for k, v in state.items()}
    return state


def _load_uer_module(backend: str, project_root: str):
    """Load classifier utilities from the selected backend repository."""
    root = os.path.abspath(project_root)
    if root not in sys.path:
        sys.path.insert(0, root)
    old = os.getcwd()
    os.chdir(root)
    try:
        if backend == "et_bert":
            from fine_tuning.run_classifier import (  # type: ignore
                Classifier,
                batch_loader,
                count_labels_num,
                read_dataset,
            )
        elif backend == "trafficformer":
            import importlib.util

            path = os.path.join(root, "fine-tuning", "run_classifier.py")
            spec = importlib.util.spec_from_file_location("tf_run_classifier", path)
            assert spec and spec.loader
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            Classifier = mod.Classifier
            batch_loader = mod.batch_loader
            count_labels_num = mod.count_labels_num
            read_dataset = mod.read_dataset
        elif backend == "netgpt":
            from finetune.run_understanding import (  # type: ignore
                Classifier,
                batch_loader,
                count_labels_num,
                read_dataset,
            )
        else:
            raise ValueError(f"unknown backend: {backend}")
    finally:
        os.chdir(old)
    return Classifier, batch_loader, count_labels_num, read_dataset


def _relax_finetune_path_requirements(parser: argparse.ArgumentParser) -> None:
    """Relax train/dev requirements for inference-only runs."""
    for a in parser._actions:
        dest = getattr(a, "dest", None)
        if dest in ("train_path", "dev_path"):
            a.required = False
            if getattr(a, "default", None) is argparse.SUPPRESS:
                continue
            a.default = None


def _build_parser_et_tf(backend: str) -> argparse.ArgumentParser:
    from uer.opts import finetune_opts  # type: ignore

    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    finetune_opts(p)
    _relax_finetune_path_requirements(p)
    p.add_argument(
        "--pooling",
        choices=["mean", "max", "first", "last"],
        default="first",
        help="Pooling type.",
    )
    p.add_argument(
        "--tokenizer",
        choices=["bert", "char", "space"],
        default="bert",
        help="Tokenizer name used in training.",
    )
    p.add_argument("--soft_targets", action="store_true")
    p.add_argument("--soft_alpha", type=float, default=0.5)
    if backend == "trafficformer":
        p.add_argument("--earlystop", type=int, default=5)
        p.add_argument("--is_moe", action="store_true")
        p.add_argument("--vocab_size", type=int, default=None)
        p.add_argument("--moebert_expert_dim", type=int, default=3072)
        p.add_argument("--moebert_expert_num", type=int, default=None)
        p.add_argument(
            "--moebert_route_method",
            choices=["gate-token", "gate-sentence", "hash-random", "hash-balance", "proto"],
            default="hash-random",
        )
        p.add_argument("--moebert_route_hash_list", default=None, type=str)
        p.add_argument("--moebert_load_balance", type=float, default=0.0)
    return p


def _build_parser_netgpt() -> argparse.ArgumentParser:
    from uer.opts import adv_opts, finetune_opts, tokenizer_opts  # type: ignore

    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    finetune_opts(p)
    _relax_finetune_path_requirements(p)
    tokenizer_opts(p)
    adv_opts(p)
    p.add_argument("--soft_targets", action="store_true")
    p.add_argument("--soft_alpha", type=float, default=0.5)
    p.add_argument("--labels_num", type=int, default=None)
    return p


def _load_hyperparam_et_tf(args: Namespace) -> Namespace:
    from uer.utils.config import load_hyperparam  # type: ignore

    return load_hyperparam(args)


def _netgpt_explicit_keys_from_rest(rest: Sequence[str]) -> Set[str]:
    """Collect keys explicitly provided in CLI args."""
    keys: Set[str] = set()
    for tok in rest:
        if not tok.startswith("--"):
            continue
        body = tok[2:]
        if body == "local_rank" or body.startswith("local_rank="):
            continue
        if body.startswith("no-"):
            keys.add(body.replace("-", "_"))
            continue
        if "=" in body:
            name = body.split("=", 1)[0]
        else:
            name = body
        keys.add(name.replace("-", "_"))
    return keys


def _load_hyperparam_netgpt_like(model_args: Namespace, rest: Sequence[str]) -> Namespace:
    """Match NetGPT config merge order: defaults -> config -> explicit CLI."""
    with open(model_args.config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    d = vars(model_args).copy()
    d.update(cfg)
    explicit = _netgpt_explicit_keys_from_rest(rest)
    for k in explicit:
        if hasattr(model_args, k):
            d[k] = getattr(model_args, k)
    return Namespace(**d)


def _normalize_netgpt_nargs_plus_fields(args: Namespace) -> None:
    """Normalize nargs='+' fields to lists for NetGPT compatibility."""
    for name in ("embedding", "tgt_embedding", "target"):
        v = getattr(args, name, None)
        if isinstance(v, str):
            setattr(args, name, [v])
        elif isinstance(v, tuple):
            setattr(args, name, list(v))


def _json_to_argv(cfg: Dict[str, Any]) -> List[str]:
    """Convert JSON key/value pairs to argparse-style argv."""
    out: List[str] = []
    for k, v in cfg.items():
        if v is None:
            continue
        key = k if k.startswith("--") else f"--{k}"
        if isinstance(v, bool):
            if v:
                out.append(key)
        else:
            out.extend([key, str(v)])
    return out


def _split_config_for_bench_and_model(
    cfg: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    bench: Dict[str, Any] = {}
    model: Dict[str, Any] = {}
    for k, v in cfg.items():
        if k in _BENCH_JSON_KEYS:
            bench[k] = v
        else:
            model[k] = v
    return bench, model


def _run_inference_loop(
    model: Any,
    args: Namespace,
    dataset: List,
    batch_loader_fn: Callable,
    device: Any,
    infer_batch_size: int,
    target_samples: Optional[int],
) -> Tuple[float, float, float]:
    """Run one timed pass and return total/data-prep/inference seconds."""
    import torch
    import torch.nn as nn

    model.eval()

    def one_full_pass(src_c: Any, tgt_c: Any, seg_c: Any) -> None:
        for src_b, _tgt_b, seg_b, _ in batch_loader_fn(infer_batch_size, src_c, tgt_c, seg_c):
            src_b = src_b.to(device)
            seg_b = seg_b.to(device)
            with torch.no_grad():
                _, logits = model(src_b, None, seg_b)
                if logits is not None:
                    _ = torch.argmax(nn.Softmax(dim=1)(logits), dim=1)

    def one_exact_k(src_c: Any, tgt_c: Any, seg_c: Any, k: int) -> None:
        seen = 0
        while seen < k:
            for src_b, _tgt_b, seg_b, _ in batch_loader_fn(infer_batch_size, src_c, tgt_c, seg_c):
                need = k - seen
                if src_b.size(0) > need:
                    src_b = src_b[:need]
                    seg_b = seg_b[:need]
                src_b = src_b.to(device)
                seg_b = seg_b.to(device)
                with torch.no_grad():
                    _, logits = model(src_b, None, seg_b)
                    if logits is not None:
                        _ = torch.argmax(nn.Softmax(dim=1)(logits), dim=1)
                seen += int(src_b.size(0))
                if seen >= k:
                    break

    src_w = torch.LongTensor([sample[0] for sample in dataset])
    tgt_w = torch.LongTensor([sample[1] for sample in dataset])
    seg_w = torch.LongTensor([sample[2] for sample in dataset])
    for _ in range(max(0, args.warmup_batches)):
        if target_samples is None:
            one_full_pass(src_w, tgt_w, seg_w)
        else:
            one_exact_k(src_w, tgt_w, seg_w, int(target_samples))
    if device.type == "cuda":
        _sync_all_cuda_devices()

    def _forward_one(src_b: Any, seg_b: Any) -> None:
        with torch.no_grad():
            _, logits = model(src_b, None, seg_b)
            if logits is not None:
                _ = torch.argmax(nn.Softmax(dim=1)(logits), dim=1)

    def timed_full_passes() -> Tuple[float, float]:
        data_sec = 0.0
        infer_sec = 0.0
        it = batch_loader_fn(infer_batch_size, src_cpu, tgt_cpu, seg_cpu)
        while True:
            if device.type == "cuda":
                _sync_all_cuda_devices()
            t0 = time.perf_counter()
            try:
                src_b, _tgt_b, seg_b, _ = next(it)
            except StopIteration:
                break
            src_b = src_b.to(device)
            seg_b = seg_b.to(device)
            if device.type == "cuda":
                _sync_all_cuda_devices()
            t1 = time.perf_counter()
            _forward_one(src_b, seg_b)
            if device.type == "cuda":
                _sync_all_cuda_devices()
            t2 = time.perf_counter()
            data_sec += t1 - t0
            infer_sec += t2 - t1
        return data_sec, infer_sec

    def timed_exact_k(k: int) -> Tuple[float, float]:
        data_sec = 0.0
        infer_sec = 0.0
        seen = 0
        while seen < k:
            it = batch_loader_fn(infer_batch_size, src_cpu, tgt_cpu, seg_cpu)
            while seen < k:
                if device.type == "cuda":
                    _sync_all_cuda_devices()
                t0 = time.perf_counter()
                try:
                    src_b, _tgt_b, seg_b, _ = next(it)
                except StopIteration:
                    break
                need = k - seen
                if src_b.size(0) > need:
                    src_b = src_b[:need]
                    seg_b = seg_b[:need]
                src_b = src_b.to(device)
                seg_b = seg_b.to(device)
                if device.type == "cuda":
                    _sync_all_cuda_devices()
                t1 = time.perf_counter()
                _forward_one(src_b, seg_b)
                if device.type == "cuda":
                    _sync_all_cuda_devices()
                t2 = time.perf_counter()
                data_sec += t1 - t0
                infer_sec += t2 - t1
                seen += int(src_b.size(0))
        return data_sec, infer_sec

    data_acc = 0.0
    infer_acc = 0.0

    if device.type == "cuda":
        _sync_all_cuda_devices()
    t_mat0 = time.perf_counter()
    src_cpu = torch.LongTensor([sample[0] for sample in dataset])
    tgt_cpu = torch.LongTensor([sample[1] for sample in dataset])
    seg_cpu = torch.LongTensor([sample[2] for sample in dataset])
    if device.type == "cuda":
        _sync_all_cuda_devices()
    data_acc += time.perf_counter() - t_mat0

    if target_samples is None:
        for _ in range(int(args.repeat_rounds)):
            d1, i1 = timed_full_passes()
            data_acc += d1
            infer_acc += i1
    else:
        d2, i2 = timed_exact_k(int(target_samples))
        data_acc += d2
        infer_acc += i2

    if device.type == "cuda":
        _sync_all_cuda_devices()
    total = data_acc + infer_acc
    return total, data_acc, infer_acc


def _compute_accuracy_uer(
    model: Any,
    dataset: List,
    batch_loader_fn: Callable,
    device: Any,
    infer_batch_size: int,
) -> Tuple[int, int, float]:
    """Compute top-1 accuracy with argmax(softmax(logits))."""
    import torch
    import torch.nn as nn

    src = torch.LongTensor([sample[0] for sample in dataset])
    tgt = torch.LongTensor([sample[1] for sample in dataset])
    seg = torch.LongTensor([sample[2] for sample in dataset])
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for src_b, tgt_b, seg_b, _ in batch_loader_fn(infer_batch_size, src, tgt, seg):
            src_b = src_b.to(device)
            tgt_b = tgt_b.to(device)
            seg_b = seg_b.to(device)
            _, logits = model(src_b, None, seg_b)
            pred = torch.argmax(nn.Softmax(dim=1)(logits), dim=1)
            correct += int((pred == tgt_b).sum().item())
            total += int(tgt_b.size(0))
    acc = correct / total if total else 0.0
    return correct, total, acc


def _parse_bench_parser() -> argparse.ArgumentParser:
    bench = argparse.ArgumentParser(
        add_help=True,
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    bench.add_argument(
        "--config_json",
        type=str,
        default=None,
        help="Optional JSON config; CLI args override duplicated keys.",
    )
    bench.add_argument(
        "--backend",
        choices=["et_bert", "trafficformer", "netgpt"],
        default=None,
        help="Backend repository used to import Classifier.",
    )
    bench.add_argument("--project_root", type=str, default=None, help="Root directory of backend repository.")
    bench.add_argument("--checkpoint", type=str, default=None, help="Path to finetuned classifier checkpoint.")
    bench.add_argument(
        "--infer_batch_size",
        type=int,
        default=None,
        help="Inference batch size; overrides model config batch_size.",
    )
    bench.add_argument(
        "--target_samples",
        type=int,
        default=None,
        help="Optional exact number of timed samples; ignores repeat_rounds when set.",
    )
    bench.add_argument(
        "--warmup_batches",
        type=int,
        default=None,
        help="Warmup full-pass rounds before timing.",
    )
    bench.add_argument(
        "--repeat_rounds",
        type=int,
        default=None,
        help="Full-pass rounds per timed run when target_samples is unset.",
    )
    bench.add_argument("--timed_runs", type=int, default=None, help="Number of independent timed runs.")
    bench.add_argument("--device", type=str, default=None)
    bench.add_argument(
        "--data_parallel",
        action="store_true",
        help="Enable nn.DataParallel for multi-GPU inference.",
    )
    bench.add_argument(
        "--gpu_ids",
        type=str,
        default=None,
        help="Comma-separated GPU ids for DataParallel, e.g. 0,1.",
    )
    bench.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Output JSON path; defaults to inference_benchmark/outputs/uer_<backend>_<ts>.json.",
    )
    bench.add_argument(
        "--no_save_json",
        action="store_true",
        help="Print JSON to stdout only without writing file.",
    )
    bench.add_argument(
        "--labels_num",
        type=int,
        default=None,
        help="Label count; inferred from test TSV when omitted.",
    )
    bench.add_argument("--seed", type=int, default=None)
    return bench


def _merge_bench_from_json(
    bench_ns: Namespace, jbench: Dict[str, Any], require_checkpoint: bool = True
) -> Namespace:
    """Merge benchmark args with JSON config while keeping CLI precedence."""
    d: Dict[str, Any] = {
        k: v for k, v in vars(bench_ns).items() if k != "config_json"
    }
    for k, v in jbench.items():
        if k == "config_json":
            continue
        d[k] = v
    for k, v in vars(bench_ns).items():
        if k == "config_json":
            continue
        if v is not None:
            d[k] = v
    # Fill default values when neither JSON nor CLI provides them.
    if d.get("warmup_batches") is None:
        d["warmup_batches"] = 2
    if d.get("repeat_rounds") is None:
        d["repeat_rounds"] = 1
    if d.get("timed_runs") is None:
        d["timed_runs"] = 3
    if d.get("device") is None:
        d["device"] = "cuda:0"
    if d.get("seed") is None:
        d["seed"] = 42
    if d.get("no_save_json") is None:
        d["no_save_json"] = False
    if d.get("data_parallel") is None:
        d["data_parallel"] = False
    d.pop("timing_mode", None)
    out = Namespace(**d)
    if out.backend is None:
        raise SystemExit("Please specify --backend or provide backend in config_json.")
    if out.project_root is None:
        raise SystemExit("Please specify --project_root or provide project_root in config_json.")
    if require_checkpoint and not out.checkpoint:
        raise SystemExit("Please specify --checkpoint or provide checkpoint in config_json.")
    return out


def _default_uer_output_json_path(backend: str) -> str:
    out_dir = os.path.join(_SCRIPT_DIR, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(out_dir, f"uer_{backend}_{ts}.json")


def _resolve_uer_output_path(bench_args: Namespace) -> Optional[str]:
    if getattr(bench_args, "no_save_json", False):
        return None
    if getattr(bench_args, "output_json", None):
        p = os.path.abspath(bench_args.output_json)
        parent = os.path.dirname(p)
        if parent:
            os.makedirs(parent, exist_ok=True)
        return p
    return _default_uer_output_json_path(str(bench_args.backend))


def _safe_torch_load(path: str, map_location: str = "cpu"):
    import torch

    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def parse_uer_bench_argv(argv: Sequence[str]) -> Tuple[Namespace, List[str]]:
    """Parse benchmark/config args and return remaining model argv."""
    argv = list(argv)

    # JSON-only shorthand: python script.py config.json
    if len(argv) == 1 and argv[0].endswith(".json") and os.path.isfile(argv[0]):
        argv = ["--config_json", argv[0]]

    bench_parser = _parse_bench_parser()
    bench_args, rest = bench_parser.parse_known_args(argv)

    json_cfg: Dict[str, Any] = {}
    if bench_args.config_json:
        with open(bench_args.config_json, "r", encoding="utf-8") as f:
            json_cfg = json.load(f)
        jbench, jmodel = _split_config_for_bench_and_model(json_cfg)
        bench_args = _merge_bench_from_json(bench_args, jbench)
        rest = _json_to_argv(jmodel) + rest
    else:
        bench_args = _merge_bench_from_json(bench_args, {})

    return bench_args, rest


def prepare_uer_classifier(
    bench_args: Namespace, rest: Sequence[str]
) -> Tuple[Any, Namespace, Callable, Callable]:
    """Build classifier and load checkpoint without device move/DP wrapping."""
    root = os.path.abspath(bench_args.project_root)
    sys.path.insert(0, root)
    os.chdir(root)

    if bench_args.backend in ("et_bert", "trafficformer"):
        model_parser = _build_parser_et_tf(bench_args.backend)
    else:
        model_parser = _build_parser_netgpt()

    model_args = model_parser.parse_args(list(rest))

    if bench_args.backend in ("et_bert", "trafficformer"):
        args = _load_hyperparam_et_tf(model_args)
    else:
        args = _load_hyperparam_netgpt_like(model_args, rest)
        _normalize_netgpt_nargs_plus_fields(args)

    if not getattr(args, "train_path", None):
        args.train_path = args.test_path
    if not getattr(args, "dev_path", None):
        args.dev_path = args.test_path

    if bench_args.infer_batch_size is not None:
        args.batch_size = bench_args.infer_batch_size

    args.warmup_batches = bench_args.warmup_batches
    args.repeat_rounds = bench_args.repeat_rounds
    args.seed = bench_args.seed

    from uer.utils.seed import set_seed  # type: ignore

    set_seed(args.seed)

    Classifier, batch_loader_fn, count_labels_num, read_dataset = _load_uer_module(
        bench_args.backend, root
    )

    ln_bench = getattr(bench_args, "labels_num", None)
    ln_model = getattr(args, "labels_num", None)
    if ln_bench is not None:
        args.labels_num = int(ln_bench)
    elif ln_model is not None:
        args.labels_num = ln_model
    else:
        args.labels_num = count_labels_num(args.test_path)

    from uer.utils import str2tokenizer  # type: ignore

    args.tokenizer = str2tokenizer[args.tokenizer](args)

    model = Classifier(args)
    ckpt = _safe_torch_load(bench_args.checkpoint, map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt:
        ckpt = ckpt["model"]
    ckpt = _strip_module_prefix(ckpt)
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    if missing or unexpected:
        print("load_state_dict strict=False; missing:", len(missing), "unexpected:", len(unexpected))

    return model, args, batch_loader_fn, read_dataset


def main(argv: Optional[Sequence[str]] = None) -> None:
    import numpy as np
    import torch

    argv = list(sys.argv[1:] if argv is None else argv)
    bench_args, rest = parse_uer_bench_argv(argv)

    model, args, batch_loader_fn, read_dataset = prepare_uer_classifier(bench_args, rest)
    root = os.path.abspath(bench_args.project_root)

    device = torch.device(bench_args.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model, used_dp, dp_gpu_ids = _maybe_wrap_data_parallel(model, bench_args)
    args.model = model

    test_ds = read_dataset(args, args.test_path)
    n = len(test_ds)
    if n == 0:
        raise RuntimeError("test set is empty")

    times: List[float] = []
    times_data_prep: List[float] = []
    times_infer: List[float] = []
    for _ in range(bench_args.timed_runs):
        t_sec, d_sec, i_sec = _run_inference_loop(
            model,
            args,
            test_ds,
            batch_loader_fn,
            device,
            args.batch_size,
            getattr(bench_args, "target_samples", None),
        )
        times.append(t_sec)
        times_data_prep.append(d_sec)
        times_infer.append(i_sec)

    times_arr = np.array(times, dtype=np.float64)
    td_arr = np.array(times_data_prep, dtype=np.float64)
    ti_arr = np.array(times_infer, dtype=np.float64)
    target_samples = getattr(bench_args, "target_samples", None)
    total_flows = int(target_samples) if target_samples is not None else (n * args.repeat_rounds)
    mean_t = float(times_arr.mean())
    mean_d = float(td_arr.mean())
    mean_i = float(ti_arr.mean())
    ms_per_flow = mean_t / total_flows * 1000.0
    ms_per_flow_data_prep = mean_d / total_flows * 1000.0
    ms_per_flow_infer = mean_i / total_flows * 1000.0
    flows_per_sec = total_flows / mean_t if mean_t > 0 else float("nan")

    correct, total_lbl, accuracy = _compute_accuracy_uer(
        model, test_ds, batch_loader_fn, device, args.batch_size
    )

    out_path = _resolve_uer_output_path(bench_args)

    out = {
        "backend": bench_args.backend,
        "project_root": root,
        "checkpoint": bench_args.checkpoint,
        "test_path": args.test_path,
        "train_path": args.train_path,
        "train_dev_path_note": "train/dev fallback to test_path when omitted; inference reads test_path only.",
        "labels_num": int(args.labels_num),
        "num_test_flows": n,
        "infer_batch_size": int(args.batch_size),
        "target_samples": (int(target_samples) if target_samples is not None else None),
        "num_timed_samples": int(total_flows),
        "warmup_batches": args.warmup_batches,
        "repeat_rounds": args.repeat_rounds,
        "timed_runs": bench_args.timed_runs,
        "times_sec": times,
        "times_data_prep_sec": times_data_prep,
        "times_infer_sec": times_infer,
        "mean_time_sec": mean_t,
        "mean_data_prep_time_sec": mean_d,
        "mean_infer_time_sec": mean_i,
        "ms_per_flow": ms_per_flow,
        "ms_per_flow_data_prep": ms_per_flow_data_prep,
        "ms_per_flow_infer": ms_per_flow_infer,
        "flows_per_sec": flows_per_sec,
        "timing_split_note": "mean_time_sec equals mean_data_prep_time_sec + mean_infer_time_sec for each timed run.",
        "device": str(device),
        "data_parallel": used_dp,
        "data_parallel_gpu_ids": dp_gpu_ids,
        "correct": correct,
        "num_evaluated": total_lbl,
        "accuracy": accuracy,
        "accuracy_note": "Top-1 on test_path; pred=argmax(softmax(logits)), same as run_classifier.evaluate",
        "result_json_path": out_path,
    }

    print(json.dumps(out, indent=2, ensure_ascii=False))
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"[uer_classifier_inference_benchmark] wrote: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
