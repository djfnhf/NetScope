#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Inference benchmark for YaTC classifier models."""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
from typing import Any, List, Optional, Tuple

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _parse_gpu_ids_str(s: Optional[str]) -> Optional[List[int]]:
    if s is None or not str(s).strip():
        return None
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def _maybe_wrap_data_parallel(model: Any, bench_args: Any) -> Tuple[Any, bool, List[int]]:
    """Wrap model with DataParallel when multi-GPU inference is enabled."""
    import sys
    import torch
    import torch.nn as nn

    if not torch.cuda.is_available():
        return model, False, []
    if not getattr(bench_args, "data_parallel", False):
        return model, False, []
    n = torch.cuda.device_count()
    if n < 2:
        print(
            "[yatc_classifier_inference_benchmark] --data_parallel is set but visible GPU < 2, skip DataParallel",
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


def _sync_all_cuda_devices() -> None:
    import torch

    if not torch.cuda.is_available():
        return
    for i in range(torch.cuda.device_count()):
        torch.cuda.synchronize(i)


def _strip_module_prefix_state(state: Any) -> Any:
    if not isinstance(state, dict) or not state:
        return state
    if any(k.startswith("module.") for k in state.keys()):
        return {k.replace("module.", "", 1): v for k, v in state.items()}
    return state


def _compute_accuracy_yatc(
    model: Any,
    data_loader_val: Any,
    device: Any,
    use_amp: bool,
) -> Tuple[int, int, float]:
    """Compute top-1 accuracy with argmax(output)."""
    import torch

    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for batch in data_loader_val:
            images = batch[0].to(device, non_blocking=True)
            labels = batch[1].to(device, non_blocking=True)
            if use_amp:
                with torch.cuda.amp.autocast():
                    output = model(images)
            else:
                output = model(images)
            pred = output.argmax(dim=1)
            correct += int((pred == labels).sum().item())
            total += int(labels.size(0))
    acc = correct / total if total else 0.0
    return correct, total, acc


def _default_yatc_output_json_path(model_name: str) -> str:
    out_dir = os.path.join(_SCRIPT_DIR, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in model_name)
    return os.path.join(out_dir, f"yatc_{safe}_{ts}.json")


def _resolve_yatc_output_path(bench_args: Any, model_name: str) -> Optional[str]:
    if getattr(bench_args, "no_save_json", False):
        return None
    if getattr(bench_args, "output_json", None):
        p = os.path.abspath(bench_args.output_json)
        parent = os.path.dirname(p)
        if parent:
            os.makedirs(parent, exist_ok=True)
        return p
    return _default_yatc_output_json_path(model_name)


def _parse_bench_and_rest(argv: List[str]):
    bench = argparse.ArgumentParser(add_help=False)
    bench.add_argument("--yatc_root", type=str, required=True)
    bench.add_argument("--resume", type=str, required=True, help="Path to finetuned checkpoint (e.g. best_acc.pth).")
    bench.add_argument("--infer_batch_size", type=int, default=None, help="Override model batch_size for inference.")
    bench.add_argument("--warmup_epochs", type=int, default=1, help="Number of warmup full-pass rounds before timing.")
    bench.add_argument("--repeat_rounds", type=int, default=1, help="Full-pass rounds per timed run.")
    bench.add_argument("--timed_runs", type=int, default=3)
    bench.add_argument(
        "--target_samples",
        type=int,
        default=None,
        help="Optional exact number of timed samples; ignores repeat_rounds when set.",
    )
    bench.add_argument("--device", type=str, default="cuda")
    bench.add_argument("--num_workers", type=int, default=10, help="DataLoader workers.")
    bench.add_argument("--pin_mem", action="store_true", default=True)
    bench.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    bench.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Output JSON path; defaults to inference_benchmark/outputs/yatc_<model>_<ts>.json.",
    )
    bench.add_argument("--no_save_json", action="store_true", help="Print JSON to stdout only without writing file.")
    bench.add_argument("--seed", type=int, default=0)
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
    return bench.parse_known_args(argv)


def main() -> None:
    import numpy as np
    import torch
    import torch.backends.cudnn as cudnn
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    from torchvision import datasets

    bench_args, rest = _parse_bench_and_rest(sys.argv[1:])
    root = os.path.abspath(bench_args.yatc_root)
    if root not in sys.path:
        sys.path.insert(0, root)

    import importlib.util

    fin_path = os.path.join(root, "fin.py")
    spec = importlib.util.spec_from_file_location("yatc_fin", fin_path)
    assert spec and spec.loader
    fin = importlib.util.module_from_spec(spec)
    os.chdir(root)
    spec.loader.exec_module(fin)
    get_args_parser = fin.get_args_parser

    parser = get_args_parser()
    model_args = parser.parse_args(rest)

    if bench_args.infer_batch_size is not None:
        model_args.batch_size = bench_args.infer_batch_size

    torch.manual_seed(bench_args.seed)
    np.random.seed(bench_args.seed)
    cudnn.benchmark = True

    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        if getattr(bench_args, "data_parallel", False) and n_gpu >= 2:
            p = _parse_gpu_ids_str(bench_args.gpu_ids)
            gids = list(range(n_gpu)) if p is None else p
            for gid in gids:
                if gid < 0 or gid >= n_gpu:
                    raise SystemExit(f"Invalid --gpu_ids: {gid} is outside visible range 0..{n_gpu - 1}")
            device = torch.device(f"cuda:{gids[0]}")
        else:
            device = torch.device(bench_args.device)
    else:
        device = torch.device("cpu")

    mean = [0.5]
    std = [0.5]
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    test_root = model_args.test_dir if model_args.test_dir else os.path.join(model_args.data_path, "test")
    dataset_val = datasets.ImageFolder(test_root, transform=transform)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=model_args.batch_size,
        num_workers=bench_args.num_workers,
        pin_memory=bench_args.pin_mem,
        drop_last=False,
    )

    import models_YaTC

    model = models_YaTC.__dict__[model_args.model](
        num_classes=model_args.nb_classes,
        drop_path_rate=model_args.drop_path,
    )
    model.to(device)

    def _safe_torch_load(path: str):
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")

    ckpt = _safe_torch_load(bench_args.resume)
    if "model" in ckpt:
        sd = _strip_module_prefix_state(ckpt["model"])
        model.load_state_dict(sd, strict=True)
    else:
        sd = _strip_module_prefix_state(ckpt)
        model.load_state_dict(sd, strict=False)

    model, used_dp, dp_gpu_ids = _maybe_wrap_data_parallel(model, bench_args)

    model.eval()
    n = len(dataset_val)

    use_amp = device.type == "cuda"

    def one_full_pass() -> None:
        with torch.no_grad():
            for batch in data_loader_val:
                images = batch[0].to(device, non_blocking=True)
                if use_amp:
                    with torch.cuda.amp.autocast():
                        _ = model(images)
                else:
                    _ = model(images)

    def one_exact_k(k: int) -> None:
        """Run inference for exactly k samples, reusing dataset when needed."""
        seen = 0
        while seen < k:
            for batch in data_loader_val:
                images = batch[0]
                need = k - seen
                if images.size(0) > need:
                    images = images[:need]
                images = images.to(device, non_blocking=True)
                with torch.no_grad():
                    if use_amp:
                        with torch.cuda.amp.autocast():
                            _ = model(images)
                    else:
                        _ = model(images)
                seen += int(images.size(0))
                if seen >= k:
                    break

    for _ in range(max(0, bench_args.warmup_epochs)):
        if getattr(bench_args, "target_samples", None) is None:
            one_full_pass()
        else:
            one_exact_k(int(bench_args.target_samples))
    if device.type == "cuda":
        _sync_all_cuda_devices()

    def timed_one_full_pass() -> Tuple[float, float]:
        """Return data-prep time and forward time for one full pass."""
        data_sec = 0.0
        infer_sec = 0.0
        it = iter(data_loader_val)
        while True:
            if device.type == "cuda":
                _sync_all_cuda_devices()
            t0 = time.perf_counter()
            try:
                batch = next(it)
            except StopIteration:
                break
            images = batch[0].to(device, non_blocking=True)
            if device.type == "cuda":
                _sync_all_cuda_devices()
            t1 = time.perf_counter()
            with torch.no_grad():
                if use_amp:
                    with torch.cuda.amp.autocast():
                        _ = model(images)
                else:
                    _ = model(images)
            if device.type == "cuda":
                _sync_all_cuda_devices()
            t2 = time.perf_counter()
            data_sec += t1 - t0
            infer_sec += t2 - t1
        return data_sec, infer_sec

    def timed_one_exact_k(k: int) -> Tuple[float, float]:
        data_sec = 0.0
        infer_sec = 0.0
        seen = 0
        while seen < k:
            it = iter(data_loader_val)
            while seen < k:
                if device.type == "cuda":
                    _sync_all_cuda_devices()
                t0 = time.perf_counter()
                try:
                    batch = next(it)
                except StopIteration:
                    break
                images = batch[0]
                need = k - seen
                if images.size(0) > need:
                    images = images[:need]
                images = images.to(device, non_blocking=True)
                if device.type == "cuda":
                    _sync_all_cuda_devices()
                t1 = time.perf_counter()
                with torch.no_grad():
                    if use_amp:
                        with torch.cuda.amp.autocast():
                            _ = model(images)
                    else:
                        _ = model(images)
                if device.type == "cuda":
                    _sync_all_cuda_devices()
                t2 = time.perf_counter()
                data_sec += t1 - t0
                infer_sec += t2 - t1
                seen += int(images.size(0))
        return data_sec, infer_sec

    times: List[float] = []
    times_data_prep: List[float] = []
    times_infer: List[float] = []
    for _ in range(bench_args.timed_runs):
        if device.type == "cuda":
            _sync_all_cuda_devices()
        d_acc = 0.0
        i_acc = 0.0
        if getattr(bench_args, "target_samples", None) is None:
            for _ in range(bench_args.repeat_rounds):
                d1, i1 = timed_one_full_pass()
                d_acc += d1
                i_acc += i1
        else:
            d_acc, i_acc = timed_one_exact_k(int(bench_args.target_samples))
        if device.type == "cuda":
            _sync_all_cuda_devices()
        times.append(d_acc + i_acc)
        times_data_prep.append(d_acc)
        times_infer.append(i_acc)

    times_arr = np.array(times, dtype=np.float64)
    td_arr = np.array(times_data_prep, dtype=np.float64)
    ti_arr = np.array(times_infer, dtype=np.float64)
    mean_t = float(times_arr.mean())
    mean_d = float(td_arr.mean())
    mean_i = float(ti_arr.mean())
    target_samples = getattr(bench_args, "target_samples", None)
    total_imgs = int(target_samples) if target_samples is not None else (n * bench_args.repeat_rounds)
    ms_per_flow = mean_t / total_imgs * 1000.0
    ms_per_flow_data_prep = mean_d / total_imgs * 1000.0
    ms_per_flow_infer = mean_i / total_imgs * 1000.0
    flows_per_sec = total_imgs / mean_t if mean_t > 0 else float("nan")

    correct, total_lbl, accuracy = _compute_accuracy_yatc(
        model, data_loader_val, device, use_amp
    )

    out_path = _resolve_yatc_output_path(bench_args, model_args.model)

    out = {
        "yatc_root": root,
        "resume": bench_args.resume,
        "test_dir": test_root,
        "num_test_flows": n,
        "infer_batch_size": int(model_args.batch_size),
        "target_samples": (int(target_samples) if target_samples is not None else None),
        "num_timed_samples": int(total_imgs),
        "warmup_epochs": bench_args.warmup_epochs,
        "repeat_rounds": bench_args.repeat_rounds,
        "timed_runs": bench_args.timed_runs,
        "num_workers": bench_args.num_workers,
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
        "model": model_args.model,
        "nb_classes": model_args.nb_classes,
        "correct": correct,
        "num_evaluated": total_lbl,
        "accuracy": accuracy,
        "accuracy_note": "Top-1 on test_dir (ImageFolder); same head as engine.evaluate",
        "result_json_path": out_path,
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"[yatc_classifier_inference_benchmark] wrote: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
