import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from importlib import import_module
from typing import Any, Dict, List, Optional, Tuple

import yaml
import pandas as pd


@dataclass
class TrialResult:
    trial_id: int
    params: Dict[str, Any]
    metric: float
    config_path: str
    workdir: str
    success: bool
    notes: str = ""


DEFAULT_SEARCH_SPACE = {
    # Decoder / Encoder hidden dimension + feedforward MLP width
    "MODEL.ENCODER.HIDDEN_DIM": [256, 384, 512],
    "MODEL.DECODER.HIDDEN_DIM": ["=MODEL.ENCODER.HIDDEN_DIM"],  # tie to encoder by default
    "MODEL.DECODER.DIM_FEEDFORWARD": [512, 1024, 2048],
    # Attention heads must divide hidden dim
    "MODEL.DECODER.NHEADS": ["divisors(MODEL.DECODER.HIDDEN_DIM, [4, 8])"],  # auto-filtered
    # Depth: encoder/decoder transformer layers
    "MODEL.DECODER.ENC_LAYERS": [2, 3, 4],
    "MODEL.DECODER.DEC_LAYERS": [2, 3, 4],
    # Dropout
    "MODEL.DECODER.DROPOUT": [0.0, 0.1],
    # Feature pyramid levels (optional; keep fixed unless you know what you're doing)
    # "MODEL.ENCODER.NUM_FEATURE_LEVELS": [3, 4],
    # Auxiliary loss is off per your base config; keep it constant for fairness.
}

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def dump_yaml(obj: Dict[str, Any], path: str) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)

def set_by_dotted_key(cfg: Dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    cur = cfg
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value

def get_by_dotted_key(cfg: Dict[str, Any], key: str) -> Any:
    cur = cfg
    for p in key.split("."):
        cur = cur[p]
    return cur

def divisors(n: int, whitelist: Optional[List[int]] = None) -> List[int]:
    ds = [d for d in range(1, n + 1) if n % d == 0]
    if whitelist:
        ds = [d for d in ds if d in whitelist]
    return ds

def materialize_space(space: Dict[str, Any], base_cfg: Dict[str, Any]) -> Dict[str, List[Any]]:
    """
    Resolve dynamic entries like "=MODEL.ENCODER.HIDDEN_DIM" or "divisors(MODEL.DECODER.HIDDEN_DIM, [4,8])".
    """
    out = {}
    for k, v in space.items():
        if isinstance(v, list):
            out[k] = []
            for item in v:
                if isinstance(item, str) and item.startswith("="):
                    ref = item[1:]
                    out[k].append(get_by_dotted_key(base_cfg, ref))
                elif isinstance(item, str) and item.startswith("divisors("):
                    # parse "divisors(PATH, [list])"
                    m = re.match(r"divisors\(([^,]+)(?:,\s*(\[.*\]))?\)", item)
                    if not m:
                        raise ValueError(f"Bad divisors spec: {item}")
                    ref = m.group(1).strip()
                    wl = json.loads(m.group(2)) if m.group(2) else None
                    n = int(get_by_dotted_key(base_cfg, ref))
                    out[k] = divisors(n, wl)
                else:
                    out[k].append(item)
        else:
            out[k] = v
    return out

def sample_params(space: Dict[str, List[Any]]) -> Dict[str, Any]:
    return {k: random.choice(v) for k, v in space.items()}

def apply_overrides(base_cfg: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    cfg = json.loads(json.dumps(base_cfg))  # deep copy
    for k, v in params.items():
        set_by_dotted_key(cfg, k, v)
    return cfg

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def run_subprocess_runner(cmd_template: str, config_path: str, workdir: str, metric_regex: str) -> Tuple[bool, Optional[float], str]:
    cmd = cmd_template.format(config_path=config_path, workdir=workdir)
    try:
        proc = subprocess.run(cmd, shell=True, cwd=workdir, capture_output=True, text=True, check=False)
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        with open(os.path.join(workdir, "runner_stdout.txt"), "w") as f:
            f.write(stdout)
        with open(os.path.join(workdir, "runner_stderr.txt"), "w") as f:
            f.write(stderr)
        if proc.returncode != 0:
            return False, None, f"Runner exit code {proc.returncode}"
        m = re.search(metric_regex, stdout)
        if not m:
            return False, None, f"Metric regex not found in stdout."
        val = float(m.group(1))
        return True, val, ""
    except Exception as e:
        return False, None, f"Exception: {e}"

def run_python_runner(py_spec: str, config_dict: Dict[str, Any], workdir: str, metric_key: Optional[str]) -> Tuple[bool, Optional[float], str]:
    """
    py_spec: "mypkg.module:function"
    The function signature must be: func(config_dict: dict, workdir: str) -> Dict[str, Any]
    It should return a dict that includes either:
      - the metric under metric_key, or
      - a single float under key "metric"
    """
    try:
        mod_name, func_name = py_spec.split(":")
        mod = import_module(mod_name)
        fn = getattr(mod, func_name)
        result = fn(config_dict, workdir)  # user-implemented
        if metric_key and metric_key in result:
            return True, float(result[metric_key]), ""
        if "metric" in result:
            return True, float(result["metric"]), ""
        return False, None, "Metric not found in result dict."
    except Exception as e:
        return False, None, f"Exception: {e}"

def extract_architecture_patch(base_cfg: Dict[str, Any], best_cfg: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
    patch = {}
    for k in keys:
        patch[k] = get_by_dotted_key(best_cfg, k)
    return patch



########################################################
########################################################
########################################################

def main(args):
    
    print('~'*100)
    print('STARTING HYPERPARAMETER OPTIMIZATION')
    print('~'*100)
    
    hpo_type = "pretraining" if args.hpo_type == "pretaining" else args.hpo_type
    is_arch = (hpo_type == "architecture")
    is_pre  = (hpo_type == "pretraining")
    is_ft   = (hpo_type == "finetuning")

    random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    trials_dir = os.path.join(args.out_dir, "trials")
    ensure_dir(trials_dir)

    base_cfg = load_yaml(args.base_config)

    # Validate runner choice
    if (args.runner_cmd is None) == (args.runner_py is None):
        print("ERROR: specify exactly one of --runner_cmd or --runner_py", file=sys.stderr)
        sys.exit(2)

    # Load/resolve search space
    if args.space_json:
        with open(args.space_json, "r") as f:
            space = json.load(f)
    else:
        # Only use the default (model) space when we are doing architecture HPO
        space = DEFAULT_SEARCH_SPACE if is_arch else {}

    # Resolve dynamic entries only if space is non-empty
    space = materialize_space(space, base_cfg) if space else {}

    # Only do head/divisibility helpers for architecture runs
    if is_arch:
        # If NHEADS not provided in space, derive legal choices from decoder hidden dim
        if "MODEL.DECODER.NHEADS" not in space:
            if "MODEL.DECODER.HIDDEN_DIM" in space and space["MODEL.DECODER.HIDDEN_DIM"]:
                hd = int(space["MODEL.DECODER.HIDDEN_DIM"][0])
            else:
                hd = int(get_by_dotted_key(base_cfg, "MODEL.DECODER.HIDDEN_DIM"))
            space["MODEL.DECODER.NHEADS"] = [h for h in (4, 8) if hd % h == 0]

    results: List[TrialResult] = []
    best: Optional[TrialResult] = None

    # Record metadata
    with open(os.path.join(args.out_dir, "run_meta.json"), "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "seed": args.seed,
            "trials": args.trials,
            "base_config": args.base_config,
            "runner": args.runner_py or args.runner_cmd,
        }, f, indent=2)

    for t in range(args.trials):
        trial_id = t + 1
        params = sample_params(space) if space else {}

        if is_arch:
            # Tie decoder hidden dim to encoder hidden dim (DETR-style)
            enc_hd = params.get("MODEL.ENCODER.HIDDEN_DIM",
                                get_by_dotted_key(base_cfg, "MODEL.ENCODER.HIDDEN_DIM"))
            params["MODEL.DECODER.HIDDEN_DIM"] = enc_hd
            # Pick nheads compatible with final hidden dim
            hd = int(params["MODEL.DECODER.HIDDEN_DIM"])
            allowed_heads = [h for h in (4, 8) if hd % h == 0] or [4]
            params["MODEL.DECODER.NHEADS"] = random.choice(allowed_heads)


        # Apply overrides
        cfg = apply_overrides(base_cfg, params)
        # Force shorter training and frequent validation for early stopping
        try:
            cfg["TRAIN"]["EPOCHS"] = int(args.force_epochs)
        except Exception:
            pass
        try:
            cfg["TRAIN"]["VAL_INTERVAL"] = int(args.force_val_interval)
        except Exception:
            pass

        # (Recommended) Ensure stage flags for architecture search (target, from scratch)
        # Users may adjust these if their code expects other semantics.
        # Stage flags per hpo_type
        try:
            if is_arch:
                # Target-only supervised, DA off (capacity ranking)
                if "TRAIN" in cfg:
                    cfg["TRAIN"]["COMPUTE_TARGET_GRAPH_LOSS"] = True
                    cfg["TRAIN"]["IMAGE_ADVERSARIAL"] = False
                    cfg["TRAIN"]["GRAPH_ADVERSARIAL"] = False

            elif is_pre:
                # Source-only supervised pretraining HPO (no target loss, no DA)
                if "TRAIN" in cfg:
                    cfg["TRAIN"]["COMPUTE_TARGET_GRAPH_LOSS"] = False
                    cfg["TRAIN"]["IMAGE_ADVERSARIAL"] = False
                    cfg["TRAIN"]["GRAPH_ADVERSARIAL"] = False

            elif is_ft:
                # Target fine-tuning HPO (supervised on target). DA optional via space_json/base_cfg.
                if "TRAIN" in cfg:
                    cfg["TRAIN"]["COMPUTE_TARGET_GRAPH_LOSS"] = True
                    # Leave DA flags as in base_cfg, allow tuning via space_json if desired.
        except Exception:
            pass


        workdir = os.path.join(trials_dir, f"trial_{trial_id:04d}")
        ensure_dir(workdir)
        config_path = os.path.join(workdir, "config.yaml")
        dump_yaml(cfg, config_path)

        success = False
        metric = None
        notes = ""

        if args.runner_cmd:
            success, metric, notes = run_subprocess_runner(args.runner_cmd, config_path, workdir, args.metric_regex)
        else:
            # python runner
            success, metric, notes = run_python_runner(args.runner_py, cfg, workdir, args.metric_key)

        if not success or metric is None:
            print(f"[Trial {trial_id:04d}] FAILED | params={params} | notes={notes}")
            tr = TrialResult(trial_id, params, float("-inf"), config_path, workdir, False, notes)
        else:
            print(f"[Trial {trial_id:04d}] metric={metric:.6f} | params={params}")
            tr = TrialResult(trial_id, params, metric, config_path, workdir, True, "")

        results.append(tr)
        # Track best
        if tr.success and (best is None or tr.metric > best.metric):
            best = tr

            # Keys actually tuned in this run
            tuned_keys = list(best.params.keys())

            # Load the winning trial config and extract just the tuned fields
            best_cfg = load_yaml(best.config_path)
            patch = extract_architecture_patch(base_cfg, best_cfg, tuned_keys)

            # Normalize hpo_type if you did earlier (e.g., fix 'pretaining')
            hpo_type_norm = "pretraining" if getattr(args, "hpo_type", "") == "pretaining" else args.hpo_type

            # Write merged config (base + tuned patch)
            out_best_yaml = os.path.join(args.out_dir, f"best_{hpo_type_norm}.yaml")
            merged = apply_overrides(base_cfg, patch)
            dump_yaml(merged, out_best_yaml)

            # Also save the patch and the metric
            with open(os.path.join(args.out_dir, f"best_{hpo_type_norm}.txt"), "w") as f:
                f.write(json.dumps({
                    "trial_id": best.trial_id,
                    "metric": best.metric,
                    "tuned_params": patch
                }, indent=2))

            # (Nice to have) copy the exact winning trial config for full reproducibility
            shutil.copyfile(best.config_path, os.path.join(args.out_dir, f"best_{hpo_type_norm}_trial_config.yaml"))


    # Save summary CSV
    rows = []
    for tr in results:
        row = {
            "trial_id": tr.trial_id,
            "metric": tr.metric if tr.metric != float("-inf") else None,
            "success": tr.success,
            "notes": tr.notes,
            "config_path": tr.config_path,
            "workdir": tr.workdir,
        }
        row.update({f"param::{k}": v for k, v in tr.params.items()})
        rows.append(row)
    df = pd.DataFrame(rows).sort_values(by=["success", "metric"], ascending=[False, False])
    df.to_csv(os.path.join(args.out_dir, "summary.csv"), index=False)

    # Print final best
    if best and best.success:
        print(f"\nBest trial: {best.trial_id:04d}  metric={best.metric:.6f}")
        print(f"Best config: {os.path.join(args.out_dir, f'best_{hpo_type}.yaml')}")
    else:
        print("\nNo successful trials. Check runner configuration and logs.", file=sys.stderr)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Architecture search harness (random search).")
    parser.add_argument(
        "--hpo_type",
        choices=["architecture", "pretraining", "finetuning", "pretaining"],  # accept the typo too
        required=True,
        help="Type of HPO to run."
    )
    parser.add_argument("--base_config", required=True, help="Path to the base YAML config.")
    parser.add_argument("--out_dir", required=True, help="Output directory.")
    parser.add_argument("--trials", type=int, default=30, help="Number of random trials.")
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--space_json", type=str, default=None, help="Optional JSON file describing the search space.")
    # Runner options: choose exactly one
    parser.add_argument("--runner_cmd", type=str, default=None, help="Subprocess command template. Use {config_path} and/or {workdir}. Must print metric parseable by --metric_regex.")
    parser.add_argument("--metric_regex", type=str, default=r"VAL_METRIC=([0-9.]+)", help="Regex to capture a single float metric from stdout.")
    parser.add_argument("--runner_py", type=str, default=None, help="Python runner spec 'module:function'.")
    parser.add_argument("--metric_key", type=str, default=None, help="Key in result dict returned by python runner to read as metric.")
    # Force target validation cadence (optional)
    parser.add_argument("--force_epochs", type=int, default=20, help="Override TRAIN.EPOCHS (proxy training length).")
    parser.add_argument("--force_val_interval", type=int, default=1, help="Override TRAIN.VAL_INTERVAL for early stopping.")
    
    
    # --- 1: ARCHITECTURE ---
    # args = parser.parse_args([
    #     '--hpo_type', 'architecture',
    #     '--base_config', '/home/scavone/cross-dim_i2g/configs/HPO/1-architecture.yaml',
    #     '--out_dir', '/data/scavone/cross-dim_i2g/HPO',
    #     '--trials', '30',
    #     '--force_epochs', '20', '--force_val_interval', '1',
    #     '--runner_cmd', 'python /home/scavone/cross-dim_i2g/train_copy.py --config {config_path} --exp_name HPO_0',
    #     '--metric_regex', 'VAL_METRIC=([-0-9.]+)'
    # ])
    
    
    # --- 2: PRETRAINING ---
    args = parser.parse_args([
        '--hpo_type', 'pretraining',
        '--base_config', '/home/scavone/cross-dim_i2g/configs/HPO/2-pretraining.yaml',
        '--out_dir', '/data/scavone/cross-dim_i2g/HPO_pretraining',
        '--space_json', '/home/scavone/cross-dim_i2g/HPO/search_spaces/2-pretraining.json',
        '--trials', '30',
        '--force_epochs', '20', '--force_val_interval', '1',
        '--runner_cmd', 'python /home/scavone/cross-dim_i2g/train_copy.py --config {config_path} --exp_name HPO_pretraining',
        '--metric_regex', 'VAL_METRIC=([-0-9.]+)'
    ])
    
    
    main(args)
