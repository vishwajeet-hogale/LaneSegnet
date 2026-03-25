"""Lightweight inference script that groups results by scenario tags

Usage example:
  python tools/test_scenario.py \
    projects/configs/lanesegnet_r50_8x1_24e_olv2_subset_A.py \
    D:\\LaneSegNet\\ckpts\\lanesegnet_r50_8x1_24e_olv2_subset_A.pth \
    --data-root D:/TopoNet/data/OpenLane-V2 --split train --max-samples 200 \
    --out-dir results_scenario

The script follows the project's `tools/test.py` conventions but produces
per-scenario sample images and a small summary plot of counts per scenario.
"""
import argparse
import os
import os.path as osp
import warnings
from collections import defaultdict
from datetime import datetime

import mmcv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from mmcv import Config
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel

from mmdet.datasets import replace_ImageToTensor
from projects.lanesegnet.datasets.openlanev2_subset_A_lanesegnet_dataset import OpenLaneV2_subset_A_LaneSegNet_Dataset
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet3d.apis import single_gpu_test


def parse_args():
    parser = argparse.ArgumentParser(description='Run inference and group by scenario')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--data-root', help='OpenLane-V2 root', required=True)
    parser.add_argument('--split', default='train', help='dataset split to use')
    parser.add_argument('--max-samples', type=int, default=None, help='limit samples for quick runs')
    parser.add_argument('--out-dir', default='results_scenario', help='output directory for plots')
    parser.add_argument('--device', default='cuda:0', help='device to run model on')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--show', action='store_true', help='also use model show hooks (if supported)')
    parser.add_argument('--no-model', action='store_true', help='skip model building and inference (dry run)')
    parser.add_argument('--workers', type=int, default=None, help='workers per gpu (overrides config). If omitted defaults to 0')
    parser.add_argument('--debug-preds', action='store_true', help='Save and print a debug summary of model predictions')
    return parser.parse_args()


def get_scenario_tags_from_info(info):
    # Try multiple likely keys for scenario tags in dataset info
    keys = ['scenario_tags', 'scenario', 'scenarios', 'tags']
    for k in keys:
        if k in info:
            val = info[k]
            if isinstance(val, (list, tuple)):
                return [str(x) for x in val]
            return [str(val)]
    # fallback to annotation-level tags
    ann = info.get('annotation', {}) if isinstance(info, dict) else {}
    for k in ['scenario_tags', 'tags']:
        if k in ann:
            val = ann[k]
            if isinstance(val, (list, tuple)):
                return [str(x) for x in val]
            return [str(val)]
    return ['unknown']


def ensure_out_dirs(base_out):
    mmcv.mkdir_or_exist(base_out)
    mmcv.mkdir_or_exist(osp.join(base_out, 'per_scenario'))


def make_run_dir(base_out, run_name=None):
    import datetime
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    base = run_name or f'run_{ts}'
    run_dir = osp.join(base_out, base)
    i = 1
    while osp.exists(run_dir):
        run_dir = osp.join(base_out, f'{base}_{i}')
        i += 1
    mmcv.mkdir_or_exist(run_dir)
    return run_dir


def plot_metrics_by_category(global_metrics, per_metrics, counts, save_path):
    import numpy as np
    import matplotlib.pyplot as plt
    import numbers
    # pick numeric metric keys (accept numpy numeric types too)
    metric_keys = [k for k, v in (global_metrics or {}).items() if isinstance(v, numbers.Number) or np.isscalar(v)]
    if not metric_keys:
        return
    metric_keys = metric_keys[:4]
    tags = list(per_metrics.keys())
    n_tags = len(tags)
    data = np.zeros((len(metric_keys), n_tags))
    global_vals = []
    for mi, mk in enumerate(metric_keys):
        global_val = float((global_metrics or {}).get(mk, 0.0))
        global_vals.append(global_val)
        for ti, tag in enumerate(tags):
            data[mi, ti] = float(per_metrics.get(tag, {}).get(mk, 0.0) or 0.0)

    fig = plt.figure(figsize=(12, 10))
    ax1 = fig.add_subplot(2, 1, 1)
    x = np.arange(n_tags)
    width = 0.15
    for i in range(len(metric_keys)):
        ax1.bar(x + i * width, data[i], width=width, label=metric_keys[i])
    ax1.set_xticks(x + width * (len(metric_keys) - 1) / 2)
    xticklabels = [f"{t}\n(n={counts[i]})" for i, t in enumerate(tags)]
    ax1.set_xticklabels(xticklabels, rotation=30, ha='right')
    ax1.set_title('Metrics By Category')
    ax1.legend()

    ax2 = fig.add_subplot(2, 1, 2)
    gaps = data[0] - global_vals[0]
    ax2.bar(x, gaps, color='C0')
    ax2.axhline(0, color='k')
    ax2.set_xticks(x)
    ax2.set_xticklabels(xticklabels, rotation=30, ha='right')
    ax2.set_title('Category Gap vs Global (negative = worse)')
    plt.tight_layout()
    plt.savefig(save_path)


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    print(f"[test_scenario] Loaded config: {args.config}")
    # override test dataset to use folder dataset loader
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        # force batch size to 1 for deterministic single-sample inference
        cfg.data.test.samples_per_gpu = 1
        cfg.data.test.type = 'OpenLaneV2FolderDataset'
        cfg.data.test.data_root = args.data_root
        cfg.data.test.split = args.split
        # ann_file not needed for folder loader
        cfg.data.test.ann_file = ''
    else:
        # replace the single entry
        cfg.data.test[0].test_mode = True
        cfg.data.test[0].samples_per_gpu = 1
        cfg.data.test[0].type = 'OpenLaneV2FolderDataset'
        cfg.data.test[0].data_root = args.data_root
        cfg.data.test[0].split = args.split
        cfg.data.test[0].ann_file = ''

    # If a preprocessed annotation pkl exists, prefer the LaneSegNet dataset
    # which provides proper formatting and evaluation utilities.
    pkl_candidate = osp.join(args.data_root, 'data_dict_subset_A_train_lanesegnet.pkl')
    if args.split == 'train' and osp.exists(pkl_candidate):
        print(f"[test_scenario] Found preprocessed annotation pkl: {pkl_candidate}, using LaneSegNet dataset for evaluation")
        if isinstance(cfg.data.test, dict):
            cfg.data.test.type = 'OpenLaneV2_subset_A_LaneSegNet_Dataset'
            cfg.data.test.ann_file = pkl_candidate
        else:
            cfg.data.test[0].type = 'OpenLaneV2_subset_A_LaneSegNet_Dataset'
            cfg.data.test[0].ann_file = pkl_candidate

    # ensure one sample per gpu for deterministic iteration
    samples_per_gpu = 1
    # set workers per gpu (allow CLI override). Default to 0 if not present.
    workers_per_gpu = args.workers if args.workers is not None else getattr(cfg.data, 'workers_per_gpu', 0)

    # build the dataset and optionally limit samples
    print(f"[test_scenario] Building dataset with config.data.test: {cfg.data.test}")
    dataset = build_dataset(cfg.data.test)
    print(f"[test_scenario] Built dataset, found {len(getattr(dataset, 'data_infos', []))} samples")
    if args.max_samples is not None and hasattr(dataset, 'data_infos'):
        dataset.data_infos = dataset.data_infos[: args.max_samples]
        print(f"[test_scenario] Trimmed dataset to max_samples={args.max_samples}, now {len(dataset.data_infos)} samples")

    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=workers_per_gpu,
        dist=False,
        shuffle=False,
    )
    print(f"[test_scenario] Built dataloader (samples_per_gpu={samples_per_gpu})")


    def ensure_prediction_keys(pred):
        # Ensures all required keys are present in prediction dict
        required_keys = [
            'lane_results',
            'bbox_results',
            'lsls_results',
            'lste_results',
            # 'area_results'  # Uncomment if your model outputs this
        ]
        if not isinstance(pred, dict):
            return pred
        for k in required_keys:
            if k not in pred or pred[k] is None:
                # Use empty np arrays for missing keys
                if k.endswith('_results'):
                    import numpy as _np
                    pred[k] = _np.zeros(0, dtype=_np.float32)
        # Filter out non-finite values in all result arrays
        import numpy as _np
        for k in pred:
            if k.endswith('_results') and pred[k] is not None:
                arr = pred[k]
                # If it's a list, check if it's ragged (list of arrays with different shapes)
                if isinstance(arr, list):
                    # If all elements are arrays and shapes match, stack; else, treat as ragged
                    if len(arr) > 0 and all(isinstance(a, _np.ndarray) for a in arr):
                        shapes = [a.shape for a in arr]
                        if all(s == shapes[0] for s in shapes):
                            arr = _np.stack(arr)
                            # Now arr is a regular ndarray
                            arr = arr[_np.all(_np.isfinite(arr), axis=tuple(range(1, arr.ndim)))]
                            pred[k] = arr
                        else:
                            # Ragged: filter each array individually
                            arr = [a for a in arr if _np.all(_np.isfinite(a))]
                            pred[k] = arr
                    else:
                        # Try to convert to ndarray if possible
                        try:
                            arr2 = _np.array(arr)
                            if arr2.dtype.kind in {'f', 'i'}:
                                arr2 = arr2[_np.isfinite(arr2)]
                                pred[k] = arr2
                        except Exception:
                            pass
                elif isinstance(arr, _np.ndarray):
                    if arr.dtype.kind in {'f', 'i'}:
                        if arr.ndim == 1:
                            arr = arr[_np.isfinite(arr)]
                        else:
                            arr = arr[_np.all(_np.isfinite(arr), axis=1)]
                        pred[k] = arr
        return pred
        return pred

    outputs = None
    if not args.no_model:
        print(f"[test_scenario] Building model and loading checkpoint: {args.checkpoint}")
        # build model
        cfg.model.pretrained = None
        model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
        fp16_cfg = cfg.get('fp16', None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)

        checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
        print(f"[test_scenario] Loaded checkpoint")
        if args.device and args.device.startswith('cuda'):
            device_id = int(args.device.split(':')[-1])
        else:
            device_id = None

        # fuse and move to GPU
        try:
            model = fuse_conv_bn(model)
        except Exception:
            # optional
            pass

        # set classes from checkpoint or dataset
        if 'meta' in checkpoint and 'CLASSES' in checkpoint['meta']:
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            model.CLASSES = getattr(dataset, 'CLASSES', None)

        # If device is cpu, do not wrap in MMDataParallel and move model to cpu
        if args.device and args.device.lower().startswith('cpu'):
            model = model.cpu()
            print("[test_scenario] Running on CPU (no DataParallel)")

            # Patch single_gpu_test to unwrap DataContainer for CPU mode
            from mmcv.parallel import DataContainer
            from mmdet3d.apis.test import single_gpu_test as orig_single_gpu_test
            import types

            def single_gpu_test_unwrap(model, data_loader, show=False, out_dir=None):
                model.eval()
                results = []
                dataset = data_loader.dataset
                prog_bar = mmcv.ProgressBar(len(dataset))
                for i, data in enumerate(data_loader):
                    # Unwrap DataContainer to plain data for CPU
                    for k, v in data.items():
                        if isinstance(v, DataContainer):
                            data[k] = v.data[0]
                    # Special case for img_metas
                    if 'img_metas' in data and isinstance(data['img_metas'], DataContainer):
                        data['img_metas'] = data['img_metas'].data[0]
                    with torch.no_grad():
                        result = model(return_loss=False, rescale=True, **data)
                    results.append(result)
                    prog_bar.update()
                return results

            outputs = single_gpu_test_unwrap(model, data_loader, args.show, None)
        else:
            model = MMDataParallel(model, device_ids=[device_id] if device_id is not None else None)
            print(f"[test_scenario] Running inference...")
            outputs = single_gpu_test(model, data_loader, args.show, None)
        print(f"[test_scenario] Inference finished, got {len(outputs) if outputs is not None else 0} outputs")
    else:
        # dry run: create placeholder outputs aligned with dataset length
        infos = getattr(dataset, 'data_infos', [])
        outputs = [None] * len(infos)
        print(f"[test_scenario] Dry run mode: created {len(outputs)} placeholder outputs")

    # Ensure all predictions have required keys
    outputs = [ensure_prediction_keys(o) for o in outputs]

    # save raw outputs into a run-specific directory to avoid overwrites
    mmcv.mkdir_or_exist(args.out_dir)
    run_dir = make_run_dir(args.out_dir)
    mmcv.mkdir_or_exist(osp.join(run_dir, 'per_scenario'))
    out_pkl = osp.join(run_dir, 'results.pkl')
    mmcv.dump(outputs, out_pkl)
    # save run metadata
    meta = dict(cmdline=' '.join(os.sys.argv), config=args.config, checkpoint=args.checkpoint)
    mmcv.dump(meta, osp.join(run_dir, 'run_meta.json'))
    print(f"[test_scenario] Wrote raw outputs to {out_pkl} (run dir: {run_dir})")

    # Debug: serialize and print lightweight prediction summaries
    if getattr(args, 'debug_preds', False):
        import json
        import numpy as _np

        def _ser(o):
            # convert numpy + scalar types to python native types for JSON
            if isinstance(o, dict):
                return {k: _ser(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)):
                return [_ser(x) for x in o]
            if isinstance(o, _np.ndarray):
                return o.tolist()
            try:
                # handle numpy scalars
                if hasattr(o, 'item'):
                    return o.item()
            except Exception:
                pass
            # fallback primitives
            if isinstance(o, (int, float, str, type(None), bool)):
                return o
            # unknown type: string repr
            return str(o)


        total_debug = len(outputs)
        preds_ser = []
        for i, out in enumerate(outputs[:total_debug]):
            # If output is a single-item list containing a dict, flatten it
            if isinstance(out, list) and len(out) == 1 and isinstance(out[0], dict):
                out = out[0]
            preds_ser.append(_ser(out))

        debug_path = osp.join(run_dir, 'predictions_debug.json')
        with open(debug_path, 'w', encoding='utf-8') as f:
            json.dump(preds_ser, f, indent=2)

        print(f"[test_scenario] Saved prediction debug JSON to: {debug_path}")
        # Print brief preview to console
        for i, p in enumerate(preds_ser[:min(10, len(preds_ser))]):
            keys = list(p.keys()) if isinstance(p, dict) else []
            print(f"PRED[{i}] keys={keys} preview:")
            if isinstance(p, dict):
                # show small nested preview for common fields
                for k in keys[:6]:
                    v = p.get(k)
                    if isinstance(v, list):
                        print(f"  {k}: list(len={len(v)}) sample={v[:2]}")
                    else:
                        print(f"  {k}: {str(v)[:200]}")
            else:
                print(str(p)[:400])

    # group by scenario tags and save example images
    per_scenario = defaultdict(list)
    # dataset.data_infos should align with outputs
    infos = getattr(dataset, 'data_infos', None)
    if infos is None:
        warnings.warn('Dataset has no data_infos attribute; cannot group by scenario.')
        return

    for idx, raw in enumerate(infos):
        # if dataset is lazy-loading, raw entries contain 'info_path'
        if isinstance(raw, dict) and 'info_path' in raw:
            info = dataset._load_info(idx)
        else:
            info = raw
        tags = get_scenario_tags_from_info(info)
        for t in tags:
            per_scenario[t].append(idx)
    print(f"[test_scenario] Grouped samples into {len(per_scenario)} scenario tags")

    # save small sample images per scenario and a count plot
    base_dir = osp.join(run_dir, 'per_scenario')
    for tag, indices in per_scenario.items():
        safe_tag = str(tag).replace(' ', '_')[:120]
        tag_dir = osp.join(base_dir, safe_tag)
        mmcv.mkdir_or_exist(tag_dir)
        # take up to 6 samples to show
        for i, idx in enumerate(indices[:6]):
            data_info = dataset.get_data_info(idx)
            # get first camera image if available
            img_field = data_info.get('img_filename') or data_info.get('img')
            if isinstance(img_field, (list, tuple)):
                img_path = img_field[0]
            else:
                img_path = img_field
            if img_path and osp.exists(img_path):
                img = mmcv.imread(img_path)
                # simple plot with text overlay
                fig, ax = plt.subplots(1, 1, figsize=(10, 6))
                ax.imshow(img[:, :, ::-1])  # BGR -> RGB
                ax.axis('off')
                ax.set_title(f'sample={idx} tag={tag}')
                save_path = osp.join(tag_dir, f'sample_{i}_{idx}.png')
                fig.savefig(save_path, bbox_inches='tight')
                plt.close(fig)
            else:
                # create placeholder image
                fig, ax = plt.subplots(1, 1, figsize=(6, 3))
                ax.text(0.5, 0.5, f'No image for sample {idx}', ha='center')
                ax.axis('off')
                save_path = osp.join(tag_dir, f'sample_{i}_{idx}_noimg.png')
                fig.savefig(save_path, bbox_inches='tight')
                plt.close(fig)
        print(f"[test_scenario] Wrote up to 6 sample images for tag '{tag}' ({len(indices)} total)")

    # summary bar plot
    tags = list(per_scenario.keys())
    counts = [len(per_scenario[t]) for t in tags]
    # sort by count
    pairs = sorted(zip(tags, counts), key=lambda x: -x[1])
    tags_sorted, counts_sorted = zip(*pairs) if pairs else ([], [])
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(tags_sorted)), counts_sorted)
    plt.xticks(range(len(tags_sorted)), tags_sorted, rotation=45, ha='right')
    plt.ylabel('sample count')
    plt.title('Samples per scenario tag')
    plt.tight_layout()
    summary_path = osp.join(run_dir, 'scenario_counts.png')
    plt.savefig(summary_path)
    print(f'Wrote scenario summary to {summary_path}')

    # Ensure dataset.data_infos entries contain full annotations before evaluation
    print(f"[test_scenario] Preparing annotations for evaluation...")
    try:
        for i in range(len(dataset.data_infos)):
            info = dataset.data_infos[i]
            if isinstance(info, dict) and 'annotation' not in info:
                dataset.data_infos[i] = dataset._load_info(i)
    except Exception as e:
        print(f"[test_scenario] Warning: failed to fully load some annotations: {e}")

    # Normalize annotation dicts so evaluation code finds expected keys
    def _ensure_annotation_fields(info):
        if 'annotation' not in info or info['annotation'] is None:
            info['annotation'] = {}
        ann = info['annotation']
        # keys expected by format_results/format_openlanev2_gt
        if 'area' not in ann or ann['area'] is None:
            ann['area'] = []
        if 'lane_segment' not in ann or ann['lane_segment'] is None:
            ann['lane_segment'] = []
        if 'traffic_element' not in ann or ann['traffic_element'] is None:
            ann['traffic_element'] = []
        if 'topology_lsls' not in ann or ann['topology_lsls'] is None:
            ann['topology_lsls'] = []
        if 'topology_lste' not in ann or ann['topology_lste'] is None:
            ann['topology_lste'] = []
        # convert common list fields to numpy arrays where expected
        try:
            import numpy as _np
            # areas: ensure points is ndarray (N,2) or (N,?)
            for a in ann.get('area', []):
                if 'points' in a and not isinstance(a['points'], _np.ndarray):
                    a['points'] = _np.array(a.get('points', []), dtype=_np.float32)

            # lane segments: centerline, left_laneline, right_laneline
            for ls in ann.get('lane_segment', []):
                if 'centerline' in ls and not isinstance(ls['centerline'], _np.ndarray):
                    ls['centerline'] = _np.array(ls.get('centerline', []), dtype=_np.float32)
                if 'left_laneline' in ls and not isinstance(ls['left_laneline'], _np.ndarray):
                    ls['left_laneline'] = _np.array(ls.get('left_laneline', []), dtype=_np.float32)
                if 'right_laneline' in ls and not isinstance(ls['right_laneline'], _np.ndarray):
                    ls['right_laneline'] = _np.array(ls.get('right_laneline', []), dtype=_np.float32)

            # traffic elements: points
            for te in ann.get('traffic_element', []):
                if 'points' in te and not isinstance(te['points'], _np.ndarray):
                    te['points'] = _np.array(te.get('points', []), dtype=_np.float32)

            # topology matrices
            if ann.get('topology_lsls') is not None and not isinstance(ann['topology_lsls'], _np.ndarray):
                ann['topology_lsls'] = _np.array(ann.get('topology_lsls', []), dtype=_np.float32)
            if ann.get('topology_lste') is not None and not isinstance(ann['topology_lste'], _np.ndarray):
                ann['topology_lste'] = _np.array(ann.get('topology_lste', []), dtype=_np.float32)
        except Exception:
            pass

        info['annotation'] = ann
        return info

    for i in range(len(dataset.data_infos)):
        try:
            dataset.data_infos[i] = _ensure_annotation_fields(dataset.data_infos[i])
        except Exception:
            # last resort: reload and normalize
            try:
                info = dataset._load_info(i)
                dataset.data_infos[i] = _ensure_annotation_fields(info)
            except Exception:
                dataset.data_infos[i] = {'annotation': {'area': [], 'lane_segment': [], 'traffic_element': [], 'topology_lsls': [], 'topology_lste': []}}

    # Use the richer evaluator/aggregator to produce per-scenario CSVs and plots
    try:
        eval_kwargs = _sanitize_eval_kwargs(getattr(cfg, 'evaluation', {}) or {})
    except Exception:
        eval_kwargs = {}

    try:
        # pass debug flag to evaluator via attribute
        evaluate_by_scenario.debug = bool(getattr(args, 'debug_preds', False))
        eval_out_dir = evaluate_by_scenario(dataset, outputs, eval_kwargs, args.out_dir)
        print(f"[test_scenario] Aggregated evaluation and visualizations saved to: {eval_out_dir}")
    except Exception as e:
        print(f"[test_scenario] Warning: evaluate_by_scenario failed: {e}")


# --- Advanced visualizations and scenario evaluator (replicated from user's aggregator) ---
METRIC_LABELS = {
    'OpenLane-V2 Score': 'OLV2 Score',
    'DET_l': 'DET$_l$',
    'DET_t': 'DET$_t$',
    'TOP_ll': 'TOP$_{ll}$',
    'TOP_lt': 'TOP$_{lt}$',
}
METRIC_COLS = list(METRIC_LABELS.keys())
SCENARIO_TITLES = {
    'curvature': 'Road Curvature',
    'lighting': 'Lighting Condition',
    'occlusion': 'Occlusion Level',
    'topology_complexity': 'Topology Complexity',
}
PALETTE = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3']
METRIC_COLS_EVAL = ['OpenLane-V2 Score', 'DET_l', 'DET_t', 'TOP_ll', 'TOP_lt']


def _bar_chart_per_scenario(df, scenario_type, global_score, run_dir):
    categories = df.index.tolist()
    n_cats = len(categories)
    n_metrics = len(METRIC_COLS)
    x = np.arange(n_cats)
    width = 0.15

    fig, ax = plt.subplots(figsize=(max(7, n_cats * 2.2), 5))
    for j, metric in enumerate(METRIC_COLS):
        vals = df[metric].values.astype(float)
        bars = ax.bar(x + j * width, vals, width, label=METRIC_LABELS[metric],
                      color=PALETTE[j], edgecolor='white', linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=7)

    ax.axhline(y=global_score, color='grey', linestyle='--', linewidth=1,
               label=f'Global OLV2 ({global_score:.3f})')
    ax.set_xticks(x + width * (n_metrics - 1) / 2)
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title(f'Scenario: {SCENARIO_TITLES.get(scenario_type, scenario_type)}',
                 fontsize=13, fontweight='bold')
    ax.set_ylim(0, min(1.0, df[METRIC_COLS].max().max() + 0.12))
    ax.legend(fontsize=8, ncol=3, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(osp.join(run_dir, f'bar_{scenario_type}.png'), dpi=200)
    fig.savefig(osp.join(run_dir, f'bar_{scenario_type}.pdf'))
    plt.close(fig)


def _score_drop_chart(scenario_tables, global_score, run_dir):
    rows = []
    for stype, df in scenario_tables.items():
        for cat in df.index:
            val = df.loc[cat, 'OpenLane-V2 Score']
            delta = float(val if val is not None else 0.0) - global_score
            samples = df.loc[cat, 'samples']
            rows.append({'scenario': SCENARIO_TITLES.get(stype, stype),
                         'category': cat, 'delta': delta,
                         'samples': int(samples if samples is not None else 0)})
    if not rows:
        return
    delta_df = pd.DataFrame(rows).sort_values('delta')

    fig, ax = plt.subplots(figsize=(8, max(4, len(rows) * 0.55)))
    colors = ['#C44E52' if d < 0 else '#55A868' for d in delta_df['delta']]
    labels = [f"{r['category']}  (n={r['samples']})" for _, r in delta_df.iterrows()]
    bars = ax.barh(range(len(delta_df)), delta_df['delta'], color=colors,
                   edgecolor='white', height=0.6)
    ax.set_yticks(range(len(delta_df)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(x=0, color='grey', linewidth=0.8)
    ax.set_xlabel('$\\Delta$ OLV2 Score (vs. global)', fontsize=11)
    ax.set_title('Performance Gap by Scenario Category', fontsize=13, fontweight='bold')
    for bar, v in zip(bars, delta_df['delta']):
        ax.text(v + (0.003 if v >= 0 else -0.003),
                bar.get_y() + bar.get_height() / 2,
                f'{v:+.3f}', ha='left' if v >= 0 else 'right',
                va='center', fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(osp.join(run_dir, 'score_delta.png'), dpi=200)
    fig.savefig(osp.join(run_dir, 'score_delta.pdf'))
    plt.close(fig)


def _heatmap(scenario_tables, run_dir):
    rows = []
    labels = []
    for stype, df in scenario_tables.items():
        prefix = SCENARIO_TITLES.get(stype, stype)
        for cat in df.index:
            labels.append(f'{prefix} / {cat}')
            row_vals = [float(df.loc[cat, m] if df.loc[cat, m] is not None else 0.0) for m in METRIC_COLS]
            rows.append(np.array(row_vals, dtype=float))
    if not rows:
        return
    mat = np.array(rows)

    fig, ax = plt.subplots(figsize=(8, max(4, len(labels) * 0.5)))
    im = ax.imshow(mat, aspect='auto', cmap='RdYlGn',
                   vmin=0, vmax=max(0.6, mat.max() + 0.05))
    ax.set_xticks(range(len(METRIC_COLS)))
    ax.set_xticklabels([METRIC_LABELS[m] for m in METRIC_COLS], fontsize=9)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f'{mat[i, j]:.3f}', ha='center', va='center',
                    fontsize=8, color='white' if mat[i, j] < 0.2 else 'black')
    ax.set_title('Metric Heatmap Across All Scenarios', fontsize=13, fontweight='bold')
    fig.colorbar(im, ax=ax, shrink=0.8, label='Score')
    fig.tight_layout()
    fig.savefig(osp.join(run_dir, 'heatmap.png'), dpi=200)
    fig.savefig(osp.join(run_dir, 'heatmap.pdf'))
    plt.close(fig)


def _sample_distribution(scenario_tables, run_dir):
    n_types = len(scenario_tables)
    if n_types == 0:
        return
    fig, axes = plt.subplots(1, n_types, figsize=(4.5 * n_types, 4))
    if n_types == 1:
        axes = [axes]
    for ax, (stype, df) in zip(axes, scenario_tables.items()):
        counts = df['samples'].astype(int)
        cat_labels = [f'{cat}\n(n={int(c)})' for cat, c in zip(df.index, counts)]
        ax.pie(counts, labels=cat_labels, autopct='%1.0f%%', startangle=90,
               colors=PALETTE[:len(counts)], textprops={'fontsize': 8})
        ax.set_title(SCENARIO_TITLES.get(stype, stype), fontsize=11, fontweight='bold')
    fig.suptitle('Sample Distribution by Scenario', fontsize=13,
                 fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(osp.join(run_dir, 'sample_distribution.png'), dpi=200,
                bbox_inches='tight')
    fig.savefig(osp.join(run_dir, 'sample_distribution.pdf'),
                bbox_inches='tight')
    plt.close(fig)


def _radar_chart(scenario_tables, global_row, run_dir):
    angles = np.linspace(0, 2 * np.pi, len(METRIC_COLS), endpoint=False).tolist()
    angles += angles[:1]

    for stype, df in scenario_tables.items():
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        gvals = [float(global_row.get(m) if global_row.get(m) is not None else 0.0) for m in METRIC_COLS]
        gvals += gvals[:1]
        ax.plot(angles, gvals, 'k--', linewidth=1.2, label='Global')
        ax.fill(angles, gvals, alpha=0.05, color='grey')
        for idx, cat in enumerate(df.index):
            vals = [float(df.loc[cat, m] if df.loc[cat, m] is not None else 0.0) for m in METRIC_COLS]
            vals += vals[:1]
            ax.plot(angles, vals, linewidth=1.5,
                    label=f'{cat} (n={int(df.loc[cat, "samples"])})',
                    color=PALETTE[idx % len(PALETTE)])
            ax.fill(angles, vals, alpha=0.08,
                    color=PALETTE[idx % len(PALETTE)])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([METRIC_LABELS[m] for m in METRIC_COLS], fontsize=9)
        ax.set_ylim(0, min(1.0, df[METRIC_COLS].max().max() + 0.15))
        ax.set_title(f'{SCENARIO_TITLES.get(stype, stype)}',
                     fontsize=13, fontweight='bold', pad=20)
        ax.legend(fontsize=8, loc='upper right', bbox_to_anchor=(1.3, 1.1))
        fig.tight_layout()
        fig.savefig(osp.join(run_dir, f'radar_{stype}.png'), dpi=200,
                    bbox_inches='tight')
        fig.savefig(osp.join(run_dir, f'radar_{stype}.pdf'),
                    bbox_inches='tight')
        plt.close(fig)


def generate_visualizations(scenario_tables, global_row, run_dir):
    global_score = float(global_row.get('OpenLane-V2 Score') or 0.0)

    for stype, df in scenario_tables.items():
        _bar_chart_per_scenario(df, stype, global_score, run_dir)

    _score_drop_chart(scenario_tables, global_score, run_dir)
    _heatmap(scenario_tables, run_dir)
    _sample_distribution(scenario_tables, run_dir)
    _radar_chart(scenario_tables, global_row, run_dir)

    print(f"\nVisualizations saved to: {run_dir}")


def _sanitize_eval_kwargs(eval_kwargs):
    cleaned = dict(eval_kwargs) if eval_kwargs is not None else {}
    for key in ['interval', 'tmpdir', 'start', 'gpu_collect', 'save_best', 'rule']:
        cleaned.pop(key, None)
    return cleaned


def _get_sample_scenario_labels(info):
    out = {}
    meta = info.get('scenario_meta', {})

    if 'curvature' in meta:
        v = meta['curvature'].get('value_m_inv')
        t = meta['curvature'].get('thresholds_m_inv', {})
        if v is not None:
            if v <= t.get('straight', 0.003):
                out['curvature'] = 'straight'
            elif v <= t.get('low', 0.008):
                out['curvature'] = 'low curvature'
            elif v <= t.get('medium', 0.02):
                out['curvature'] = 'medium curvature'
            else:
                out['curvature'] = 'high curvature'

    if 'topology_complexity' in meta:
        v = meta['topology_complexity'].get('value')
        if v is not None:
            if v <= 0.3:
                out['topology_complexity'] = 'low topology complexity'
            elif v <= 0.6:
                out['topology_complexity'] = 'medium topology complexity'
            else:
                out['topology_complexity'] = 'high topology complexity'

    if 'lighting' in meta:
        lbl = meta['lighting'].get('label')
        if lbl:
            out['lighting'] = lbl

    if 'occlusion' in meta:
        lbl = meta['occlusion'].get('label')
        if lbl:
            out['occlusion'] = lbl

    return out


def evaluate_by_scenario(dataset, outputs, eval_kwargs, out_dir):
    # debug flag (backwards compatible) - look for global var set by caller
    debug = False
    metric_cols = METRIC_COLS_EVAL
    all_rows = []

    print("\n==============================")
    print("GLOBAL METRICS")
    print("==============================")

    # Respect trimmed dataset.data_infos (max-samples): determine total samples
    total = len(getattr(dataset, 'data_infos', dataset))
    # ensure outputs length matches total
    outputs = outputs[:total]

    # Prepare eval run directory (so per-category evaluators can write there)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = osp.join(out_dir, f"eval_{timestamp}")
    mmcv.mkdir_or_exist(run_dir)

    # Monkey-patch numpy.percentile to accept the newer 'method' kwarg on older numpy
    try:
        import numpy as _np
        import inspect as _inspect
        sig = _inspect.signature(_np.percentile)
        if 'method' not in sig.parameters:
            _orig_percentile = _np.percentile
            def _percentile_with_method(*args, **kwargs):
                method = kwargs.pop('method', None)
                if method is not None:
                    allowed = {'linear', 'midpoint', 'lower', 'higher', 'nearest'}
                    interp = method if method in allowed else 'linear'
                    if 'interpolation' not in kwargs:
                        kwargs['interpolation'] = interp
                return _orig_percentile(*args, **kwargs)
            _np.percentile = _percentile_with_method
    except Exception:
        pass

    # Use LaneSegNet dataset evaluator which expects formatted outputs
    try:
        global_ds = OpenLaneV2_subset_A_LaneSegNet_Dataset.__new__(OpenLaneV2_subset_A_LaneSegNet_Dataset)
        global_ds.data_root = getattr(dataset, 'data_root', None)
        global_ds.split = getattr(dataset, 'split', None)
        global_ds.points_num = getattr(dataset, 'points_num', None)
        global_ds.CLASSES = getattr(dataset, 'CLASSES', None)
        global_ds.data_infos = dataset.data_infos
        logger = mmcv.get_logger('test_scenario')
        global_metrics = OpenLaneV2_subset_A_LaneSegNet_Dataset.evaluate(global_ds, outputs, logger=logger, out_dir=run_dir)
    except Exception as e:
        print(f"[test_scenario] Warning: global evaluation failed: {e}")
        global_metrics = {}

    global_row = {'scenario_type': 'global', 'category': 'all', 'samples': total}
    global_row.update({k: global_metrics.get(k) for k in metric_cols})
    all_rows.append(global_row)

    # Helper: readable sample id
    def _sample_identifier(info, idx):
        if isinstance(info, dict):
            for k in ('segment_id', 'scene_token', 'token'):
                if info.get(k) is not None:
                    return str(info.get(k))
        return f'sample_{idx:07d}'

    # Helper: summarize prediction for debug
    def _pred_summary(obj, max_items=3):
        try:
            if obj is None:
                return 'None'
            if isinstance(obj, dict):
                return {k: _pred_summary(v, max_items) for k, v in list(obj.items())[:max_items]}
            if isinstance(obj, (list, tuple)):
                return [_pred_summary(x, max_items) for x in obj[:max_items]]
            import numpy as _np
            if isinstance(obj, _np.ndarray):
                if obj.size == 0:
                    return {'ndarray': obj.shape, 'finite': True}
                finite = _np.isfinite(obj).all()
                return {'ndarray': obj.shape, 'finite': bool(finite), 'min': float(_np.nanmin(obj)), 'max': float(_np.nanmax(obj))}
            # numpy scalars
            try:
                if hasattr(obj, 'item'):
                    v = obj.item()
                    return v
            except Exception:
                pass
            if isinstance(obj, (int, float, str, bool)):
                return obj
            return str(type(obj))
        except Exception as e:
            return f'ERR:{e}'

    # If debug enabled (caller may set attribute on module), check and print summaries
    if getattr(evaluate_by_scenario, 'debug', False):
        print(f"[eval debug] total samples={total}, outputs={len(outputs)}")
        for i in range(total):
            info = dataset.data_infos[i]
            sid = _sample_identifier(info, i)
            pred = outputs[i] if i < len(outputs) else None
            print(f"[eval debug] SAMPLE {i} id={sid}")
            print("  info keys:", list(info.keys())[:10])
            print("  pred summary:", _pred_summary(pred))

    # Additional debug: scan predictions and annotations for non-finite values
    if getattr(evaluate_by_scenario, 'debug', False):
        import numpy as _np
        bad_samples = []
        for i in range(total):
            issues = []
            # check annotations
            info = dataset.data_infos[i]
            ann = info.get('annotation', {}) if isinstance(info, dict) else {}
            # inspect arrays in ann
            def _scan_obj(o, prefix='ann'):
                if isinstance(o, dict):
                    for k, v in o.items():
                        _scan_obj(v, prefix=f"{prefix}.{k}")
                elif isinstance(o, (list, tuple)):
                    for idx, v in enumerate(o[:10]):
                        _scan_obj(v, prefix=f"{prefix}[{idx}]")
                else:
                    try:
                        arr = _np.array(o)
                        if arr.size > 0 and not _np.isfinite(arr).all():
                            issues.append(f"{prefix} has non-finite values")
                    except Exception:
                        pass

            _scan_obj(ann)

            # check prediction arrays
            pred = outputs[i] if i < len(outputs) else None
            def _scan_pred(o, prefix='pred'):
                if isinstance(o, dict):
                    for k, v in o.items():
                        _scan_pred(v, prefix=f"{prefix}.{k}")
                elif isinstance(o, (list, tuple)):
                    for idx, v in enumerate(o[:10]):
                        _scan_pred(v, prefix=f"{prefix}[{idx}]")
                else:
                    try:
                        arr = _np.array(o)
                        if arr.size > 0 and not _np.isfinite(arr).all():
                            issues.append(f"{prefix} has non-finite values")
                    except Exception:
                        pass

            _scan_pred(pred)

            if issues:
                bad_samples.append({'index': i, 'id': _sample_identifier(info, i), 'issues': issues})

        if bad_samples:
            print(f"[eval debug] Found {len(bad_samples)} samples with non-finite values:")
            for b in bad_samples:
                print(f"  idx={b['index']} id={b['id']} issues={b['issues']}")
        else:
            print("[eval debug] No non-finite values found in annotations or predictions.")

    scenarios = {
        "curvature": {},
        "lighting": {},
        "occlusion": {},
        "topology_complexity": {}
    }

    for i in range(total):
        for stype, label in _get_sample_scenario_labels(dataset.data_infos[i]).items():
            scenarios[stype].setdefault(label, []).append(i)

    print("\n==============================")
    print("SCENARIO BREAKDOWN")
    print("==============================")

    scenario_tables = {}

    for scenario_type in scenarios:
        print("\n---", scenario_type.upper(), "---")

        if len(scenarios[scenario_type]) == 0:
            print(f"  No data available")
            continue

        rows = []
        for category in sorted(scenarios[scenario_type].keys()):

            indices = scenarios[scenario_type][category]
            subset_outputs = [outputs[i] for i in indices]

            # build a lightweight LaneSegNet dataset for this subset and evaluate
            try:
                sub_ds = OpenLaneV2_subset_A_LaneSegNet_Dataset.__new__(OpenLaneV2_subset_A_LaneSegNet_Dataset)
                sub_ds.data_root = getattr(dataset, 'data_root', None)
                sub_ds.split = getattr(dataset, 'split', None)
                sub_ds.points_num = getattr(dataset, 'points_num', None)
                sub_ds.CLASSES = getattr(dataset, 'CLASSES', None)
                sub_ds.data_infos = [dataset.data_infos[i] for i in indices]
                logger = mmcv.get_logger('test_scenario')
                metrics = OpenLaneV2_subset_A_LaneSegNet_Dataset.evaluate(sub_ds, subset_outputs, logger=logger, out_dir=run_dir)
            except Exception as e:
                print(f"[test_scenario] Warning: evaluation for category {category} failed: {e}")
                metrics = {}

            row = {'category': category, 'samples': len(indices)}
            row.update({k: metrics.get(k) for k in metric_cols})
            rows.append(row)

            combined_row = {'scenario_type': scenario_type, **row}
            all_rows.append(combined_row)

            print(f"  {category}: {len(indices)} samples")
            print(f"    Metrics: {metrics}")

        df = pd.DataFrame(rows).set_index('category')
        scenario_tables[scenario_type] = df

    # run_dir already created above

    global_df = pd.DataFrame([global_row]).set_index('scenario_type')
    global_df.to_csv(osp.join(run_dir, "global_metrics.csv"))

    for scenario_type, df in scenario_tables.items():
        df.to_csv(osp.join(run_dir, f"scenario_{scenario_type}.csv"))

    combined_df = pd.DataFrame(all_rows)
    combined_df.to_csv(osp.join(run_dir, "all_metrics.csv"), index=False)

    print("\n==============================")
    print("SAVED RESULTS")
    print("==============================")
    print(f"Directory: {run_dir}")
    print(f"\nGlobal:")
    print(global_df.to_string())
    for scenario_type, df in scenario_tables.items():
        print(f"\n{scenario_type}:")
        print(df.to_string())

    generate_visualizations(scenario_tables, global_row, run_dir)

    return run_dir


if __name__ == '__main__':
    main()
