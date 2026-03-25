# Minimal OpenLane-V2 folder dataset loader
# Loads per-sample JSON files from a directory structure like:
# data/OpenLane-V2/<split>/<segment_id>/info/<timestamp>.json

import os
from glob import glob

import numpy as np
import mmcv
from mmcv.parallel import DataContainer as DC
from mmdet.datasets import DATASETS
from mmdet3d.datasets import Custom3DDataset

from .openlanev2_subset_A_lanesegnet_dataset import OpenLaneV2_subset_A_LaneSegNet_Dataset
from openlanev2.lanesegment.io import io
from mmdet3d.datasets.pipelines import Compose

@DATASETS.register_module()
class OpenLaneV2FolderDataset(Custom3DDataset):
    """
    Dataset loader that reads raw OpenLane-V2 JSON files directly from a folder
    layout instead of expecting a prebuilt .pkl annotation file.

    Usage example in config:
        dataset_type = 'OpenLaneV2FolderDataset'
        data_root = 'data/OpenLane-V2'
        ann_file can be ignored (pass dummy)
        set split via cfg or pass `split='train'` when constructing
    """

    CAMS = OpenLaneV2_subset_A_LaneSegNet_Dataset.CAMS

    def __init__(self, data_root, ann_file=None, queue_length=1,
                 filter_empty_te=False, filter_map_change=False,
                 points_num=10, split='train', **kwargs):
        self.split = split
        self.queue_length = queue_length
        self.filter_empty_te = filter_empty_te
        self.filter_map_change = filter_map_change
        self.points_num = points_num
        # Avoid calling parent __init__ because it expects an ann_file path
        # (we load annotations directly from the folder). Instead set minimal
        # attributes expected by the mmdet3d dataset APIs and then load infos.
        self.data_root = data_root
        self.ann_file = ann_file
        # copy through common attrs from kwargs when present
        # build pipeline if provided as list of transforms
        pipeline_cfg = kwargs.get('pipeline', None)
        if pipeline_cfg is not None and isinstance(pipeline_cfg, (list, tuple)):
            try:
                self.pipeline = Compose(pipeline_cfg)
            except Exception:
                self.pipeline = pipeline_cfg
        else:
            self.pipeline = pipeline_cfg
        self.test_mode = kwargs.get('test_mode', False)
        self.modality = kwargs.get('modality', {'use_camera': True})
        # Set attributes expected by Custom3DDataset pipelines
        self.box_type_3d = kwargs.get('box_type_3d', None)
        self.box_mode_3d = kwargs.get('box_mode_3d', None)
        self.filter_empty_gt = kwargs.get('filter_empty_gt', False)
        self.filter_empty_te = kwargs.get('filter_empty_te', False)
        # try to inherit CLASSES from the related dataset when available
        try:
            self.CLASSES = getattr(OpenLaneV2_subset_A_LaneSegNet_Dataset, 'CLASSES', None)
        except Exception:
            self.CLASSES = None

        # finally load annotations using our folder scanner
        self.data_infos = self.load_annotations(ann_file)

    def load_annotations(self, ann_file):
        """Walk the folder tree and load all JSON info files for the requested split.
        Returns a list of info dicts identical in structure to those produced by
        the project's data preprocessing scripts.
        """
        split_dir = os.path.join(self.data_root, self.split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"Split directory not found: {split_dir}")
        # Lazy-load strategy: only list JSON files and store lightweight entries
        info_pattern = os.path.join(split_dir, '*', 'info', '*.json')
        info_files = sorted(glob(info_pattern))
        data_infos = []
        for fpath in info_files:
            # quick parse of filename to get timestamp and segment id
            fname = os.path.splitext(os.path.basename(fpath))[0]
            try:
                timestamp = int(fname)
            except Exception:
                timestamp = fname
            seg = os.path.basename(os.path.dirname(os.path.dirname(fpath)))
            data_infos.append({'info_path': fpath, 'timestamp': timestamp, 'segment_id': seg})
        # info cache for parsed JSONs (lazy)
        self._info_cache = {}
        return data_infos

    def get_data_info(self, index):
        raw = self.data_infos[index]
        # if this entry is a full dict (backwards compat), use it directly
        if 'info_path' in raw:
            info = self._load_info(index)
        else:
            info = raw
        input_dict = dict(sample_idx=info['timestamp'], scene_token=info.get('segment_id'))

        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            for cam_name, cam_info in info.get('sensor', {}).items():
                image_path = cam_info.get('image_path')
                if image_path is None:
                    # try to construct relative path
                    img_fname = cam_info.get('image_path', '')
                    image_path = os.path.join(self.data_root, img_fname)
                else:
                    image_path = os.path.join(self.data_root, image_path)
                image_paths.append(image_path)

                lidar2cam_r = np.linalg.inv(cam_info['extrinsic']['rotation'])
                lidar2cam_t = cam_info['extrinsic']['translation'] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t

                intrinsic = np.array(cam_info['intrinsic']['K'])
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)

                lidar2img_rts.append(lidar2img_rt)
                cam_intrinsics.append(viewpad)
                lidar2cam_rts.append(lidar2cam_rt.T)

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    cam_intrinsic=cam_intrinsics,
                    lidar2cam=lidar2cam_rts,
                ))

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        # minimal can_bus and pose info to match pipelines
        can_bus = np.zeros(18)
        pose = info.get('pose', {})
        if 'translation' in pose:
            can_bus[:3] = pose['translation']
        input_dict['can_bus'] = can_bus
        input_dict['lidar2global_rotation'] = np.array(pose.get('rotation', np.eye(3)))

        return input_dict

    def get_ann_info(self, index):
        raw = self.data_infos[index]
        if 'info_path' in raw:
            info = self._load_info(index)
        else:
            info = raw
        # Return annotation dict in same layout as other dataset class expects
        ann_info = info.get('annotation', {})
        return ann_info

    def _load_info(self, index):
        """Load and cache the full JSON info for the given index."""
        entry = self.data_infos[index]
        fpath = entry.get('info_path')
        if fpath in getattr(self, '_info_cache', {}):
            return self._info_cache[fpath]
        try:
            info = io.json_load(fpath)
        except Exception:
            info = mmcv.load(fpath)
        # ensure basic fields exist
        if 'timestamp' not in info:
            info['timestamp'] = entry.get('timestamp')
        if 'segment_id' not in info:
            info['segment_id'] = entry.get('segment_id')
        # cache and return
        self._info_cache[fpath] = info
        return info
