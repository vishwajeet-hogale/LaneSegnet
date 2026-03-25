import os
import traceback
import argparse
import traceback
import argparse

from projects.lanesegnet.datasets.openlanev2_folder_dataset import OpenLaneV2FolderDataset


def main(max_samples: int):
    data_root = 'D:/TopoNet/data/OpenLane-V2'
    if not os.path.isdir(data_root):
        print(f'Data root not found: {data_root}')
        return

    try:
        # Create instance without calling parent __init__
        ds = object.__new__(OpenLaneV2FolderDataset)
        ds.data_root = data_root
        ds.split = 'train'
        ds.modality = {'use_camera': True, 'use_lidar': False}
        ds.test_mode = True

        print('Listing annotation files...')
        pattern = os.path.join(data_root, ds.split, '*', 'info', '*.json')
        files = sorted([p for p in __import__('glob').glob(pattern)])
        total = len(files)
        print(f'Found {total} info files')
        if total == 0:
            return

        # limit samples
        if max_samples is not None and max_samples > 0:
            files = files[:max_samples]

        infos = []
        for fpath in files:
            try:
                info = __import__('openlanev2.lanesegment.io.io', fromlist=['io']).io.json_load(fpath)
            except Exception:
                try:
                    info = __import__('mmcv').load(fpath)
                except Exception:
                    print(f'Failed to load {fpath}, skipping')
                    continue
            if 'timestamp' not in info:
                info['timestamp'] = int(os.path.splitext(os.path.basename(fpath))[0])
            if 'segment_id' not in info:
                info['segment_id'] = os.path.basename(os.path.dirname(os.path.dirname(fpath)))
            infos.append(info)

        # attach data_infos and run get_data_info for selected samples
        ds.data_infos = infos
        for idx in range(len(infos)):
            print(f'--- Sample {idx} ---')
            info = ds.data_infos[idx]
            print('timestamp:', info.get('timestamp'))
            print('segment_id:', info.get('segment_id'))
            sensor = info.get('sensor', {})
            print('num cams in sensor:', len(sensor))
            try:
                g = ds.get_data_info(idx)
                print('get_data_info keys:', list(g.keys()))
                img_fns = g.get('img_filename', [])
                if img_fns:
                    print('first image path exists:', os.path.exists(img_fns[0]), img_fns[0])
            except Exception:
                print('Error in get_data_info:')
                traceback.print_exc()

    except Exception:
        traceback.print_exc()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-samples', type=int, default=5, help='Maximum number of samples to load')
    args = parser.parse_args()
    main(args.max_samples)
