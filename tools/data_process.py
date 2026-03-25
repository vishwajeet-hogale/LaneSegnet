import numpy as np
from tqdm import tqdm
from shapely.geometry import LineString
from openlanev2.lanesegment.io import io
import shelve
import json

"""
This script is used to collect the data from the original OpenLane-V2 dataset.
The results will be saved in OpenLane-V2 folder.
The main difference between this script and the original one is that we don't interpolate the points for ped crossing and road bouadary.
"""

def _fix_pts_interpolate(curve, n_points):
    ls = LineString(curve)
    distances = np.linspace(0, ls.length, n_points)
    curve = np.array([ls.interpolate(distance).coords[0] for distance in distances], dtype=np.float32)
    return curve

def collect(root_path : str, data_dict : dict, collection : str, n_points : dict) -> None:

    data_list = [(split, segment_id, timestamp.split('.')[0]) \
        for split, segment_ids in data_dict.items() \
            for segment_id, timestamps in segment_ids.items() \
                for timestamp in timestamps
    ]
    # Use a shelve DB to cache frames on disk to avoid growing memory usage
    db_path = f'{root_path}/{collection}.db'
    with shelve.open(db_path, writeback=False) as db:
        for split, segment_id, timestamp in tqdm(data_list, desc=f'collecting {collection}', ncols=100):
            identifier = (split, segment_id, timestamp)
            # Try loading the '-ls.json' file first; fall back to plain '{timestamp}.json'
            info_path_ls = f'{root_path}/{split}/{segment_id}/info/{timestamp}-ls.json'
            info_path = f'{root_path}/{split}/{segment_id}/info/{timestamp}.json'
            frame = None
            try:
                try:
                    frame = io.json_load(info_path_ls)
                except FileNotFoundError:
                    frame = io.json_load(info_path)
            except FileNotFoundError:
                # If neither info file exists, skip this identifier with a warning
                print(f'Warning: info file not found for {identifier}, skipping')
                continue
            except MemoryError:
                # Skip overly large/malformed JSON that can't be loaded into memory
                print(f'Warning: MemoryError loading info file for {identifier}, skipping')
                continue
            except Exception as e:
                # Catch other JSON parsing errors and continue
                print(f'Warning: error loading info file for {identifier}: {e}, skipping')
                continue

            for k, v in frame.get('pose', {}).items():
                frame['pose'][k] = np.array(v, dtype=np.float64)
            for camera in frame.get('sensor', {}).keys():
                for para in ['intrinsic', 'extrinsic']:
                    for k, v in frame['sensor'][camera][para].items():
                        frame['sensor'][camera][para][k] = np.array(v, dtype=np.float64)

            # If no annotation, store and continue
            if 'annotation' not in frame or frame['annotation'] is None:
                key = f"{split}|||{segment_id}|||{timestamp}"
                db[key] = frame
                continue

            # NOTE: We don't interpolate the points for ped crossing and road bouadary.
            if 'area' in frame['annotation'] and frame['annotation']['area'] is not None:
                for i, area in enumerate(frame['annotation']['area']):
                    frame['annotation']['area'][i]['points'] = np.array(area.get('points', []), dtype=np.float32)

            if 'lane_segment' in frame['annotation'] and frame['annotation']['lane_segment'] is not None:
                for i, lane_segment in enumerate(frame['annotation']['lane_segment']):
                    if 'centerline' in lane_segment:
                        frame['annotation']['lane_segment'][i]['centerline'] = _fix_pts_interpolate(
                            np.array(lane_segment['centerline']), n_points['centerline'])
                    if 'left_laneline' in lane_segment:
                        frame['annotation']['lane_segment'][i]['left_laneline'] = _fix_pts_interpolate(
                            np.array(lane_segment['left_laneline']), n_points['left_laneline'])
                    if 'right_laneline' in lane_segment:
                        frame['annotation']['lane_segment'][i]['right_laneline'] = _fix_pts_interpolate(
                            np.array(lane_segment['right_laneline']), n_points['right_laneline'])

            if 'traffic_element' in frame['annotation'] and frame['annotation']['traffic_element'] is not None:
                for i, traffic_element in enumerate(frame['annotation']['traffic_element']):
                    frame['annotation']['traffic_element'][i]['points'] = np.array(traffic_element.get('points', []), dtype=np.float32)

            if 'topology_lsls' in frame['annotation']:
                frame['annotation']['topology_lsls'] = np.array(frame['annotation']['topology_lsls'], dtype=np.int8)
            if 'topology_lste' in frame['annotation']:
                frame['annotation']['topology_lste'] = np.array(frame['annotation']['topology_lste'], dtype=np.int8)

            # store frame into shelve with a string key
            key = f"{split}|||{segment_id}|||{timestamp}"
            db[key] = frame
    # After writing to shelve, assemble the final dict and dump to pkl
    # Note: this will load all entries into memory briefly when creating the final dict
    # which should be acceptable because shelve avoids holding them during collection.
    meta = {}
    with shelve.open(db_path, flag='r') as db:
        for key in db.keys():
            split, segment_id, timestamp = key.split('|||')
            meta[(split, segment_id, timestamp)] = db[key]
    io.pickle_dump(f'{root_path}/{collection}.pkl', meta)

if __name__ == '__main__':
    root_path = 'D:/TopoNet/data/OpenLane-V2'
    file = f'{root_path}/data_dict_subset_A.json'
    subset = 'data_dict_subset_A'
    # Only collect the 'train' split to avoid processing val/test
    data_dict = io.json_load(file)
    if 'train' not in data_dict:
        raise FileNotFoundError(f"'train' split not found in {file}")
    collect(
        root_path,
        {'train': data_dict['train']},
        f'{subset}_train_lanesegnet',
        n_points={
            'centerline': 10,
            'left_laneline': 10,
            'right_laneline': 10
        },
    )
