import os
import json
import numpy as np

def main():
    with open('radiology_reports.json', 'r') as f:
        reports = json.load(f)

    print(f"Total reports loaded: {len(reports)}")
    for item in reports:
        vol_file = item['volume_file']
        vol_path = os.path.join('volumes', vol_file)
        if not os.path.exists(vol_path):
            print(f"MISSING FILE: {vol_path}")
            continue
        data = np.fromfile(vol_path, dtype=np.float32)
        expected_shape = tuple(item['volume_dimensions'])
        expected_size = int(np.prod(expected_shape))
        match = len(data) == expected_size
        reshaped = data.reshape(expected_shape) if match else None
        print(f"Case: {item['case_id']}")
        print(f"  Pathology: {item['pathology']}")
        print(f"  File: {vol_file} ({len(data)} floats, expected {expected_size}) -> Matches: {match}")
        if reshaped is not None:
            print(f"  Shape: {reshaped.shape}, Min: {reshaped.min():.4f}, Max: {reshaped.max():.4f}, Mean: {reshaped.mean():.4f}, Std: {reshaped.std():.4f}")
            print(f"  NaNs: {np.isnan(reshaped).any()}, Infs: {np.isinf(reshaped).any()}")
        print(f"  Report: {item['clinical_radiology_report']}")
        print("-" * 60)

if __name__ == '__main__':
    main()
