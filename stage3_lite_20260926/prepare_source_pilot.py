"""Prepare an auditable, grouped source-pretraining pilot, without test metrics."""
import collections
import hashlib
import io
import json
import os
from pathlib import Path
import random
import re
import tarfile
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / 'data/Pseudo_label_training_data'
OUT = ROOT / 'source_pilot'
SEED, TRAIN_COUNT, VAL_COUNT = 42, 12000, 600
REMOTE = '/root/autodl-tmp/work/stage3_lite_20260926/source_pilot_v1'

def digest(data):
    return hashlib.sha256(data).hexdigest()

def index(path, suffixes):
    result = {}
    with os.scandir(path) as entries:
        for e in entries:
            if e.is_file() and Path(e.name).suffix.lower() in suffixes:
                key = Path(e.name).stem.lower()
                if key in result:
                    raise ValueError('Duplicate case-insensitive stem: ' + key)
                result[key] = Path(e.path)
    return result

def group(stem):
    if not re.search(r'_x\d+_y\d+_', stem):
        raise ValueError('Unrecognized crop group, manual review needed: ' + stem)
    return re.sub(r'_x\d+_y\d+_', '_', stem)

def pixels(data):
    with Image.open(io.BytesIO(data)) as image:
        rgb = image.convert('RGB')
        return rgb.size, digest(rgb.tobytes())

def main():
    started = time.time()
    OUT.mkdir(exist_ok=True)
    if (OUT / 'prepared.json').exists():
        print('Pilot already prepared; not resampling', flush=True)
        return
    images = index(DATA / 'Panicle_data_2025_early', {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'})
    masks = index(DATA / 'mask', {'.png', '.bmp', '.tif', '.tiff'})
    paired = sorted(images.keys() & masks.keys())
    groups = sorted({group(s) for s in paired})
    rng = random.Random(SEED)
    rng.shuffle(groups)
    val_groups = set(groups[:max(1, round(len(groups)*.05))])
    pools = {'train': [s for s in paired if group(s) not in val_groups],
             'val': [s for s in paired if group(s) in val_groups]}
    selected = {k: random.Random(SEED).sample(v, n) for k, v, n in (
        ('train', pools['train'], TRAIN_COUNT), ('val', pools['val'], VAL_COUNT))}
    assert not ({group(s) for s in selected['train']} & {group(s) for s in selected['val']})
    # File/pixel duplicate checks use input RGB only. No labels, test predictions,
    # or test metrics are consulted for selection or development.
    blocked_files, blocked_pixels = set(), set()
    rice_roots = [ROOT.parent / 'data/RiceSEG', ROOT.parent / '_records/server_work/public_backup_20260916/cea_public_20260915/data/RiceSEG/images']
    for root in rice_roots:
        for directory, _, files in os.walk(root):
            for name in files:
                if Path(name).suffix.lower() in {'.jpg', '.jpeg'}:
                    blob = (Path(directory) / name).read_bytes()
                    blocked_files.add(digest(blob))
                    blocked_pixels.add(pixels(blob))
    plan = {'seed': SEED, 'paired_images': len(paired), 'all_images': len(images), 'all_masks': len(masks),
            'unmatched_masks': len(masks)-len(paired), 'source_groups': len(groups),
            'split': '5% of sorted original-photo groups shuffled with random.Random(42); fixed random sample inside disjoint pools',
            'counts': {k: len(v) for k, v in selected.items()}, 'available_pools': {k: len(v) for k, v in pools.items()},
            'validation_groups': sorted(val_groups), 'rice_rgb_hashes_checked': len(blocked_pixels),
            'limitations': 'Exact RGB duplicate screening is not a proof against all near-duplicates. Source original-photo IDs/year suffixes differ from RiceSEG. This is a fixed small pilot, not all-source pretraining.'}
    (OUT / 'selection.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding='utf-8')
    train_pixels, val_pixels = set(), set()
    records, manifests, mask_values = [], {}, collections.Counter()
    archive = OUT / 'source_pilot_v1.tar'
    with tarfile.open(archive, 'w') as tar:
        def put(name, data):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mtime = 0
            tar.addfile(info, io.BytesIO(data))
        for split, stems in selected.items():
            manifests[split] = []
            for number, stem in enumerate(stems):
                image_blob, mask_blob = images[stem].read_bytes(), masks[stem].read_bytes()
                image_sha = digest(image_blob)
                size, pixel_sha = pixels(image_blob)
                if image_sha in blocked_files or (size, pixel_sha) in blocked_pixels:
                    raise ValueError('Source/RiceSEG exact RGB overlap: ' + stem)
                if size != (512, 512):
                    raise ValueError('Pilot expects original 512 crops: ' + stem)
                (train_pixels if split == 'train' else val_pixels).add(pixel_sha)
                with Image.open(io.BytesIO(mask_blob)) as mask_image:
                    arr = np.asarray(mask_image)
                if arr.shape[:2] != (512, 512):
                    raise ValueError('Image/mask dimensions differ: ' + stem)
                values = np.unique(arr).tolist()
                if any(x not in (0, 1, 255) for x in values):
                    raise ValueError('Unexpected pseudo label values: ' + str(values))
                mask_values[str(values)] += 1
                if arr.ndim == 3:
                    arr = arr[..., :3].max(2)
                normalized = (arr > 0).astype(np.uint8)
                stream = io.BytesIO()
                Image.fromarray(normalized).save(stream, format='PNG')
                normalized_blob = stream.getvalue()
                image_name, mask_name = 'images/' + stem + images[stem].suffix.lower(), 'masks/' + stem + '.png'
                put(image_name, image_blob)
                put(mask_name, normalized_blob)
                manifests[split].append({'image': image_name, 'mask': mask_name})
                records.append({'stem': stem, 'group': group(stem), 'split': split,
                                'source_image': str(images[stem]), 'source_mask': str(masks[stem]),
                                'image': image_name, 'mask': mask_name, 'image_sha256': image_sha,
                                'image_rgb_sha256': pixel_sha, 'original_mask_sha256': digest(mask_blob),
                                'normalized_mask_sha256': digest(normalized_blob),
                                'foreground_fraction': float(normalized.mean())})
                if (number + 1) % 1000 == 0:
                    print(json.dumps({'split': split, 'done': number+1, 'seconds': time.time()-started}), flush=True)
        if train_pixels & val_pixels:
            raise ValueError('Pixel duplicate across source train/validation')
        for split, rows in manifests.items():
            blob = json.dumps(rows, indent=2).encode()
            put(split + '.json', blob)
            (OUT / (split + '.json')).write_bytes(blob)
        put('provenance.json', json.dumps(records, ensure_ascii=False).encode())
        put('selection.json', json.dumps(plan, ensure_ascii=False, indent=2).encode())
    archive_hash = hashlib.sha256()
    with archive.open('rb') as inp:
        for block in iter(lambda: inp.read(8*1024**2), b''):
            archive_hash.update(block)
    result = {**plan, 'archive': str(archive), 'bytes': archive.stat().st_size,
              'sha256': archive_hash.hexdigest(), 'mask_value_counts': dict(mask_values),
              'mask_transform': 'Original Stage3 convention: any nonzero value is foreground; stored as 0/1 PNG.',
              'exact_rice_overlap': 0, 'exact_source_train_val_overlap': 0, 'seconds': time.time()-started}
    (OUT / 'provenance.json').write_text(json.dumps(records, ensure_ascii=False), encoding='utf-8')
    (OUT / 'prepared.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'validation_groups'}, ensure_ascii=False), flush=True)

if __name__ == '__main__':
    main()
