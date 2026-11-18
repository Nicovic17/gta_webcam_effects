"""Download/warm up and benchmark local GPU; uses a repository meme, not webcam."""
import argparse
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--runtime-root', type=Path, required=True)
parser.add_argument('--input', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--download', action='store_true')
args = parser.parse_args()
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
os.environ['HF_HOME'] = str(args.runtime_root/'hf-home')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ.pop('OPENAI_API_KEY', None)
from vice_local_engine import StreamEngine

frame = cv2.imread(str(args.input))
if frame is None:
    raise RuntimeError('Input di benchmark non trovato.')
cv2.setNumThreads(2)
measurements, previews = [], []
for width, height in ((512, 512), (512, 320)):
    print(f'Loading {width}x{height}', flush=True)
    engine = StreamEngine(args.runtime_root, width=width, height=height, download=args.download)
    for _ in range(5):
        engine.infer(frame)
    elapsed = []
    for index in range(30):
        image = np.roll(frame, index%6, axis=1)
        start = time.perf_counter()
        result = engine.infer(image)
        elapsed.append((time.perf_counter()-start)*1000)
    report = {**engine.info, 'median_ms': round(float(np.median(elapsed)), 2),
              'p95_ms': round(float(np.percentile(elapsed, 95)), 2),
              'mean_fps': round(1000/float(np.mean(elapsed)), 2),
              'peak_vram_mb': engine.memory_mb()}
    measurements.append(report)
    print(json.dumps(report), flush=True)
    if height == 320:
        reference = cv2.resize(frame, (512, round(frame.shape[0]*512/frame.shape[1])))
        previews.append(reference)
        for strength in (1, 2, 3):
            for _ in range(3):
                result = engine.infer(frame, strength)
            previews.append(cv2.resize(result, (reference.shape[1], reference.shape[0])))
    del engine
    import gc, torch
    gc.collect()
    torch.cuda.empty_cache()
args.output.mkdir(parents=True, exist_ok=True)
(args.output/'benchmark.json').write_text(json.dumps(measurements, indent=2), encoding='utf-8')
cv2.imwrite(str(args.output/'style_comparison.jpg'), np.concatenate(previews, axis=1))
print('Benchmark saved.', flush=True)
