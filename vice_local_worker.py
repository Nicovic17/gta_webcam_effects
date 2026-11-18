"""Isolated Python 3.10 worker. Images travel through RAM pipes, never the network."""
import argparse
import base64
import json
import os
from pathlib import Path
import sys
import time
import traceback

parser = argparse.ArgumentParser()
parser.add_argument('--runtime-root', type=Path, required=True)
parser.add_argument('--width', type=int, default=512)
parser.add_argument('--height', type=int, default=320)
args = parser.parse_args()
protocol = sys.stdout.buffer
sys.stdout = sys.stderr  # Third-party progress/logs cannot corrupt the protocol.
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ.pop('OPENAI_API_KEY', None)


def send(message):
    protocol.write(json.dumps(message).encode('ascii')+b'\n')
    protocol.flush()


try:
    import cv2
    import numpy as np
    from vice_local_engine import StreamEngine
    from vice_local_settings import DEFAULT_PROMPT
    cv2.setNumThreads(2)
    engine = StreamEngine(args.runtime_root, width=args.width, height=args.height)
    for _ in range(5):
        engine.infer(np.zeros((args.height, args.width, 3), np.uint8))
    send({'type': 'ready', **engine.info})
    for line in sys.stdin.buffer:
        if len(line) > 12*1024*1024:
            raise RuntimeError('Frame locale troppo grande.')
        packet = json.loads(line)
        if packet.get('type') == 'stop':
            break
        image = cv2.imdecode(np.frombuffer(base64.b64decode(packet['image'], validate=True), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError('Frame locale non valido.')
        started = time.perf_counter()
        result = engine.infer(image, int(packet.get('strength', 2)),
                              prompt=packet.get('prompt', DEFAULT_PROMPT))
        elapsed_ms = (time.perf_counter()-started)*1000
        ok, jpeg = cv2.imencode('.jpg', result, [cv2.IMWRITE_JPEG_QUALITY, 94])
        if not ok:
            raise RuntimeError('Output locale non valido.')
        send(dict(type='frame', generation=packet['generation'], sequence=packet['sequence'],
                  elapsed_ms=elapsed_ms, image=base64.b64encode(jpeg).decode('ascii')))
except Exception:
    traceback.print_exc(file=sys.stderr)
    send({'type': 'error', 'message': 'Worker GPU non disponibile. Controlla .vice-local/worker.log.'})
    sys.exit(1)
