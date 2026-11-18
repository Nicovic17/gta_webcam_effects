"""Bounded latest-frame pipe to an isolated local GPU process."""
import base64
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
from threading import Condition, Thread
import time

import cv2
import numpy as np
from vice_local_settings import DEFAULT_PROMPT, normalize_prompt

BASE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class LocalResult:
    image: np.ndarray
    generation: int
    sequence: int
    elapsed_ms: float
    inference_ms: float


class LocalStreamClient:
    def __init__(self, folder=BASE_DIR, runtime_root=None):
        folder = Path(folder)
        self.root = Path(runtime_root) if runtime_root else folder/'.vice-local'
        python = self.root/'venv'/'Scripts'/'python.exe'
        worker = folder/'vice_local_worker.py'
        if not python.is_file() or not worker.is_file():
            raise RuntimeError('Prima esegui SETUP_VICE_LOCALE.ps1 nella cartella gta_webcam_effects.')
        self.condition = Condition()
        self.pending = self.result = None
        self.closed = self.ready = False
        self.generation = self.sequence = 0
        self.error = ''
        self.info = {}
        env = os.environ.copy()
        env.pop('OPENAI_API_KEY', None)
        env.update(HF_HUB_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1',
                   HF_HUB_DISABLE_IMPLICIT_TOKEN='1', PYTHONUNBUFFERED='1',
                   HF_HOME=str(self.root/'hf-home'))
        self.log = (self.root/'worker.log').open('wb')
        try:
            self.process = subprocess.Popen([str(python), '-B', str(worker),
                '--runtime-root', str(self.root), '--width', '512', '--height', '320'],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
                env=env, cwd=str(folder), creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        except Exception:
            self.log.close()
            raise
        self.thread = Thread(target=self._run, daemon=True, name='ViceLocalPipe')
        self.thread.start()

    def _read(self):
        line = self.process.stdout.readline(12*1024*1024+1)
        if not line or len(line) > 12*1024*1024:
            raise RuntimeError('Worker locale terminato o risposta non valida.')
        message = json.loads(line)
        if message.get('type') == 'error':
            raise RuntimeError(message.get('message', 'Errore del worker GPU.'))
        return message

    def _run(self):
        try:
            ready = self._read()
            if ready.get('type') != 'ready':
                raise RuntimeError('Avvio worker non valido.')
            with self.condition:
                self.ready, self.info = True, ready
            while True:
                with self.condition:
                    self.condition.wait_for(lambda: self.closed or self.pending is not None)
                    if self.closed:
                        return
                    frame, generation, sequence, strength, prompt, captured = self.pending
                    self.pending = None
                # Resize/JPEG happens off the display thread. Exactly one frame
                # is in flight and one replaceable pending frame is retained.
                ratio = min(1., 640/frame.shape[1])
                small = cv2.resize(frame, (max(1, round(frame.shape[1]*ratio)), max(1, round(frame.shape[0]*ratio))),
                                   interpolation=cv2.INTER_AREA)
                ok, jpeg = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, 94])
                if not ok:
                    raise RuntimeError('Frame non valido.')
                packet = dict(type='frame', generation=generation, sequence=sequence,
                              strength=strength, prompt=prompt,
                              image=base64.b64encode(jpeg).decode('ascii'))
                self.process.stdin.write(json.dumps(packet).encode('ascii')+b'\n')
                self.process.stdin.flush()
                message = self._read()
                decoded = cv2.imdecode(np.frombuffer(base64.b64decode(message['image'], validate=True), np.uint8), cv2.IMREAD_COLOR)
                if decoded is None or message.get('type') != 'frame':
                    raise RuntimeError('Immagine dal worker non valida.')
                result = LocalResult(decoded, message['generation'], message['sequence'],
                                     (time.monotonic()-captured)*1000, message['elapsed_ms'])
                with self.condition:
                    if not self.closed and result.generation == self.generation:
                        self.result = result
        except Exception:
            with self.condition:
                if not self.closed:
                    self.error = 'Errore GPU locale. Esegui il setup o controlla .vice-local/worker.log.'

    def submit(self, frame, generation, strength=2, prompt=DEFAULT_PROMPT):
        prompt = normalize_prompt(prompt)
        with self.condition:
            if self.closed or self.error:
                return
            self.sequence += 1
            self.pending = (frame.copy(), generation, self.sequence, strength, prompt, time.monotonic())
            self.condition.notify()

    def clear(self, generation):
        with self.condition:
            self.generation = generation
            self.pending = self.result = None

    def latest(self):
        with self.condition:
            return self.result

    def close(self):
        with self.condition:
            if self.closed:
                return
            self.closed = True
            self.pending = self.result = None
            self.condition.notify_all()
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=3)
        self.thread.join(timeout=2)
        for stream in (self.process.stdin, self.process.stdout):
            try:
                stream.close()
            except OSError:
                pass
        self.log.close()
