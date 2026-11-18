"""One explicit webcam edit per click, sent directly to the OpenAI Images API.

No frame/key is written to disk, logged, retried or sent to other hosts.
"""
import base64
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import socket
from threading import Lock, Thread
import urllib.error
import urllib.request
import uuid

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
EDIT_URL = 'https://api.openai.com/v1/images/edits'
PROMPT = '''Edit this webcam photograph into premium Grand Theft Auto loading-screen
artwork: confident ink contours, sophisticated painterly cel shading, carefully
rendered skin and hair, rich cinematic lighting, vibrant Vice City pink and teal
accents. Make the person look like a polished, recognisable illustrated version
of themselves. Preserve their exact facial identity, age, expression, hairstyle,
beard, clothing, pose, hands, body proportions and camera angle. Keep every person
in precisely the same position and size in the frame, with the same head and body
outline, so the edited photograph can dissolve cleanly over the original. Keep
the same scene layout, illustrating the surroundings with tropical Miami mood
and art-deco colour where appropriate. Do not add people, accessories, weapons,
logos, borders, lettering or watermarks. Deliver a detailed full-bleed landscape
illustration at the requested aspect ratio, not a collage or a generic avatar.'''


class ViceAPIError(RuntimeError):
    pass


@dataclass(frozen=True)
class ImageConfig:
    api_key: str = field(repr=False)
    model: str = 'gpt-image-2'
    quality: str = 'high'
    size: str = '1536x864'
    timeout: float = 240

    @classmethod
    def load(cls, folder=BASE_DIR, environ=None):
        # Read documented settings as data, never execute dotenv content.
        values = {}
        path = Path(folder)/'.env'
        if path.is_file():
            for line in path.read_text(encoding='utf-8-sig').splitlines():
                line = line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                name, value = line.split('=', 1)
                values[name.strip()] = value.strip().strip('\"\'')
        env = os.environ if environ is None else environ
        def get(name, default=''):
            return env.get(name, values.get(name, default)).strip()
        key = get('OPENAI_API_KEY')
        if not key:
            raise ViceAPIError('Imposta OPENAI_API_KEY oppure inserisci la chiave nella finestra iniziale.')
        quality = get('OPENAI_IMAGE_QUALITY', 'high')
        if quality not in ('low', 'medium', 'high', 'auto'):
            raise ViceAPIError('OPENAI_IMAGE_QUALITY deve essere low, medium, high oppure auto.')
        model = get('OPENAI_IMAGE_MODEL', 'gpt-image-2')
        if not model or '\n' in key or '\r' in key:
            raise ViceAPIError('Configurazione OpenAI non valida.')
        return cls(key, model=model, quality=quality)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class OpenAIImageEditor:
    def __init__(self, config, opener=None):
        self.config = config
        self.opener = opener or urllib.request.build_opener(NoRedirect())

    def edit(self, frame):
        ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 94])
        if not ok:
            raise ViceAPIError('Impossibile preparare la foto della webcam.')
        boundary = 'vice-'+uuid.uuid4().hex
        parts = []
        fields = dict(model=self.config.model, prompt=PROMPT, n='1',
                      quality=self.config.quality, size=self.config.size, output_format='png')
        for name, value in fields.items():
            parts.append((f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n'
                          f'{value}\r\n').encode('utf-8'))
        parts.append((f'--{boundary}\r\nContent-Disposition: form-data; name="image[]"; '
                      'filename="webcam.jpg"\r\nContent-Type: image/jpeg\r\n\r\n').encode())
        parts.extend((encoded.tobytes(), f'\r\n--{boundary}--\r\n'.encode()))
        request = urllib.request.Request(EDIT_URL, data=b''.join(parts), method='POST', headers={
            'Authorization': 'Bearer '+self.config.api_key,
            'Content-Type': 'multipart/form-data; boundary='+boundary,
            'Accept': 'application/json'})
        try:
            with self.opener.open(request, timeout=self.config.timeout) as response:
                payload = response.read(40*1024*1024+1)
        except urllib.error.HTTPError as error:
            # Never surface raw server messages, request headers or credentials.
            messages = {401: 'Chiave OpenAI non valida.', 403: 'Accesso al modello immagini non abilitato.',
                        429: 'Quota o limite OpenAI raggiunto. Verifica credito e limiti API.',
                        400: 'Richiesta immagini rifiutata. Verifica modello, permessi e contenuto della foto.'}
            code = error.code
            error.close()
            raise ViceAPIError(messages.get(code, f'Errore OpenAI ({code}). Riprova con un nuovo scatto.')) from None
        except (TimeoutError, socket.timeout):
            raise ViceAPIError('La generazione ha superato il tempo di attesa. Puoi riprovare.') from None
        except (urllib.error.URLError, OSError):
            raise ViceAPIError('Connessione a OpenAI non riuscita. Controlla la rete e riprova.') from None
        if len(payload) > 40*1024*1024:
            raise ViceAPIError('Risposta immagine troppo grande.')
        try:
            data = json.loads(payload)
            image = base64.b64decode(data['data'][0]['b64_json'], validate=True)
            decoded = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
            if decoded is None or decoded.size == 0:
                raise ValueError('empty image')
        except (KeyError, IndexError, TypeError, ValueError):
            raise ViceAPIError('OpenAI non ha restituito un\'immagine valida.') from None
        return decoded


@dataclass(frozen=True)
class EditResult:
    image: np.ndarray | None = None
    error: str = ''


class ImageEditJob:
    """Single daemon worker. Cancelled/closed generations can never reappear."""
    def __init__(self, editor):
        self.editor = editor
        self.lock = Lock()
        self.busy = False
        self.closed = False
        self.generation = 0
        self.result = None

    def start(self, frame):
        with self.lock:
            if self.busy or self.closed:
                return False
            self.busy = True
            self.generation += 1
            generation = self.generation
            self.result = None
        try:
            Thread(target=self._run, args=(frame.copy(), generation), daemon=True,
                   name='ViceOpenAI').start()
        except Exception:
            with self.lock:
                self.busy = False
            raise
        return True

    def _run(self, frame, generation):
        try:
            result = EditResult(image=self.editor.edit(frame))
        except ViceAPIError as error:
            result = EditResult(error=str(error))
        except Exception:
            result = EditResult(error='Generazione non riuscita. Riprova con un nuovo scatto.')
        with self.lock:
            if not self.closed and generation == self.generation:
                self.result = result
            self.busy = False

    def take_result(self):
        with self.lock:
            result, self.result = self.result, None
            return result

    def cancel(self):
        with self.lock:
            self.generation += 1
            self.result = None

    def close(self):
        with self.lock:
            self.closed = True
            self.generation += 1
            self.result = None
