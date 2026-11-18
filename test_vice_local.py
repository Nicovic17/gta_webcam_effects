"""Local pipeline scheduling and video controls; no GPU, network or camera needed."""
import base64
from contextlib import ExitStack
import io
from itertools import chain, count, repeat
import json
from pathlib import Path
from queue import Queue
import tempfile
from threading import Event
import time
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from test_support import source_frame
import vice_local_client as pipe
import webcam_vice_local as app
from vice_local_engine import StreamEngine
from vice_local_settings import DEFAULT_PROMPT, normalize_prompt
from vice_prompt_panel import PanelUpdate, PromptDraft


def eventually(check):
    deadline = time.monotonic()+2
    while time.monotonic() < deadline:
        if check():
            return
        time.sleep(.005)
    raise AssertionError('Timed out waiting for local pipe')


class FakeProcess:
    """One blocked inference exposes pending-frame replacement/cancellation."""
    def __init__(self, *, invalid=False):
        self.lines = Queue()
        self.lines.put(b'{"type":"ready","gpu":"test"}\n')
        self.packets = []
        self.entered, self.release = Event(), Event()
        self.invalid, self.dead = invalid, False
        self.stdin = io.BytesIO()
        self.stdin.flush = self.flush
        self.stdout = Mock()
        self.stdout.readline.side_effect = self.readline

    def flush(self):
        self.stdin.seek(0)
        packet = json.loads(self.stdin.read())
        self.stdin.seek(0)
        self.stdin.truncate()
        self.packets.append(packet)
        self.lines.put(packet)

    def readline(self, limit):
        packet = self.lines.get(timeout=2)
        if isinstance(packet, bytes):
            return packet
        if len(self.packets) == 1:
            self.entered.set()
            self.release.wait(2)
        if self.dead:
            return b''
        if self.invalid:
            return b'not-json\n'
        return json.dumps({**packet, 'elapsed_ms': 50}).encode()+b'\n'

    def poll(self):
        return 0 if self.dead else None

    def terminate(self):
        self.dead = True
        self.release.set()
        self.lines.put(b'')

    def wait(self, timeout):
        return 0


class LocalPipeTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        root = Path(self.folder.name)
        (root/'.vice-local/venv/Scripts').mkdir(parents=True)
        (root/'.vice-local/venv/Scripts/python.exe').touch()
        (root/'vice_local_worker.py').touch()
        self.root = root

    def start(self, process):
        with patch.object(pipe.subprocess, 'Popen', return_value=process) as spawn, \
             patch.dict(pipe.os.environ, {'OPENAI_API_KEY': 'must-not-enter-worker'}):
            client = pipe.LocalStreamClient(folder=self.root)
        self.addCleanup(client.close)
        eventually(lambda: client.ready)
        self.assertNotIn('OPENAI_API_KEY', spawn.call_args.kwargs['env'])
        self.assertEqual(spawn.call_args.kwargs['env']['HF_HUB_OFFLINE'], '1')
        return client

    def test_keeps_only_newest_pending_frame_and_copies_camera_buffer(self):
        process = FakeProcess()
        client = self.start(process)
        frame = source_frame()
        client.submit(frame, 0)
        self.assertTrue(process.entered.wait(1))
        for _ in range(20):
            client.submit(frame, 0, 3, 'A cyberpunk portrait with neon lights')
        frame[:] = 0
        process.release.set()
        eventually(lambda: client.latest() is not None and client.latest().sequence == 21)
        self.assertEqual([p['sequence'] for p in process.packets], [1, 21])
        self.assertEqual(process.packets[-1]['strength'], 3)
        self.assertEqual(process.packets[-1]['prompt'], 'A cyberpunk portrait with neon lights')
        self.assertGreater(client.latest().image.mean(), 0)
        client.close()
        client.close()
        self.assertTrue(process.dead)
        self.assertFalse(client.thread.is_alive())

    def test_reset_discards_inflight_generation_and_next_frame_can_complete(self):
        process = FakeProcess()
        client = self.start(process)
        client.submit(source_frame(), 0)
        self.assertTrue(process.entered.wait(1))
        client.clear(1)
        process.release.set()
        eventually(lambda: process.lines.empty())
        # Releasing an old inference must never restore the cancelled picture.
        self.assertIsNone(client.latest())
        client.submit(source_frame(), 1, prompt='ritratto a fumetti, città futuristica')
        eventually(lambda: client.latest() is not None)
        self.assertEqual(client.latest().generation, 1)
        self.assertEqual(process.packets[-1]['prompt'], 'ritratto a fumetti, città futuristica')

    def test_worker_failure_is_reported_and_shutdown_releases_pipe(self):
        process = FakeProcess(invalid=True)
        client = self.start(process)
        client.submit(source_frame(), 0)
        process.release.set()
        eventually(lambda: bool(client.error))
        self.assertIsNone(client.latest())
        client.close()
        self.assertFalse(client.thread.is_alive())
        self.assertTrue(client.log.closed)


class LocalRuntimeTests(unittest.TestCase):
    def run_video(self, keys, *, ready=True, open_error=False, send_error=False, updates=()):
        capture, virtual, client, panel = Mock(), Mock(), Mock(), Mock()
        panel.poll.side_effect = chain(updates, repeat(PanelUpdate()))
        frame = source_frame()
        artwork = np.full_like(frame, 215)
        client.error, client.ready = '', ready
        client.latest.return_value = None
        capture.read.return_value = (True, frame.copy())
        sent, state = [], {'generation': 0, 'sequence': 0}

        def clear(generation):
            state['generation'] = generation
            client.latest.return_value = None

        def submit(image, generation, strength, prompt):
            state['sequence'] += 1
            if ready:
                client.latest.return_value = pipe.LocalResult(artwork, generation, state['sequence'], 75, 50)

        def send(image):
            if send_error:
                raise RuntimeError('send failed')
            sent.append(image.copy())

        client.clear.side_effect, client.submit.side_effect = clear, submit
        virtual.send.side_effect = send
        with ExitStack() as stack:
            stack.enter_context(patch.object(app, 'LocalStreamClient', return_value=client))
            stack.enter_context(patch.object(app, 'PromptPanel', return_value=panel))
            stack.enter_context(patch.object(app, 'FramePacer'))
            stack.enter_context(patch.object(app, 'VideoStats'))
            stack.enter_context(patch.object(app.time, 'monotonic', side_effect=count(0, 1)))
            stack.enter_context(patch.object(app.pyvirtualcam, 'Camera', return_value=virtual,
                side_effect=RuntimeError('OBS busy') if open_error else None))
            stack.enter_context(patch('builtins.print'))
            for name in ('namedWindow', 'resizeWindow', 'setMouseCallback', 'destroyAllWindows', 'imshow', 'setWindowProperty'):
                stack.enter_context(patch.object(app.cv2, name))
            stack.enter_context(patch.object(app.cv2, 'getWindowProperty', return_value=1))
            stack.enter_context(patch.object(app.cv2, 'waitKey', side_effect=keys))
            opener = Mock(return_value=(capture, frame.copy()))
            if open_error or send_error:
                with self.assertRaises(RuntimeError):
                    app.run_local_webcam(opener, width=160, height=90)
            else:
                self.assertEqual(app.run_local_webcam(opener, width=160, height=90), 0)
        client.close.assert_called_once()
        panel.close.assert_called_once()
        capture.release.assert_called_once()
        self.assertEqual(virtual.close.call_count, int(not open_error))
        return frame, artwork, sent, client

    def test_loading_never_blocks_live_feed_and_controls_invalidate_results(self):
        frame, _, sent, client = self.run_video([ord(' '), ord('3'), ord('m'), ord('r'), ord('q')], ready=False)
        for actual in sent[:3]:
            np.testing.assert_array_equal(actual, cv2.flip(frame, 1))
        for actual in sent[3:]:
            np.testing.assert_array_equal(actual, frame)
        self.assertEqual(client.clear.call_count, 4)
        self.assertEqual(client.submit.call_args.args[-2], 3)

    def test_generated_frame_reaches_obs_without_preview_controls_then_space_restores_live(self):
        frame, artwork, sent, _ = self.run_video([ord(' '), 0, 0, ord(' '), ord('q')])
        np.testing.assert_array_equal(sent[0], cv2.flip(frame, 1))
        np.testing.assert_array_equal(sent[2], artwork)
        np.testing.assert_array_equal(sent[-1], cv2.flip(frame, 1))

    def test_failure_releases_gpu_and_camera(self):
        self.run_video([ord('q')], open_error=True)
        self.run_video([ord('q')], send_error=True)

    def test_panel_can_start_video_and_change_prompt_and_strength_without_camera_restart(self):
        _, _, _, client = self.run_video([0, 0, ord('q')], updates=(
            PanelUpdate(toggle=True),
            PanelUpdate(prompt='A watercolor portrait in a fantasy forest', strength=3)))
        self.assertEqual(client.clear.call_count, 2)
        self.assertEqual(client.submit.call_args_list[0].args[3], DEFAULT_PROMPT)
        self.assertEqual(client.submit.call_args_list[1].args[1:],
                         (2, 3, 'A watercolor portrait in a fantasy forest'))
        self.assertEqual(client.submit.call_args_list[2].args[3],
                         'A watercolor portrait in a fantasy forest')


class PromptTests(unittest.TestCase):
    def test_live_edit_debounces_latest_text_and_applies_only_once(self):
        draft = PromptDraft()
        draft.edit('A watercolor', 1)
        self.assertIsNone(draft.take(1.4))
        draft.edit('A watercolor portrait', 1.5)
        self.assertIsNone(draft.take(2))
        self.assertEqual(draft.take(2.2), 'A watercolor portrait')
        self.assertIsNone(draft.take(3))
        draft.edit('  A watercolor\nportrait  ', 4)
        self.assertIsNone(draft.take(5))

    def test_apply_forces_update_and_empty_or_long_draft_preserves_last_prompt(self):
        draft = PromptDraft()
        draft.edit('A comic book superhero', 1)
        self.assertEqual(draft.take(1, force=True), 'A comic book superhero')
        for invalid in ('  ', 'a'*1001):
            draft.edit(invalid, 2)
            with self.assertRaises(ValueError):
                draft.take(2, force=True)
            self.assertEqual(draft.applied, 'A comic book superhero')
        draft.edit(DEFAULT_PROMPT, 3)
        self.assertEqual(draft.take(3, force=True), DEFAULT_PROMPT)
        self.assertEqual(normalize_prompt('una città\ncolorata'), 'una città colorata')

    def test_engine_updates_embeddings_once_and_keeps_custom_text_when_strength_changes(self):
        engine = StreamEngine.__new__(StreamEngine)
        engine.stream = Mock()
        engine.prompt, engine.strength = DEFAULT_PROMPT, 2
        engine.configure(2, 'A robot in space')
        engine.configure(2, 'A robot in space')
        engine.stream.update_prompt.assert_called_once_with('A robot in space')
        engine.stream.prepare.assert_not_called()
        engine.configure(3)
        engine.stream.prepare.assert_called_once_with('A robot in space',
            num_inference_steps=50, guidance_scale=1., seed=42)
        engine.configure(3, DEFAULT_PROMPT)
        self.assertEqual(engine.prompt, DEFAULT_PROMPT)


if __name__ == '__main__':
    unittest.main()
