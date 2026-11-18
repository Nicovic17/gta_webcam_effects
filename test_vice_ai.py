"""AI photo editing contract, async cancellation, transition and runtime tests.

Every API response is simulated. Tests never upload frames or spend API credit.
"""
import base64
from contextlib import ExitStack
import io
import json
from pathlib import Path
import tempfile
from threading import Event
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import urllib.error

import cv2
import numpy as np

from test_support import source_frame
from vice_ai_effect import ViceEffect, TRANSITION_SECONDS
from vice_openai import ImageConfig, OpenAIImageEditor, ImageEditJob, EditResult, ViceAPIError, EDIT_URL, NoRedirect
import webcam_vice_ai as app


class APIContractTests(unittest.TestCase):
    def test_local_config_environment_precedence_and_key_not_in_repr(self):
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, '.env').write_text('OPENAI_API_KEY=local-test-key\nOPENAI_IMAGE_QUALITY=medium\n', encoding='utf-8')
            config = ImageConfig.load(folder, environ={'OPENAI_API_KEY': 'env-test-key'})
            self.assertEqual(config.api_key, 'env-test-key')
            self.assertEqual(config.quality, 'medium')
            self.assertNotIn('env-test-key', repr(config))
            with self.assertRaises(ViceAPIError):
                ImageConfig.load(folder, environ={'OPENAI_API_KEY': ''})

    def test_multipart_edits_sends_exactly_one_photo_and_decodes_image(self):
        output = np.full((90, 160, 3), (24, 80, 190), np.uint8)
        _, png = cv2.imencode('.png', output)
        response = Mock()
        response.read.return_value = json.dumps({'data': [{'b64_json': base64.b64encode(png).decode()}]}).encode()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        opener = Mock()
        opener.open.return_value = response
        config = ImageConfig('test-key')
        actual = OpenAIImageEditor(config, opener).edit(source_frame())
        np.testing.assert_array_equal(actual, output)
        opener.open.assert_called_once()
        request = opener.open.call_args.args[0]
        self.assertEqual(request.full_url, EDIT_URL)
        self.assertEqual(request.get_header('Authorization'), 'Bearer test-key')
        self.assertEqual(request.data.count(b'name="image[]"'), 1)
        self.assertIn(b'filename="webcam.jpg"', request.data)
        self.assertIn(b'\xff\xd8\xff', request.data)
        self.assertIn(b'1536x864', request.data)
        self.assertIn(b'gpt-image-2', request.data)
        self.assertNotIn(b'test-key', request.data)
        self.assertIsNone(NoRedirect().redirect_request(request, None, 302, '', {}, 'https://elsewhere.example'))

    def test_api_errors_and_invalid_image_are_safe_and_not_retried(self):
        opener = Mock()
        opener.open.side_effect = urllib.error.HTTPError(EDIT_URL, 401, 'test-key in raw error', {}, io.BytesIO(b''))
        with self.assertRaises(ViceAPIError) as caught:
            OpenAIImageEditor(ImageConfig('test-key'), opener).edit(source_frame())
        self.assertNotIn('test-key', str(caught.exception))
        opener.open.assert_called_once()
        response = Mock()
        response.read.return_value = b'{"data": []}'
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        opener = Mock()
        opener.open.return_value = response
        with self.assertRaises(ViceAPIError):
            OpenAIImageEditor(ImageConfig('test-key'), opener).edit(source_frame())


class AsyncJobTests(unittest.TestCase):
    def test_single_request_and_cancelled_result_never_reappears(self):
        entered, release = Event(), Event()
        def edit(frame):
            entered.set()
            release.wait(2)
            return frame
        editor = Mock()
        editor.edit.side_effect = edit
        job = ImageEditJob(editor)
        try:
            self.assertTrue(job.start(source_frame()))
            self.assertTrue(entered.wait(1))
            self.assertFalse(job.start(source_frame()))
            job.cancel()
            self.assertFalse(job.start(source_frame()))
            release.set()
            deadline = time.monotonic()+2
            while job.busy and time.monotonic() < deadline:
                time.sleep(.005)
            self.assertFalse(job.busy)
            self.assertIsNone(job.take_result())
            editor.edit.assert_called_once()
        finally:
            release.set()
            job.close()

    def test_closing_inflight_job_discards_late_result_without_waiting(self):
        entered, release = Event(), Event()
        def edit(frame):
            entered.set()
            release.wait(2)
            return frame
        job = ImageEditJob(SimpleNamespace(edit=edit))
        job.start(source_frame())
        self.assertTrue(entered.wait(1))
        job.close()
        self.assertFalse(job.start(source_frame()))
        release.set()
        deadline = time.monotonic()+2
        while job.busy and time.monotonic() < deadline:
            time.sleep(.005)
        self.assertIsNone(job.take_result())


class TransitionTests(unittest.TestCase):
    def make_effect(self):
        job = Mock()
        job.start.return_value = True
        job.take_result.return_value = None
        return ViceEffect(160, 90, job), job

    def test_live_wait_single_capture_and_smooth_replacement_then_reset(self):
        effect, job = self.make_effect()
        photo = np.full((90, 160, 3), 20, np.uint8)
        live = np.full_like(photo, 70)
        artwork = np.full_like(photo, 220)
        self.assertTrue(effect.capture(photo, 0))
        self.assertFalse(effect.capture(live, .1))
        job.start.assert_called_once()
        np.testing.assert_array_equal(effect.render(live, None, 1), live)
        job.take_result.return_value = EditResult(image=artwork)
        np.testing.assert_array_equal(effect.render(live, None, 2), live)
        mid = effect.render(live, None, 3)
        self.assertTrue((mid > photo).all() and (mid < artwork).all())
        np.testing.assert_array_equal(effect.render(live, None, 2+TRANSITION_SECONDS), artwork)
        self.assertEqual(effect.phase, 'ready')
        effect.reset()
        np.testing.assert_array_equal(effect.render(live, None, 5), live)
        self.assertIsNone(effect.artwork)
        job.cancel.assert_called_once()

    def test_failure_keeps_live_camera_and_busy_job_prevents_more_uploads(self):
        effect, job = self.make_effect()
        frame = source_frame()
        effect.capture(frame, 0)
        job.take_result.return_value = EditResult(error='Quota raggiunta')
        np.testing.assert_array_equal(effect.render(frame, None, 1), frame)
        self.assertEqual(effect.phase, 'error')
        self.assertIn('Quota', effect.status(1))
        job.start.return_value = False
        self.assertFalse(effect.capture(frame, 2))
        self.assertIn('precedente', effect.status(2))

    def test_upload_keeps_camera_resolution_while_transition_fits_output(self):
        effect, job = self.make_effect()
        raw = cv2.resize(source_frame(), (1280, 720))
        self.assertTrue(effect.capture(raw, 0))
        self.assertEqual(job.start.call_args.args[0].shape, (720, 1280, 3))
        self.assertEqual(effect.photo.shape, (90, 160, 3))
        job.take_result.return_value = EditResult(image=np.full((864, 1536, 3), 150, np.uint8))
        effect.render(source_frame(), None, 1)
        self.assertEqual(effect.render(source_frame(), None, 2).shape, (90, 160, 3))


class ViceRuntimeTests(unittest.TestCase):
    def run_video(self, keys, *, result=None, clock=None, open_error=False, send_error=False):
        capture, virtual, job = Mock(), Mock(), Mock()
        frames = [source_frame(), 255-source_frame(), np.roll(source_frame(), 20, axis=1), source_frame()]
        capture.read.side_effect = [(True, frame.copy()) for frame in frames[1:]]
        job.start.return_value = True
        job.take_result.return_value = result
        sent = []
        def send(image):
            if send_error:
                raise RuntimeError('send failed')
            sent.append(image.copy())
        virtual.send.side_effect = send
        if clock is None:
            from itertools import count
            clock = count(0, .1)
        else:
            clock = iter(clock)
        with ExitStack() as stack:
            stack.enter_context(patch.object(app, 'load_config', return_value=ImageConfig('test-key')))
            stack.enter_context(patch.object(app, 'OpenAIImageEditor'))
            stack.enter_context(patch.object(app, 'ImageEditJob', return_value=job))
            stack.enter_context(patch.object(app, 'FramePacer'))
            stack.enter_context(patch.object(app.time, 'monotonic', side_effect=lambda: next(clock)))
            stack.enter_context(patch.object(app.pyvirtualcam, 'Camera', return_value=virtual,
                side_effect=RuntimeError('OBS busy') if open_error else None))
            stack.enter_context(patch('builtins.print'))
            for name in ('namedWindow', 'resizeWindow', 'setMouseCallback', 'destroyAllWindows', 'imshow', 'setWindowProperty'):
                stack.enter_context(patch.object(app.cv2, name))
            stack.enter_context(patch.object(app.cv2, 'getWindowProperty', return_value=1))
            stack.enter_context(patch.object(app.cv2, 'waitKey', side_effect=keys))
            opener = Mock(return_value=(capture, frames[0].copy()))
            if open_error or send_error:
                with self.assertRaises(RuntimeError):
                    app.run_vice_webcam(opener, width=160, height=90)
            else:
                self.assertEqual(app.run_vice_webcam(opener, width=160, height=90), 0)
        capture.release.assert_called_once()
        job.close.assert_called_once()
        self.assertEqual(virtual.close.call_count, int(not open_error))
        return sent, job, frames

    def test_wait_never_blocks_live_video_or_duplicates_requests(self):
        sent, job, frames = self.run_video([ord(' '), ord(' '), ord('r'), ord('q')])
        job.start.assert_called_once()
        job.cancel.assert_called_once()
        for actual, original in zip(sent, frames):
            np.testing.assert_array_equal(actual, cv2.flip(original, 1))

    def test_api_result_replaces_virtual_feed_and_errors_release_resources(self):
        artwork = np.full((90, 160, 3), 210, np.uint8)
        sent, _, _ = self.run_video([ord(' '), 0, 0, ord('q')], result=EditResult(image=artwork), clock=[0, 0, 1, 2, 3])
        np.testing.assert_array_equal(sent[-1], artwork)
        self.run_video([ord('q')], open_error=True)
        self.run_video([ord('q')], send_error=True)

    def test_cancelled_key_dialog_opens_no_camera(self):
        opener = Mock()
        with patch.object(app, 'load_config', return_value=None):
            self.assertEqual(app.run_vice_webcam(opener), 0)
        opener.assert_not_called()


if __name__ == '__main__':
    unittest.main()
