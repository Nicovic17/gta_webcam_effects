"""Real renderers and runtime checks; no physical camera or audio playback."""
from contextlib import ExitStack
from itertools import count
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from gta_webcam_effects import WantedEffect, ViceEffect
from test_support import source_frame
from test_support import ImmediateAnalysis
from video_pipeline import AnalysisResult
import webcam_neon as app


class NeonEffectTests(unittest.TestCase):
    def test_idle_trigger_reset_and_mirror_preserve_live_input(self):
        frame = source_frame()
        for factory in (WantedEffect,):
            effect = factory(160, 90)
            idle = effect.render(frame, None, 0)
            if factory is ViceEffect:
                np.testing.assert_array_equal(idle, frame)
            self.assertTrue(effect.trigger(1, (.2, .2, .4, .6)))
            self.assertFalse(effect.trigger(2))
            shown = effect.render(frame, None, 3)
            self.assertFalse(np.array_equal(shown, frame))
            effect.mirror()
            self.assertTrue(effect.running)
            self.assertEqual(effect.face_box, (.6, .2, .8, .6))
            effect.reset()
            np.testing.assert_array_equal(effect.render(frame, None, 4), idle)
            np.testing.assert_array_equal(frame, source_frame())

    def test_wanted_keeps_live_camera_until_five_stars_then_captures_face(self):
        effect = WantedEffect(320, 180)
        frame = cv2.resize(source_frame(), (320, 180))
        effect.trigger(1, (.2, .2, .4, .6))
        self.assertEqual([effect.star_count(age) for age in (0, 1.6, 3.2, 4.8, 6.4, 8, 100)], [0, 1, 2, 3, 4, 5, 5])
        shown = effect.render(frame, None, 5)
        np.testing.assert_array_equal(shown[55:125, 110:210], frame[55:125, 110:210])
        self.assertIsNone(effect.portrait)
        effect.render(frame, None, 9.1, face_box=(.6, .2, .8, .5))
        self.assertIsNotNone(effect.portrait)
        self.assertEqual(effect.portrait.shape[0], effect.portrait.shape[1])
        photo = effect.portrait.copy()
        effect.render(255-frame, None, 11)
        np.testing.assert_array_equal(effect.portrait, photo)

    def test_vice_routes_to_the_openai_capture_runtime(self):
        opener = Mock()
        with patch('webcam_vice_ai.run_vice_webcam', return_value=0) as vice:
            self.assertEqual(app.run_neon_webcam('vice', opener, width=320, height=180), 0)
            vice.assert_called_once_with(opener, width=320, height=180, fps=30)


class NeonAnalyzerTests(unittest.TestCase):
    def test_models_close_on_initialization_failure(self):
        detector = Mock()
        with patch('gta_tracking.create_face_tracker', return_value=detector), \
             patch.object(app.vision.ImageSegmenter, 'create_from_options', side_effect=RuntimeError('failed')):
            with self.assertRaises(RuntimeError):
                app.create_neon_analyzer(160, 90)
        detector.close.assert_called_once()

    def test_mirror_generation_drops_cached_mask_and_closes_models(self):
        detector, segmenter = Mock(), Mock()
        detector.detect_for_video.return_value = SimpleNamespace(face_landmarks=[[
            SimpleNamespace(x=.3, y=.2), SimpleNamespace(x=.5, y=.6)]])
        mask = np.ones((90, 160), np.float32)
        with patch('gta_tracking.create_face_tracker', return_value=detector), \
             patch('gta_tracking.person_mask', return_value=mask), \
             patch.object(app.vision.ImageSegmenter, 'create_from_options', return_value=segmenter), \
             patch.object(app, 'LatestAnalysis', ImmediateAnalysis):
            analysis = app.create_neon_analyzer(160, 90)
            frame = source_frame()
            analysis.submit((frame, 1, .1, False, 0))
            self.assertEqual(analysis.latest().data['face_box'], (.3, .2, .5, .6))
            analysis.submit((frame, 2, .2, True, 0))
            self.assertIs(analysis.latest().data['mask'], mask)
            analysis.submit((frame, 3, .3, False, 1))
            self.assertIsNone(analysis.latest().data['mask'])
            self.assertEqual(analysis.latest().data['generation'], 1)
            analysis.close()
        detector.close.assert_called_once()
        segmenter.close.assert_called_once()


class NeonRuntimeTests(unittest.TestCase):
    def test_flip_mirrors_hud_scan_and_wanted_card_without_flipping_cached_photo_twice(self):
        rendered, portraits = [], []
        render = WantedEffect.render

        def record_render(effect, frame, mask, now, face_box=None):
            image = render(effect, frame, mask, now, face_box=face_box)
            rendered.append(image.copy())
            portraits.append(None if effect.portrait is None else effect.portrait.copy())
            return image

        with patch.object(app.WantedEffect, 'render', autospec=True, side_effect=record_render):
            sent, previews, _, _ = self.run_video(
                'wanted', [ord(' '), 0, 0, ord('m'), ord('q')],
                clock_values=[0, .1, .1, 8.2, 9.5, 10, 10])
        self.assertEqual(len(sent), 5)
        for index, (output, original) in enumerate(zip(sent, rendered)):
            with self.subTest(frame=index):
                np.testing.assert_array_equal(output, cv2.flip(original, 1) if index < 4 else original)
                self.assertFalse(np.array_equal(previews[index], output))
        self.assertIsNotNone(portraits[-1])
        np.testing.assert_array_equal(portraits[-1], portraits[-2])
        np.testing.assert_array_equal(sent[-2], cv2.flip(sent[-1], 1))

    def run_video(self, mode, keys, *, open_error=False, send_error=False, no_audio=False, clock_values=None):
        capture, virtual, analysis, sound = Mock(), Mock(), Mock(), Mock()
        frame = source_frame()
        capture.read.return_value = (True, frame.copy())
        virtual.device = 'OBS Virtual Camera'
        analysis.latest.return_value = AnalysisResult(1, dict(mask=np.zeros((90, 160), np.float32),
            face_box=(.3, .2, .5, .6), face_at=0, generation=0), 0)
        sent, previews = [], []
        def send(image):
            if send_error:
                raise RuntimeError('send failed')
            sent.append(image.copy())
        virtual.send.side_effect = send
        clock = count(0, .6) if clock_values is None else iter(clock_values)
        with ExitStack() as stack:
            stack.enter_context(patch('builtins.print'))
            stack.enter_context(patch.object(app, 'FramePacer'))
            stack.enter_context(patch.object(app, 'create_neon_analyzer', return_value=analysis))
            sound_factory = stack.enter_context(patch.object(app, 'VanishSound', return_value=sound,
                                           side_effect=RuntimeError('no audio') if no_audio else None))
            stack.enter_context(patch.object(app.time, 'monotonic', side_effect=lambda: next(clock)))
            stack.enter_context(patch.object(app.pyvirtualcam, 'Camera', return_value=virtual,
                                           side_effect=RuntimeError('OBS busy') if open_error else None))
            for name in ('namedWindow', 'resizeWindow', 'destroyAllWindows', 'setMouseCallback', 'setWindowProperty'):
                stack.enter_context(patch.object(app.cv2, name))
            stack.enter_context(patch.object(app.cv2, 'imshow', side_effect=lambda title, image: previews.append(image.copy())))
            stack.enter_context(patch.object(app.cv2, 'getWindowProperty', return_value=1))
            stack.enter_context(patch.object(app.cv2, 'waitKey', side_effect=keys))
            opener = Mock(return_value=(capture, frame.copy()))
            if open_error or send_error:
                with self.assertRaises(RuntimeError):
                    app.run_neon_webcam(mode, opener, width=160, height=90)
            else:
                self.assertEqual(app.run_neon_webcam(mode, opener, width=160, height=90), 0)
        analysis.close.assert_called_once()
        capture.release.assert_called_once()
        self.assertEqual(virtual.close.call_count, int(not open_error))
        self.assertEqual(sound.close.call_count, int(not no_audio))
        sound_factory.assert_called_once_with(app.SOUNDS[mode])
        return sent, previews, sound, frame

    def test_both_modes_toggle_back_to_clean_virtual_feed(self):
        for mode in ('wanted',):
            sent, previews, sound, frame = self.run_video(mode, [ord(' '), 0, 0, 0, ord(' '), ord('q')])
            if mode == 'vice':
                np.testing.assert_array_equal(sent[0], cv2.flip(frame, 1))
            self.assertFalse(np.array_equal(sent[3], sent[0]))
            np.testing.assert_array_equal(sent[5], sent[0])
            self.assertFalse(np.array_equal(previews[0], sent[0]))
            if mode == 'wanted':
                sound.play.assert_not_called()
            else:
                sound.play.assert_called_once_with()
            sound.stop.assert_called_once()

    def test_mirror_reset_fullscreen_and_silent_fallback(self):
        for mode in ('wanted',):
            sent, _, _, frame = self.run_video(mode, [ord(' '), ord('m'), ord('r'), ord('f'), ord('q')], no_audio=True)
            if mode == 'vice':
                np.testing.assert_array_equal(sent[-1], frame)
            else:
                np.testing.assert_array_equal(sent[-1], WantedEffect(160, 90).render(frame, None, 0))

    def test_virtual_failures_close_resources(self):
        self.run_video('wanted', [ord('q')], open_error=True)
        self.run_video('wanted', [ord('q')], send_error=True)

    def test_wanted_starts_one_continuous_loop_and_vice_plays_intro_once(self):
        for mode, expected in (('wanted', 1),):
            _, _, sound, _ = self.run_video(mode, [ord(' ')]+[0]*20+[ord('q')])
            self.assertEqual(sound.play.call_count, expected)
            if mode == 'wanted':
                sound.play.assert_called_once_with(loop=True)
            else:
                sound.play.assert_called_once_with()

    def test_wanted_audio_waits_for_fifth_yellow_star_and_resets_between_runs(self):
        # The first run is cancelled just before the fifth star: no audio.
        _, _, sound, _ = self.run_video('wanted', [ord(' '), 0, ord(' '), ord('q')],
            clock_values=[0, .1, .1, 8.09, 8.099, 9])
        sound.play.assert_not_called()
        # The next sequence reaches five stars twice, with a reset in between.
        _, _, sound, _ = self.run_video('wanted', [ord(' '), 0, ord('r'), ord(' '), 0, ord('q')],
            clock_values=[0, .1, .1, 8.10, 9, 9.1, 9.1, 17.10, 18])
        self.assertEqual(sound.play.call_count, 2)
        for call in sound.play.call_args_list:
            self.assertEqual(call.kwargs, {'loop': True})
        sound.stop.assert_called_once()


if __name__ == '__main__':
    unittest.main()
