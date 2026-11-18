"""Verify replay memory, cinematic phases, recovery, and runtime cleanup."""
from contextlib import ExitStack
from itertools import count
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from test_support import source_frame
from video_pipeline import AnalysisResult
from wasted_effect import WastedEffect, TOTAL_SECONDS, REASONS, REASONS_EN
import webcam_wasted as app


class WastedEffectTests(unittest.TestCase):
    def test_language_switch_updates_visible_caption_without_restarting_replay(self):
        effect = WastedEffect(320, 180)
        frame = cv2.resize(source_frame(), (320, 180))
        effect.render(frame, 0)
        effect.trigger(.1)
        italian = effect.render(frame, 1.6)
        index, started, replay = effect.reason_index, effect.started, effect.replay
        reason = effect.reasons[index]
        effect.set_language('eng')
        self.assertEqual(effect.reasons[index], REASONS_EN[REASONS.index(reason)])
        self.assertEqual(effect.reason_index, index)
        self.assertEqual(effect.started, started)
        self.assertIs(effect.replay, replay)
        english = effect.render(frame, 1.6)
        self.assertFalse(np.array_equal(italian, english))
        effect.set_language('it')
        np.testing.assert_array_equal(effect.render(frame, 1.6), italian)
        self.assertEqual(set(WastedEffect(160, 90, language='eng').reasons), set(REASONS_EN))

    def test_idle_and_respawn_return_latest_live_image_without_mutating_input(self):
        effect = WastedEffect(160, 90)
        first, newest = source_frame(), 255-source_frame()
        np.testing.assert_array_equal(effect.render(first, 0), first)
        self.assertTrue(effect.trigger(.1, (.2, .1, .4, .5)))
        self.assertFalse(effect.trigger(.2))
        shown = effect.render(newest, 1.5)
        self.assertFalse(np.array_equal(shown, newest))
        # Text is red, but the scene outside its band becomes monochrome.
        np.testing.assert_array_equal(shown[12:23, :, 0], shown[12:23, :, 1])
        np.testing.assert_array_equal(shown[12:23, :, 1], shown[12:23, :, 2])
        np.testing.assert_array_equal(effect.render(newest, .1+TOTAL_SECONDS), newest)
        self.assertFalse(effect.running)
        self.assertEqual(effect.replay, [])
        np.testing.assert_array_equal(first, source_frame())
        np.testing.assert_array_equal(newest, 255-source_frame())

    def test_real_replay_uses_pretrigger_frames_slowly_and_retains_no_unbounded_video(self):
        effect = WastedEffect(160, 90)
        for index in range(150):
            effect.render(np.full((90, 160, 3), index, np.uint8), index*.06)
        self.assertLessEqual(len(effect.history), 16)
        self.assertLessEqual(effect.history[-1][0]-effect.history[0][0], .8)
        effect.trigger(9)
        first = effect._source(np.zeros((90, 160, 3), np.uint8), 0)
        later = effect._source(np.zeros((90, 160, 3), np.uint8), 1)
        self.assertGreater(int(later[0, 0, 0]), int(first[0, 0, 0]))
        self.assertLess(int(later[0, 0, 0]), 149)
        effect.reset()
        self.assertFalse(effect.history)
        self.assertFalse(effect.replay)

    def test_title_is_timed_and_face_anchor_changes_camera_zoom(self):
        effect = WastedEffect(320, 180)
        frame = cv2.resize(source_frame(), (320, 180))
        effect.render(frame, 0)
        effect.trigger(.1, (.1, .1, .3, .4))
        before = effect.render(frame, .5)
        after = effect.render(frame, 1.6)
        red = after[:, :, 2].astype(int)-after[:, :, 1].astype(int)
        self.assertGreater((red > 40).sum(), 200)
        self.assertFalse(np.array_equal(before, after))
        other = WastedEffect(320, 180)
        other.render(frame, 0)
        other.trigger(.1, (.7, .3, .9, .6))
        self.assertFalse(np.array_equal(other.render(frame, 1.6), after))

    def test_reset_and_new_trigger_change_caption_and_work_without_history(self):
        effect = WastedEffect(160, 90)
        frame = source_frame()
        effect.trigger(1)
        first_reason = effect.reasons[effect.reason_index]
        shown = effect.render(frame, 2.5)
        self.assertEqual(shown.shape, frame.shape)
        effect.reset()
        np.testing.assert_array_equal(effect.render(frame, 3), frame)
        effect.trigger(4)
        self.assertNotEqual(effect.reasons[effect.reason_index], first_reason)
        for age in (0, .3, .9, 2, 4.2, 4.7, 5.1, 5.3):
            image = effect.render(frame, 4+age)
            self.assertEqual(image.dtype, np.uint8)
            self.assertTrue(image.flags.c_contiguous)


class WastedRuntimeTests(unittest.TestCase):
    def test_flip_mirrors_complete_effect_including_captions_and_respawn(self):
        rendered = []
        render = WastedEffect.render

        def record_render(effect, frame, now):
            image = render(effect, frame, now)
            rendered.append(image.copy())
            return image

        with patch.object(app.WastedEffect, 'render', autospec=True, side_effect=record_render):
            sent, _, _ = self.run_video(
                [ord(' '), 0, 0, ord('m'), ord(' '), 0, 0, ord('q')],
                clock=[0, .1, .1, 1.6, 4.8, 5.5, 6, 6, 7.5, 10.7, 11.4])
        for index, (output, original) in enumerate(zip(sent, rendered)):
            with self.subTest(frame=index):
                expected = cv2.flip(original, 1) if index < 4 else original
                np.testing.assert_array_equal(output, expected)

    def test_language_button_toggles_and_preserves_mirror_mouse_control(self):
        effect, mirror = WastedEffect(960, 540), Mock()
        control = app.LanguageControl(effect, mirror, 960, 540)
        x1, y1, x2, y2 = control.bounds
        control.on_mouse(cv2.EVENT_LBUTTONUP, (x1+x2)//2, (y1+y2)//2, 0, None)
        self.assertEqual(effect.language, 'eng')
        mirror.on_mouse.assert_not_called()
        control.on_mouse(cv2.EVENT_LBUTTONUP, 0, 0, 0, None)
        mirror.on_mouse.assert_called_once_with(cv2.EVENT_LBUTTONUP, 0, 0, 0, None)
        control.toggle()
        self.assertEqual(effect.language, 'it')

    def test_language_keyboard_switch_reaches_effect(self):
        with patch.object(app.WastedEffect, 'set_language', autospec=True,
                          side_effect=WastedEffect.set_language) as set_language:
            self.run_video([ord('l'), ord(' '), ord('l'), ord('q')])
        self.assertEqual([call.args[1] for call in set_language.call_args_list],
                         ['it', 'eng', 'it'])

    def run_video(self, keys, *, clock=None, open_error=False, send_error=False,
                  sound_error=False, play_error=False):
        capture, virtual, analysis, sound = Mock(), Mock(), Mock(), Mock()
        frame = source_frame()
        capture.read.return_value = (True, frame.copy())
        virtual.device = 'OBS Virtual Camera'
        analysis.latest.return_value = AnalysisResult(1, ((.3, .2, .5, .6), 0, 0), 0)
        sent = []
        def send(image):
            if send_error:
                raise RuntimeError('send failed')
            sent.append(image.copy())
        virtual.send.side_effect = send
        if play_error:
            sound.play.side_effect = RuntimeError('no sound channel')
        clock = iter(clock) if clock is not None else count(0, .1)
        with ExitStack() as stack:
            stack.enter_context(patch('builtins.print'))
            stack.enter_context(patch.object(app, 'FramePacer'))
            stack.enter_context(patch.object(app, 'create_face_analyzer', return_value=analysis))
            stack.enter_context(patch.object(app, 'VanishSound', return_value=sound,
                                           side_effect=RuntimeError('no audio') if sound_error else None))
            stack.enter_context(patch.object(app.time, 'monotonic', side_effect=lambda: next(clock)))
            stack.enter_context(patch.object(app.pyvirtualcam, 'Camera', return_value=virtual,
                                           side_effect=RuntimeError('OBS busy') if open_error else None))
            for name in ('namedWindow', 'resizeWindow', 'destroyAllWindows', 'setMouseCallback', 'imshow', 'setWindowProperty'):
                stack.enter_context(patch.object(app.cv2, name))
            stack.enter_context(patch.object(app.cv2, 'getWindowProperty', return_value=1))
            stack.enter_context(patch.object(app.cv2, 'waitKey', side_effect=keys))
            opener = Mock(return_value=(capture, frame.copy()))
            if open_error or send_error:
                with self.assertRaises(RuntimeError):
                    app.run_wasted_webcam(opener, width=160, height=90)
            else:
                self.assertEqual(app.run_wasted_webcam(opener, width=160, height=90), 0)
        analysis.close.assert_called_once()
        capture.release.assert_called_once()
        self.assertEqual(virtual.close.call_count, int(not open_error))
        self.assertEqual(sound.close.call_count, int(not sound_error))
        return sent, sound, frame

    def test_trigger_ignores_repeats_and_reset_restores_live_video(self):
        sent, sound, frame = self.run_video([ord(' '), ord(' '), ord('r'), ord(' '), ord('q')])
        self.assertEqual(sound.play.call_count, 2)
        sound.stop.assert_called_once()
        np.testing.assert_array_equal(sent[0], cv2.flip(frame, 1))
        self.assertFalse(np.array_equal(sent[1], sent[0]))
        np.testing.assert_array_equal(sent[3], sent[0])

    def test_auto_respawn_restores_live_camera_and_mirror_control(self):
        sent, sound, frame = self.run_video([ord(' '), 0, ord('m'), ord('q')],
                                          clock=[0, .1, .1, 1.3, 5.5, 6])
        np.testing.assert_array_equal(sent[2], cv2.flip(frame, 1))
        np.testing.assert_array_equal(sent[3], frame)
        sound.play.assert_called_once()

    def test_sound_unavailable_still_renders_effect(self):
        sent, _, frame = self.run_video([ord(' '), 0, ord('q')], sound_error=True)
        self.assertFalse(np.array_equal(sent[1], cv2.flip(frame, 1)))
        self.run_video([ord(' '), 0, ord('q')], play_error=True)

    def test_virtual_camera_errors_release_camera_analyzer_and_sound(self):
        self.run_video([ord('q')], open_error=True)
        self.run_video([ord('q')], send_error=True)


if __name__ == '__main__':
    unittest.main()
