"""Check the new HUD, precise alert timing, snapshot ownership and active tracking."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from wanted_effect import WantedEffect, FULL_STARS_SECONDS, SCAN_SECONDS
from test_support import source_frame
from test_support import ImmediateAnalysis
import webcam_neon as app


class WantedHudTests(unittest.TestCase):
    def test_five_asset_stars_and_map_are_always_present_without_replacing_camera(self):
        effect = WantedEffect(640, 360)
        frame = cv2.resize(source_frame(), (640, 360))
        idle = effect.render(frame, None, 0)
        outside = np.ones(frame.shape[:2], bool)
        for x1, y1, x2, y2 in effect.hud_bounds:
            outside[y1:y2, x1:x2] = False
        np.testing.assert_array_equal(idle[outside], frame[outside])
        self.assertFalse(np.array_equal(idle, frame))
        for index in range(5):
            x = effect.stars_x+index*(effect.star_size+effect.star_gap)
            y, size = effect.stars_y, effect.star_size
            self.assertFalse(np.array_equal(idle[y:y+size, x:x+size], frame[y:y+size, x:x+size]))
        effect.trigger(1)
        filled = effect.render(frame, None, 1+FULL_STARS_SECONDS-.01)
        self.assertIsNone(effect.portrait)
        np.testing.assert_array_equal(filled[outside], frame[outside])
        effect.reset()
        np.testing.assert_array_equal(effect.render(frame, None, 50), idle)

    def test_instant_colour_timing_and_snapshot_are_taken_at_five_stars_from_current_face(self):
        effect = WantedEffect(640, 360)
        first = np.zeros((360, 640, 3), np.uint8)
        current = first.copy()
        current[100:180, 440:520] = (20, 160, 240)
        effect.trigger(10, (.1, .1, .2, .3))
        self.assertEqual(effect.star_fill(.8, 0), 0)
        self.assertEqual(effect.star_fill(.8, 1), 0)
        self.assertEqual(effect.star_fill(1.6-.001, 0), 0)
        self.assertEqual(effect.star_fill(1.6, 0), 1)
        self.assertEqual(effect.star_fill(1.6+.001, 0), 1)
        effect.render(first, None, 10+FULL_STARS_SECONDS-.01)
        self.assertIsNone(effect.scan_snapshot)
        effect.render(current, None, 10+FULL_STARS_SECONDS, face_box=(440/640, 100/360, 520/640, 180/360))
        self.assertGreater(effect.portrait[:, :, 2].mean(), 150)
        np.testing.assert_array_equal(effect.scan_snapshot, current)
        photo = effect.portrait.copy()
        effect.render(first, None, 20, face_box=(.1, .1, .2, .3))
        np.testing.assert_array_equal(effect.portrait, photo)
        np.testing.assert_array_equal(current[100:180, 440:520], np.full((80, 80, 3), (20, 160, 240), np.uint8))
        effect.reset()
        self.assertIsNone(effect.portrait)
        self.assertIsNone(effect.scan_snapshot)

    def test_scan_precedes_mugshot_and_blinking_moving_lights(self):
        effect = WantedEffect(640, 360)
        frame = np.full((360, 640, 3), 75, np.uint8)
        effect.trigger(0)
        scan = effect.render(frame, None, FULL_STARS_SECONDS+.3)
        with patch.object(effect, '_mugshot', wraps=effect._mugshot) as photo:
            effect.render(frame, None, FULL_STARS_SECONDS+.4)
            photo.assert_not_called()
            first = effect.render(frame, None, FULL_STARS_SECONDS+SCAN_SECONDS+.25)
            later = effect.render(frame, None, FULL_STARS_SECONDS+SCAN_SECONDS+.55)
            self.assertEqual(photo.call_count, 2)
        self.assertFalse(np.array_equal(first, later))
        self.assertFalse(np.array_equal(first, scan))
        self.assertTrue((first[:, :15].astype(int)-frame[:, :15]).max() > 10)
        for x1, y1, x2, y2 in effect.hud_bounds[:1]:
            self.assertFalse(np.array_equal(first[y1:y2, x1:x2], later[y1:y2, x1:x2]))

    def test_portrait_mirrors_and_handles_face_at_screen_edge(self):
        effect = WantedEffect(160, 90)
        frame = source_frame()
        effect.trigger(0, (.93, .0, 1.1, .3))
        shown = effect.render(frame, None, 9)
        self.assertEqual(shown.shape, frame.shape)
        original = effect.portrait.copy()
        effect.mirror()
        np.testing.assert_array_equal(effect.portrait, cv2.flip(original, 1))
        self.assertTrue(effect.running)

    def test_new_star_blinks_then_five_stars_share_a_bounded_phase(self):
        effect = WantedEffect(640, 360)
        # Only the newly yellow star blinks; the others stay white and steady.
        self.assertEqual(effect.star_opacity(1.6+.10, 0), 1)
        self.assertLess(effect.star_opacity(1.6+.40, 0), 1)
        self.assertEqual(effect.star_opacity(1.6+.40, 1), 1)
        self.assertEqual(effect.star_opacity(1.6+2.0, 0), 1)
        for elapsed in (.10, .40, 1.20, 2.60):
            phases = [effect.star_opacity(FULL_STARS_SECONDS+elapsed, i) for i in range(5)]
            self.assertEqual(len(set(phases)), 1)
        self.assertNotEqual(effect.star_opacity(FULL_STARS_SECONDS+.10, 0),
                            effect.star_opacity(FULL_STARS_SECONDS+.40, 0))
        for elapsed in (3, 3.4, 10, 100):
            self.assertEqual([effect.star_opacity(FULL_STARS_SECONDS+elapsed, i) for i in range(5)], [1]*5)

    def test_render_switches_directly_between_white_and_yellow_assets(self):
        effect = WantedEffect(640, 360)
        black = np.zeros((360, 640, 3), np.uint8)
        effect.trigger(0)
        before = effect.render(black, None, 1.599)
        after = effect.render(black, None, 1.6)
        sx, sy, size = effect.stars_x, effect.stars_y, effect.star_size
        white = before[sy:sy+size, sx:sx+size]
        yellow = after[sy:sy+size, sx:sx+size]
        opaque = effect.star_frames[0][1] > .95
        self.assertLess(np.abs(white[opaque, 2].astype(int)-white[opaque, 0]).mean(), 3)
        self.assertGreater((yellow[opaque, 2].astype(int)-yellow[opaque, 0]).mean(), 80)
        # An unchanged second star retains exactly its white pixels.
        sx += size+effect.star_gap
        np.testing.assert_array_equal(before[sy:sy+size, sx:sx+size], after[sy:sy+size, sx:sx+size])


class WantedAnalyzerTests(unittest.TestCase):
    def test_wanted_updates_face_during_search_and_does_not_load_segmenter(self):
        detector = Mock()
        detector.detect_for_video.side_effect = [SimpleNamespace(face_landmarks=[[
            SimpleNamespace(x=x, y=.2), SimpleNamespace(x=x+.15, y=.5)]]) for x in (.1, .6)]
        with patch('gta_tracking.create_face_tracker', return_value=detector), \
             patch.object(app.vision.ImageSegmenter, 'create_from_options') as segmenter, \
             patch.object(app, 'LatestAnalysis', ImmediateAnalysis):
            analysis = app.create_neon_analyzer(160, 90, mode='wanted')
            analysis.submit((source_frame(), 1, 0, False, 0))
            analysis.submit((source_frame(), 2, 8, True, 0))
            self.assertEqual(analysis.latest().data['face_box'], (.6, .2, .75, .5))
            self.assertEqual(analysis.latest().data['face_at'], 8)
            self.assertIsNone(analysis.latest().data['mask'])
            segmenter.assert_not_called()
            analysis.close()
        detector.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
