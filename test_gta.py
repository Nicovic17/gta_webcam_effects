"""Standalone menu/routing/acquisition checks without opening camera or OBS."""
from contextlib import ExitStack
import unittest
from unittest.mock import Mock, patch
import cv2
import numpy as np

import gta
import gta_camera
import gta_menu


class StandaloneTests(unittest.TestCase):
    def test_menu_keyboard_click_and_cancel(self):
        for index, (mode, _, _) in enumerate(gta_menu.EFFECTS):
            for use_mouse in (False, True):
                with ExitStack() as stack:
                    callback = []
                    stack.enter_context(patch.object(gta_menu.cv2, 'setMouseCallback',
                        side_effect=lambda _, fn: callback.append(fn)))
                    for name in ('namedWindow', 'imshow', 'destroyWindow'):
                        stack.enter_context(patch.object(gta_menu.cv2, name))
                    stack.enter_context(patch.object(gta_menu.cv2, 'getWindowProperty', return_value=1))
                    def key(ms):
                        if use_mouse:
                            callback[0](cv2.EVENT_LBUTTONUP, 200, 120+index*gta_menu.MENU_ROW_STEP, 0, None)
                            return 255
                        return ord('1')+index
                    stack.enter_context(patch.object(gta_menu.cv2, 'waitKey', side_effect=key))
                    self.assertEqual(gta_menu.choose_effect(), mode)
        with ExitStack() as stack:
            for name in ('namedWindow', 'setMouseCallback', 'imshow', 'destroyWindow'):
                stack.enter_context(patch.object(gta_menu.cv2, name))
            stack.enter_context(patch.object(gta_menu.cv2, 'waitKey', return_value=27))
            self.assertIsNone(gta_menu.choose_effect())

    def test_dispatch_only_opens_the_selected_effect(self):
        for mode in ('wasted', 'wanted', 'vice', 'vice_local', None):
            with patch.object(gta, 'choose_effect', return_value=mode), \
                 patch.object(gta, 'open_camera') as camera, \
                 patch('webcam_wasted.run_wasted_webcam', return_value=0) as wasted, \
                 patch('webcam_neon.run_neon_webcam', return_value=0) as wanted, \
                 patch('webcam_vice_ai.run_vice_webcam', return_value=0) as vice, \
                 patch('webcam_vice_local.run_local_webcam', return_value=0) as local:
                self.assertEqual(gta.main([]), 0)
                self.assertEqual([wasted.call_count, wanted.call_count, vice.call_count, local.call_count],
                                 [int(mode == item) for item in ('wasted','wanted','vice','vice_local')])
                camera.assert_not_called()
                if mode == 'wanted':
                    self.assertEqual(wanted.call_args.args, ('wanted', camera))
        with patch.object(gta, 'choose_effect') as menu, \
             patch('webcam_vice_local.run_local_webcam', return_value=0):
            self.assertEqual(gta.main(['--mode', 'vice_local']), 0)
            menu.assert_not_called()

    def test_mjpeg_configuration_and_latest_frame_capture_are_preserved(self):
        camera, wrapped = Mock(), Mock()
        first = np.zeros((720, 1280, 3), np.uint8)
        camera.isOpened.return_value = True
        camera.read.return_value = (True, first)
        camera.get.return_value = cv2.VideoWriter_fourcc(*'MJPG')
        with patch.object(gta_camera.cv2, 'VideoCapture', return_value=camera), \
             patch.object(gta_camera, 'LatestFrameCapture', return_value=wrapped) as factory, \
             patch('builtins.print'):
            capture, pending = gta_camera.open_camera()
        self.assertIs(capture, wrapped)
        self.assertIsNone(pending)
        factory.assert_called_once_with(camera, first)
        self.assertEqual(camera.set.call_args_list[-1].args,
                         (cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG')))
