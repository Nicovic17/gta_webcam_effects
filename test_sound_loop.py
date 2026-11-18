"""Both supported audio backends must loop natively and remain stoppable."""
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from vanish_sound import VanishSound


class SoundLoopTests(unittest.TestCase):
    def test_pygame_loop_and_default_one_shot(self):
        player = object.__new__(VanishSound)
        player.ffplay = None
        player.sound = Mock()
        player.play(loop=True)
        player.sound.play.assert_called_once_with(loops=-1)
        player.stop()
        self.assertEqual(player.sound.stop.call_count, 2)
        player.sound.reset_mock()
        player.play()
        player.sound.play.assert_called_once_with()

    def test_ffplay_native_loop_single_process_and_stop(self):
        player = object.__new__(VanishSound)
        player.ffplay, player.path, player.process = 'ffplay', Path('gtapolice.mp3'), None
        process = Mock()
        process.poll.return_value = None
        with patch('vanish_sound.subprocess.Popen', return_value=process) as start:
            player.play(loop=True)
            command = start.call_args.args[0]
            self.assertEqual(command[command.index('-loop')+1], '0')
            self.assertEqual(command[-2:], ['-i', 'gtapolice.mp3'])
            player.stop()
            process.terminate.assert_called_once()
            process.poll.return_value = 0
            player.play()
            self.assertNotIn('-loop', start.call_args.args[0])


if __name__ == '__main__':
    unittest.main()
