"""Standalone entry point for the four GTA webcam effects."""
import argparse

from gta_camera import open_camera, CAMERA_WIDTH, CAMERA_HEIGHT, VIRTUAL_CAMERA_FPS
from gta_menu import choose_effect
from video_pipeline import configure_video_processing


def main(argv=None):
    parser = argparse.ArgumentParser(description='GTA Webcam Effects')
    parser.add_argument('--mode', choices=('wasted', 'wanted', 'vice', 'vice_local'),
                        help='Salta il menu e avvia direttamente questo effetto.')
    args = parser.parse_args(argv)
    configure_video_processing()
    effect = args.mode or choose_effect()
    options = dict(width=CAMERA_WIDTH, height=CAMERA_HEIGHT, fps=VIRTUAL_CAMERA_FPS)
    if effect == 'wasted':
        from webcam_wasted import run_wasted_webcam
        return run_wasted_webcam(open_camera, **options)
    if effect == 'wanted':
        from webcam_neon import run_neon_webcam
        return run_neon_webcam('wanted', open_camera, **options)
    if effect == 'vice':
        from webcam_vice_ai import run_vice_webcam
        return run_vice_webcam(open_camera, **options)
    if effect == 'vice_local':
        from webcam_vice_local import run_local_webcam
        return run_local_webcam(open_camera, **options)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
