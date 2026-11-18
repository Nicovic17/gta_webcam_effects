"""Shared GTA test fixtures, independent of Webcam Memes and its tests."""
import numpy as np
from video_pipeline import AnalysisResult


def source_frame():
    y, x = np.mgrid[:90, :160]
    return np.stack(((x*3+y)%256, (y*5)%256, (x+y*2)%256), axis=2).astype(np.uint8)


class ImmediateAnalysis:
    def __init__(self, process, cleanup=None):
        self.process, self.cleanup = process, cleanup
        self.result, self.sequence = None, 0

    def submit(self, data):
        self.sequence += 1
        self.result = AnalysisResult(self.sequence, self.process(data), 0)

    def latest(self):
        return self.result

    def close(self):
        if self.cleanup:
            self.cleanup()
