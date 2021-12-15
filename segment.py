import numpy as np


class Segment1D:
    """
    Segment 1D signal
    """

    @classmethod
    def apply(cls, x, window_size=200, step_size=50):
        """
        @param x: 1D signal
        @param window_size: window size.Default: 200.
        @param step_size: step size. Default: 50
        """
        length = int(len(x) % window_size)
        pad = np.array([0] * length)
        x_padded = np.concatenate([x, pad])
        segments = [x_padded[i:i + window_size] for i in range(0, len(x_padded), step_size)]
        return segments


class SegmentND:
    """
    Segment multi-channels signal
    """

    @classmethod
    def apply(cls, x, window_size=200, step_size=50):
        """
        @param x: 2D signal
        @param window_size: window size. Default: 200.
        @param step_size: step size. Default: 50
        """
        length = int(len(x) % window_size)
        # make sure there are an even number of windows before stride tricks
        pad = np.zeros((length, x.shape[1]))
        x_padded = np.vstack([x, pad])
        segments = [x_padded[i:i + window_size, :] for i in range(0, x_padded.shape[0], step_size)]
        return segments
