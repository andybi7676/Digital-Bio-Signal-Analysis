import numpy as np
from scipy import stats, signal


class MaxPeak:
    @classmethod
    def apply(cls, segment, window_size=50):
        """
        @param segment: signal segment
        @param window_size: window size. Default: 50
        """
        assert 0 <= window_size <= segment.shape[1], 'window_size={} is invalid!'.format(window_size)
        length = segment.shape[1]
        peaks = np.array([])
        for i in range(0, length, window_size):
            start = i
            end = i + window_size if (i + window_size <= length) else 2 * i + window_size - length
            peak = np.max(segment[:, start: end], axis=1)
            peak = np.expand_dims(peak, axis=1)
            if i == 0:
                peaks = peak
                continue
            peaks = np.hstack((peaks, peak))
        return peaks


class Mean:
    @classmethod
    def apply(cls, segment, window_size=50):
        """
        @param segment: signal segment
        @param window_size: window size. Default: 50
        """
        assert 0 <= window_size <= segment.shape[1], 'window_size={} is invalid!'.format(window_size)
        length = segment.shape[1]
        means = np.array([])
        for i in range(0, length, window_size):
            start = i
            end = i + window_size if (i + window_size <= length) else 2 * i + window_size - length
            mean = np.mean(segment[:, start: end], axis=1)
            mean = np.expand_dims(mean, axis=1)
            if i == 0:
                means = mean
                continue
            means = np.hstack((means, mean))
        return means


class Variance:
    @classmethod
    def apply(cls, segment, window_size=50):
        """
        @param segment: signal segment
        @param window_size: window size. Default: 50.
        """
        assert 0 <= window_size <= segment.shape[1], 'window_size={} is invalid!'.format(window_size)
        length = segment.shape[1]
        vars = np.array([])
        for i in range(0, length, window_size):
            start = i
            end = i + window_size if (i + window_size <= length) else 2 * i + window_size - length
            var = np.var(segment[:, start: end], axis=1)
            var = np.expand_dims(var, axis=1)
            if i == 0:
                vars = var
                continue
            vars = np.hstack((vars, var))
        return vars


class StandardDeviation:
    @classmethod
    def apply(cls, segment, window_size=50):
        """
        @param segment: signal segment
        @param window_size: window size. Default: 50.
        """
        assert 0 <= window_size <= segment.shape[1], 'window_size={} is invalid!'.format(window_size)
        length = segment.shape[1]
        stds = np.array([])
        for i in range(0, length, window_size):
            start = i
            end = i + window_size if (i + window_size <= length) else 2 * i + window_size - length
            std = np.std(segment[:, start: end], axis=1)
            std = np.expand_dims(std, axis=1)
            if i == 0:
                stds = std
                continue
            stds = np.hstack((stds, std))
        return stds


class Skewness:
    @classmethod
    def apply(cls, segment, window_size=50):
        """
        @param segment: signal segment
        @param window_size: window size. Default: 50.
        """
        assert 0 <= window_size <= segment.shape[1], 'window_size={} is invalid!'.format(window_size)
        length = segment.shape[1]
        skews = np.array([])
        for i in range(0, length, window_size):
            start = i
            end = i + window_size if (i + window_size <= length) else 2 * i + window_size - length
            skew = stats.skew(segment[:, start: end], axis=1)
            skew = np.expand_dims(skew, axis=1)
            if i == 0:
                skews = skew
                continue
            skews = np.hstack((skews, skew))
        return skews


class Kurtosis:
    @classmethod
    def apply(cls, segment, window_size=50):
        """
        @param segment: signal segment
        @param window_size: window size. Default: 50.
        """
        assert 0 <= window_size <= segment.shape[1], 'window_size={} is invalid!'.format(window_size)
        length = segment.shape[1]
        kurts = np.array([])
        for i in range(0, length, window_size):
            start = i
            end = i + window_size if (i + window_size <= length) else 2 * i + window_size - length
            kurt = stats.kurtosis(segment[:, start: end], axis=1)
            kurt = np.expand_dims(kurt, axis=1)
            if i == 0:
                kurts = kurt
                continue
            kurts = np.hstack((kurts, kurt))
        return kurts


class RootMeanSquare:
    @classmethod
    def apply(cls, segment, window_size=50):
        """
        @param segment: signal segment
        @param window_size: window size. Default: 50.
        """
        assert 0 <= window_size <= segment.shape[1], 'window_size={} is invalid!'.format(window_size)
        length = segment.shape[1]
        rmss = np.array([])
        for i in range(0, length, window_size):
            start = i
            end = i + window_size if (i + window_size <= length) else 2 * i + window_size - length
            rms = np.sqrt(np.mean(segment[:, start: end] ** 2, axis=1))
            rms = np.expand_dims(rms, axis=1)
            if i == 0:
                rmss = rms
                continue
            rmss = np.hstack((rmss, rms))
        return rmss


class WaveformLength:
    @classmethod
    def apply(cls, segment, window_size=50):
        """
        @param segment: signal segment
        @param window_size: window size. Default: 50.
        """
        assert 0 <= window_size <= segment.shape[1], 'window_size={} is invalid!'.format(window_size)
        length = segment.shape[1]
        wls = np.array([])
        for i in range(0, length, window_size):
            start = i
            end = i + window_size if (i + window_size <= length) else 2 * i + window_size - length
            wl = np.sum(np.abs(np.diff(segment[:, start: end], axis=1)), axis=1)
            wl = np.expand_dims(wl, axis=1)
            if i == 0:
                wls = wl
                continue
            wls = np.hstack((wls, wl))
        return wls


class WillisonAmplitude:
    @classmethod
    def apply(cls, segment, window_size=50, eps=0.5):
        """
        @param segment: signal segment
        @param window_size: window size. Default: 50.
        @param eps: threshold value. Default: 0.5.
        """
        assert 0 <= window_size <= segment.shape[1], 'window_size={} is invalid!'.format(window_size)
        if eps > 1:
            eps = 1
        elif eps < 0:
            eps = 0
        length = segment.shape[1]
        wamps = np.array([])
        for i in range(0, length, window_size):
            start = i
            end = i + window_size if (i + window_size <= length) else 2 * i + window_size - length
            wl = np.abs(np.diff(segment[:, start: end], axis=1))
            mask = wl[wl > eps]
            wl[mask is True] = 1
            wl[mask is False] = 0
            wamp = np.sum(wl, axis=1)
            wamp = np.expand_dims(wamp, axis=1)
            if i == 0:
                wamps = wamp
                continue
            wamps = np.hstack((wamps, wamp))
        return wamps


class PSD:
    """
    Power Spectral Density
    """

    @classmethod
    def apply(cls, segment, window_size=50, fs=512, average='mean'):
        """
        @param segment: signal segment.
        @param window_size: window size. Default: 50.
        @param fs: sampling rate. Default: 512.
        @param average: 'mean' or 'median'. Default: 'mean'
        """
        assert 0 <= window_size <= segment.shape[1], 'window_size={} is invalid!'.format(window_size)
        length = segment.shape[1]
        psds = np.array([])
        for i in range(0, length, window_size):
            start = i
            end = i + window_size if (i + window_size <= length) else 2 * i + window_size - length
            freq, power = signal.welch(segment[:, start: end],
                                       fs=fs,
                                       scaling='density',
                                       detrend=False,
                                       nfft=window_size,
                                       average=average,
                                       nperseg=window_size,
                                       return_onesided=True,
                                       axis=1)
            if i == 0:
                psds = power
                continue
            psds = np.hstack((psds, power))
        return psds


class STFT:
    """
    Short Time Frequency Transform
    """

    @classmethod
    def apply(cls, segment, window_size=50, fs=512):
        """
                @param segment: signal segment.
                @param window_size: window size. Default: 50.
                @param fs: sampling rate. Default: 512.
                """
        assert 0 <= window_size <= segment.shape[1], 'window_size={} is invalid!'.format(window_size)
        length = segment.shape[1]
        stfts = np.array([])
        for i in range(0, length, window_size):
            start = i
            end = i + window_size if (i + window_size <= length) else 2 * i + window_size - length
            freq, t, stft = signal.stft(segment[:, start: end],
                                        fs=fs,
                                        detrend=False,
                                        nfft=window_size,
                                        nperseg=window_size,
                                        return_onesided=True,
                                        axis=1)
            if i == 0:
                stfts = stft
                continue
            stfts = np.dstack((stfts, stft))
        return stfts


class CWT:
    """
    Continuous Wavelet Transform
    """

    @classmethod
    def apply(cls, segment, window_size=50, widths=50):
        """
                @param segment: signal segment.
                @param window_size: window size. Default: 50.
                @param widths: number of scale. Default: 50.
                """
        assert 0 <= window_size <= segment.shape[1], 'window_size={} is invalid!'.format(window_size)
        length = segment.shape[1]
        num_channels = segment.shape[0]
        if isinstance(widths, int):
            widths = np.arange(1, widths + 1)
        cwts = np.array([])
        for i in range(0, length, window_size):
            start = i
            end = i + window_size if (i + window_size <= length) else 2 * i + window_size - length
            cwt = np.array([])
            for j in range(num_channels):
                cwt_ = signal.cwt(segment[j, start: end], signal.ricker, widths, dtype=np.float64)
                if j == 0:
                    cwt = cwt_
                    continue
                cwt = np.dstack((cwt, cwt_))
            cwt = np.transpose(cwt, (2, 0, 1))
            if i == 0:
                cwts = cwt
            cwts = np.dstack((cwts, cwt))
        return cwts
