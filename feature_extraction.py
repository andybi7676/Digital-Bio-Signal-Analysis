import numpy as np
from scipy import stats, signal


class MaxPeak:
    @classmethod
    def apply(cls, segment, window_size=50):
        """
        @param segment: signal segment
        @param window_size: window size. Default: 50
        """
        assert window_size <= 0 or window_size > segment.shape[1], 'window_size={} is invalid!'.format(window_size)
        length = segment.shape[1]
        peaks = np.array([])
        for i in range(0, length, window_size):
            start = i
            end = i + window_size if (i + window_size <= length) else 2 * i + window_size - length
            peak = np.max(segment[:, start: end], axis=0)
            peaks = np.append(peaks, peak)
        return peaks


class Mean:
    @classmethod
    def apply(cls, segment, window_size=50):
        """
        @param segment: signal segment
        @param window_size: window size. Default: 50
        """
        assert window_size <= 0 or window_size > segment.shape[1], 'window_size={} is invalid!'.format(window_size)
        length = segment.shape[1]
        mns = np.array([])
        for i in range(0, length, window_size):
            start = i
            end = i + window_size if (i + window_size <= length) else 2 * i + window_size - length
            mn = np.mean(segment[:, start: end], axis=0)
            mns = np.append(mns, mn)
        return mns


class Variance:
    @classmethod
    def apply(cls, segment, window_size=50):
        """
        @param segment: signal segment
        @param window_size: window size. Default: 50.
        """
        assert window_size <= 0 or window_size > segment.shape[1], 'window_size={} is invalid!'.format(window_size)
        length = segment.shape[1]
        vars = np.array([])
        for i in range(0, length, window_size):
            start = i
            end = i + window_size if (i + window_size <= length) else 2 * i + window_size - length
            var = np.var(segment[:, start: end], axis=0)
            vars = np.append(vars, var)
        return vars


class StandardDeviation:
    @classmethod
    def apply(cls, segment, window_size=50):
        """
        @param segment: signal segment
        @param window_size: window size. Default: 50.
        """
        assert window_size <= 0 or window_size > segment.shape[1], 'window_size={} is invalid!'.format(window_size)
        length = segment.shape[1]
        stds = np.array([])
        for i in range(0, length, window_size):
            start = i
            end = i + window_size if (i + window_size <= length) else 2 * i + window_size - length
            std = np.std(segment[:, start: end], axis=0)
            stds = np.append(stds, std)
        return stds


class Skewness:
    @classmethod
    def apply(cls, segment, window_size=50):
        """
        @param segment: signal segment
        @param window_size: window size. Default: 50.
        """
        assert window_size <= 0 or window_size > segment.shape[1], 'window_size={} is invalid!'.format(window_size)
        length = segment.shape[1]
        skews = np.array([])
        for i in range(0, length, window_size):
            start = i
            end = i + window_size if (i + window_size <= length) else 2 * i + window_size - length
            skew = stats.skew(segment[:, start: end], axis=0)
            skews = np.append(skews, skew)
        return skews


class Kurtosis:
    @classmethod
    def apply(cls, segment, window_size=50):
        """
        @param segment: signal segment
        @param window_size: window size. Default: 50.
        """
        assert window_size <= 0 or window_size > segment.shape[1], 'window_size={} is invalid!'.format(window_size)
        length = segment.shape[1]
        kurts = np.array([])
        for i in range(0, length, window_size):
            start = i
            end = i + window_size if (i + window_size <= length) else 2 * i + window_size - length
            kurt = stats.kurtosis(segment[:, start: end], axis=0)
            kurts = np.append(kurts, kurt)
        return kurts


class RootMeanSquare:
    @classmethod
    def apply(cls, segment, window_size=50):
        """
        @param segment: signal segment
        @param window_size: window size. Default: 50.
        """
        assert window_size <= 0 or window_size > segment.shape[1], 'window_size={} is invalid!'.format(window_size)
        length = segment.shape[1]
        rmss = np.array([])
        for i in range(0, length, window_size):
            start = i
            end = i + window_size if (i + window_size <= length) else 2 * i + window_size - length
            rms = np.sqrt(np.mean(segment[:, start: end] ** 2, axis=0))
            rmss = np.append(rmss, rms)
        return rmss


class WaveformLength:
    @classmethod
    def apply(cls, segment, window_size=50):
        """
        @param segment: signal segment
        @param window_size: window size. Default: 50.
        """
        assert window_size <= 0 or window_size > segment.shape[1], 'window_size={} is invalid!'.format(window_size)
        length = segment.shape[1]
        wls = np.array([])
        for i in range(0, length, window_size):
            start = i
            end = i + window_size if (i + window_size <= length) else 2 * i + window_size - length
            wl = np.sum(np.abs(np.diff(segment[:, start: end], axis=0)), axis=0)
            wls = np.append(wls, wl)
        return wls


class WillisonAmplitude:
    @classmethod
    def apply(cls, segment, window_size=50, eps=0.5):
        """
        @param segment: signal segment
        @param window_size: window size. Default: 50.
        @param eps: threshold value. Default: 0.5.
        """
        assert window_size <= 0 or window_size > segment.shape[1], 'window_size={} is invalid!'.format(window_size)
        if eps > 1:
            eps = 1
        elif eps < 0:
            eps = 0
        length = segment.shape[1]
        wamps = np.array([])
        for i in range(0, length, window_size):
            start = i
            end = i + window_size if (i + window_size <= length) else 2 * i + window_size - length
            wl = np.abs(np.diff(segment[:, start: end], axis=0))
            mask = wl[wl > eps]
            wl[mask is True] = 1
            wl[mask is False] = 0
            wamp = np.sum(wl, axis=0)
            wamps = np.append(wamps, wamp)
        return wamps


class PSD:
    """
    Power Spectral Density
    """

    @classmethod
    def apply(cls, segment, window_size=50, fs=512, min_fs=0, average='mean'):
        """
        @param segment: signal segment.
        @param window_size: window size. Default: 50.
        @param fs: sampling rate. Default: 512.
        @param min_fs: minimum frequency. Default: 0.
        @param average: 'mean' or 'median'. Default: 'mean'
        """
        assert window_size <= 0 or window_size > segment.shape[1], 'window_size={} is invalid!'.format(window_size)
        if min_fs == 0:
            min_fs = 0.001
        if window_size is not None:
            nperseg = int(window_size * fs)
        else:
            nperseg = int((2 / min_fs) * fs)
        if nperseg > len(segment) / 2:
            nperseg = int(len(segment) / 2)
        length = segment.shape[1]
        psds = np.array([])
        for i in range(0, length, window_size):
            start = i
            end = i + window_size if (i + window_size <= length) else 2 * i + window_size - length
            freq, power = signal.welch(segment[:, start: end],
                                       fs=fs,
                                       scaling='density',
                                       detrend=False,
                                       nfft=int(nperseg * 2),
                                       average=average,
                                       nperseg=nperseg,
                                       return_onesided=True,
                                       axis=0)
            psds = np.append(psds, power)
        return psds


class STFT:
    """
    Short Time Frequency Transform
    """

    @classmethod
    def apply(cls, segment, window_size=50, fs=512, min_fs=0):
        """
                @param segment: signal segment.
                @param window_size: window size. Default: 50.
                @param fs: sampling rate. Default: 512.
                @param min_fs: minimum frequency. Default: 0.
                """
        assert window_size <= 0 or window_size > segment.shape[1], 'window_size={} is invalid!'.format(window_size)
        if min_fs == 0:
            min_fs = 0.001
        if window_size is not None:
            nperseg = int(window_size * fs)
        else:
            nperseg = int((2 / min_fs) * fs)
        if nperseg > len(segment) / 2:
            nperseg = int(len(segment) / 2)
        length = segment.shape[1]
        stfts = np.array([])
        for i in range(0, length, window_size):
            start = i
            end = i + window_size if (i + window_size <= length) else 2 * i + window_size - length
            freq, t, stft = signal.stft(segment[:, start: end],
                                        fs=fs,
                                        detrend=False,
                                        nfft=int(nperseg * 2),
                                        nperseg=nperseg,
                                        return_onesided=True,
                                        axis=0)
            stfts = np.append(stfts, stft)
        return stfts


class CWT:
    """
    Continuous Wavelet Transform
    """

    @classmethod
    def apply(cls, segment, window_size=50, width=50):
        """
                @param segment: signal segment.
                @param window_size: window size. Default: 50.
                @param width: frequency width. Default: 50.
                """
        assert window_size <= 0 or window_size > segment.shape[1], 'window_size={} is invalid!'.format(window_size)
        length = segment.shape[1]
        cwts = np.array([])
        for i in range(0, length, window_size):
            start = i
            end = i + window_size if (i + window_size <= length) else 2 * i + window_size - length
            cwt = signal.cwt(segment[:, start: end], signal.ricker, width, axis=0)
            cwts = np.append(cwts, cwt)
        return cwts
