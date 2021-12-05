import matplotlib.pyplot as plt
import numpy as np


def plot_signal1d(x, fs=None, title='Signal'):
    """
    Plot the single-channel signal
    """
    plt.figure(figsize=(25, 5))
    if fs:
        t = np.arange(0, len(x) / fs, 1. / fs)
        plt.plot(t, x)
    else:
        plt.plot(x)
    plt.autoscale(tight=True)
    if fs:
        plt.xlabel('Time')
    else:
        plt.xlabel('Sample')
    plt.ylabel('Amplitude (mV)')
    plt.title(title)
    plt.show()


def plot_signalnd(x, fs=None, title='Signal'):
    """
    Plot the n-channel signal
    """
    t = None
    if fs:
        t = np.arange(0, len(x) / fs, 1. / fs)
    num_channels = len(x[0])
    fig, axs = plt.subplots(num_channels, 1, figsize=(25, 25))
    for i in range(num_channels):
        if fs:
            axs[i].plot(t, x[:, i])
            axs[i].set_xlabel('Time')
        else:
            axs[i].plot(x[:, i])
            axs[i].set_xlabel('Sample')
        axs[i].set_ylabel('Amplitude (mV)')
        axs[i].set_title(title + ' channel {}'.format(i + 1))
    plt.show()
