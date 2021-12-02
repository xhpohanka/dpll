import numpy as np
from scipy import signal
from matplotlib import pyplot as plt

from numpy import pi, fix
from scipy.interpolate import interp1d


def quant(v, b):
    return np.round(v * 2**b) / 2**b

def mhilbert(tap):
    ct = tap // 2

    hb = np.zeros(tap)
    hb[0:-1:2] = 1 / np.arange(-ct, ct, 2)
    hb = 2 / np.pi * hb
    hb = hb * signal.blackman(tap)
    hb = quant(hb, 24)

    hbd = np.zeros(ct+1)
    hbd[-1] = 1

    return (hb, hbd)

def psd_simple(x, nfft, fs):
    noverlap = nfft/2;
    window = signal.kaiser(nfft, 7)
    N = len(x)
    if N < nfft:
        window = signal.kaiser(N, 7)
        noverlap = 0

    noverlap = np.floor(noverlap / N)
    [f, pxx] = signal.welch(x, fs, window, noverlap=noverlap, nfft=nfft)
    PdB = 10 * np.log10(pxx * fs/nfft)

    return (PdB, f)

def psd(x, nfft, fs):
    (P, f) = psd_simple(x, nfft, fs)
    plt.plot(f, P)
    plt.grid(True)

def freqzz(b, a, **kwargs):
    (w, h) = signal.freqz(b, a, **kwargs)
    fig, axs = plt.subplots(2, num=10)
    axs[0].plot(w, 20*np.log10(abs(h)))
    axs[1].plot(w, np.unwrap(np.angle(h)))
