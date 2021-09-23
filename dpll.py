import numpy as np
from scipy import signal
from matplotlib import pyplot as plt

from numpy import pi, fix
from scipy.interpolate import interp1d

## zajimave cteni
# https://www.dsprelated.com/showarticle/1177.php
# https://www.dsprelated.com/showarticle/1147.php
# https://www.dsprelated.com/showarticle/973.php
# https://www.dsprelated.com/showarticle/967.php
# https://www.ti.com/lit/an/sprabt3a/sprabt3a.pdf?ts=1632381751458

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


hilbert = False

f_xtaln = 32e6
fref = 62.8

# zakladni frekvence MCU (zatizena chybou)
f_base = (f_xtaln * 1.00)
fs = f_base / (f_xtaln/0.341e3) # nepresna vzorkovacka
Ts = 1/fs

N = int(5.5 / Ts)

n = np.arange(N)
t = n * Ts
ti = np.arange(N * 10) * Ts / 10

(hb, hbd) = mhilbert(15)

# ref signal
A = 0.65
Vr = A * np.sin(2*pi*fref*n*Ts) + .01585 * np.random.randn(N)
Vr = 0.99 * np.sin(2*pi*fref*n*Ts)

Vr = quant(Vr, 10)

# Knco= 1/4096
Knco= 1/(2**5)
Knco= 1

# vypocet konstant, viz dsp related Part 2
KP = 2
wn = 2 * pi * 3.2
zeta = 1
KL = 2*zeta*wn*Ts / (KP * Knco)
KI = wn**2 * Ts**2 / (KP * Knco)

KL = 0.1005
KI = 5.0532e-03

fnco = fref * (1+0.05)
fnco = 55*1

timbits = 12

u = np.zeros(N)
y = np.zeros(N)
yy = np.zeros(N)
integ = np.zeros(N)
phase_error = np.zeros(N)
vtune = np.zeros(N)
timval = np.zeros(N)
timpsc  = np.zeros(N)
Ir  = np.zeros(N)
Qr  = np.zeros(N)

timpsc[:] = fix(f_xtaln / 2**timbits / fnco)
timval = fix(f_xtaln / fs / timpsc[0]) * (n+1)
timval[0] = 3000  # rekonstruovany signal zavisi na vychozi hodnote timeru, coz uplne nechapu
timval = np.mod(timval, 2**timbits)
u = timval / 2**timbits

# low pass
[bl, al] = signal.butter(4, 60 / (fs / 2))

# filter states
hbi = signal.lfilter_zi(hb, 1)
hbdi = signal.lfilter_zi(hbd, 1)
bli = signal.lfilter_zi(bl, al)

for n in np.arange(1, N):

    timval[n] = timval[n - 1] + fix(f_xtaln / fs / timpsc[n - 1])
    timval[n] = np.mod(timval[n], 2**timbits)

    u[n] = timval[n] / 2**timbits

    # kvadraturni signal vnitrni frekvence
    Inco = np.cos(2 * pi * (u[n - 0]))
    Qnco = np.sin(2 * pi * (u[n - 0]))

    y[n] = Inco
    yy[n] = Qnco

    if hilbert:
        [Ir[n], hbdi] = signal.lfilter(hbd, 1, [Vr[n - 1]], 0, hbdi)
        [Qr[n], hbi]  = signal.lfilter(hb,  1, [Vr[n - 1]], 0, hbi)

        # Phase Detector
        pe = -Ir[n - 1] * Qnco + Qr[n - 1] * Inco

    else:
        pe = Vr[n - 1] * Qnco

        # odfiltrovani druhe harmonicke
        [pe, bli] = signal.lfilter(bl, al, [pe], 0, bli)

    phase_error[n] = pe

    # Loop Filter
    integ[n] = KI*pe + integ[n - 1]
    vtune[n] = integ[n] + KL*pe

    vtune[n] = min(max(vtune[n], -1), 1)

    # kvantovat se tady da az nekde k 6 bitum
    vtune[n] = quant(vtune[n], 7)

    # opravna freq
    ffix = vtune[n] / Ts

    timpsc[n] = round(f_xtaln / 2**timbits / (fnco + ffix))

    # simulace zpozdeni v realnem hw
    # vypada to, ze ji vtune kompenzuje
    # timval[n] += (600 * 32 / timpsc[n - 1])

# obnova signalu
# % u = timval / 2^timbits
du = np.diff(u, append=u[-1])
du = np.diff(u, prepend=u[0])
du[du < 0] = 1 + du[du < 0] # oprava preteceni
dus = np.cumsum(du)
f = interp1d(t, dus, bounds_error=False)
dusi = f(ti)
Vrec  = np.cos(2 * pi * dus)
Vreci = np.cos(2 * pi * dusi)

# plt.close("all")
figs = list(map(plt.figure, plt.get_fignums()))
for f in figs:
    f.clear()

plt.figure(1)
psd(Vr, 1024, fs)
psd(y[N//2:-1], 1024, fs)

fig, axs = plt.subplots(2, num=2)
axs[0].plot(t, Vr)
axs[0].plot(t[1:-2], y[2:-1])
# axs[0].set_xlim(np.take(t, t.size*0.9), t[-1])
axs[0].set_xlim(t[0], np.take(t, t.size*0.05))
axs[1].plot(t, vtune)
axs[1].plot(t, phase_error)


plt.figure(3)
plt.plot(t, Vr)
plt.plot(t, Vrec)
plt.plot(ti, Vreci)
plt.gca().set_xlim(np.take(t, t.size*0.99), t[-1])
plt.legend(['Vr', 'Vrec', 'Vreci'])

