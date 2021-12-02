from IPython import get_ipython
get_ipython().magic('reset -sf')

import numpy as np
from scipy import signal
from matplotlib import pyplot as plt

from numpy import pi, fix
from scipy.interpolate import interp1d

from dpll_fns import *

## zajimave cteni
# https://www.dsprelated.com/showarticle/1177.php
# https://www.dsprelated.com/showarticle/1147.php
# https://www.dsprelated.com/showarticle/973.php
# https://www.dsprelated.com/showarticle/967.php
# https://www.ti.com/lit/an/sprabt3a/sprabt3a.pdf?ts=1632381751458

plt.ion()

use_hilbert = False
use_timer = True
use_prescaler = True

f_xtaln = 32e6
fref = 48.6
fs = 256

# zakladni frekvence MCU (zatizena chybou)
f_base = (f_xtaln * 1.00)
fs = f_base / (f_xtaln/fs) # nepresna vzorkovacka
Ts = 1/fs

# testovaci signal vygenerujem, nebo pouzijem nejakej zmerenej
if 1:
    N = int(5.5 / Ts)

    n = np.arange(N)
    t = n * Ts
    ti = np.arange(N * 10) * Ts / 10

    # ref signal
    A = 0.85
    Vr = A * np.sin(2*pi*fref*n*Ts) + .01585 * np.random.randn(N)
    # Vr = 0.99 * np.sin(2*pi*fref*n*Ts)
    Vr = quant(Vr, 12)

else:
    Vr = np.loadtxt('grid_200fs5.txt', delimiter=';')
    Vr = Vr[-1, :]
    # Vr = Vr - np.mean(Vr)
    Vr = Vr / 2**12
    # Vr = Vr / np.max(Vr) * 0.3
    N = len(Vr)

    n = np.arange(N)
    t = n * Ts
    ti = np.arange(N * 10) * Ts / 10

# Knco= 1/4096
Knco= 1/(2**5)
Knco= 1

# vypocet konstant, viz dsp related Part 2
KP = 2
wn = 2 * pi * 3.2
zeta = 1
KL = 2*zeta*wn*Ts / (KP * Knco)
KI = wn**2 * Ts**2 / (KP * Knco)

# KL = 0.1005 /1
# KI = 5.0532e-03
# KL = KL * 0.6
# KI = KI * 1.3

fnco = fref * (1+0.05)
fnco = 55*1

timbits = 9

u = np.zeros(N)
y = np.zeros(N)
yq = np.zeros(N)
phase_error = np.zeros(N)
vtune = np.zeros(N)
timval = np.zeros(N)
timpsc  = np.zeros(N)
arr  = np.zeros(N)

psc = fix(f_xtaln / 2**timbits / fnco)
timpsc[:] = 1 if psc == 0 else psc
arr[:] = fix(f_xtaln / timpsc[0] / fnco)
timval = fix(f_xtaln / fs / timpsc[0]) * (n + 0)
timval = np.mod(timval, 2**timbits)
u = timval / 2**timbits

# hilbert transform, pro realnou implementaci musi mit i zpozdovaci cast
# resp abych dostal slozky analytickeho signalu, tak si musim ten vstupni
# zpozdit stejne, jako zpozduje ten samotnej hb
(hb, hbd) = mhilbert(15)

# low pass
[bl, al] = signal.butter(5, 55, fs=fs)
# [bl, al] = signal.cheby1(10, 2, 65, fs=fs)
# [bl, al] = [signal.firwin(8, 75, fs=fs), 1]

# band stop
[bbs, a_bs] = signal.butter(11, np.array([60, 90]), 'bs', fs=fs)

# ewma
c = 0.01
[bma, ama] = [[c], [1, c - 1]]


# filter states
hbi = signal.lfilter_zi(hb, 1)
hbdi = signal.lfilter_zi(hbd, 1)
lpi = signal.lfilter_zi(bl, al)
lpi2 = signal.lfilter_zi(bl, al)
lpi3 = signal.lfilter_zi(bl, al)
bsi = signal.lfilter_zi(bbs, a_bs)
mai = signal.lfilter_zi(bma, ama)
mai2 = signal.lfilter_zi(bma, ama)

locked = False

# okamzite/minule hodnoty
vtune_ = 0
timval_ = 0
timpsc_ = timpsc[0]
arr_ = arr[0]
u_ = 0  # faze nco
integ_ = 0
Qr_ = 0
Ir_ = 0

for n in np.arange(N):
    if use_timer:
        # opravna freq
        ffix = vtune_ / Ts
        if use_prescaler:
            timpsc_ = round(f_xtaln / arr_ / (fnco + ffix))
        else:
            arr_ = round(f_xtaln / timpsc_ / (fnco + ffix))

        timval_ = timval_ + fix(f_xtaln / fs / timpsc_)
        timval_ = np.mod(timval_, arr_)
        u_ = timval_ / arr_

    else:
        vtune_ = quant(vtune_, 12)
        u_ = u_ + (fnco * Ts + vtune_ * Knco);
        u_ = np.mod(u_, 1)

    # kvadraturni signal vnitrni frekvence
    Inco = np.sin(2 * pi * u_)
    Qnco = np.cos(2 * pi * u_)

    if use_hilbert:
        # tohle proste neumim rozchodit tak, aby se to chytalo na nulovou fazi
        [Qr_, hbdi] = signal.lfilter(hbd, 1, [Vr[n]], 0, hbdi)
        [Ir_, hbi]  = signal.lfilter(hb,  1, [Vr[n]], 0, hbi)
        Qr_ = Qr_[0]
        Ir_ = Ir_[0]

        # Phase Detector
        pe = -Ir_ * Qnco + Qr_ * Inco

    else:
        pe = +Vr[n] * Qnco
        #+ Vr[n] * Inco

        # odfiltrovani druhe harmonicke
        [pe, lpi] = signal.lfilter(bl, al, [pe], 0, lpi)
        # [pe, bsi] = signal.lfilter(bbs, a_bs, [pe], 0, bsi)
        pe = pe[0]

    KIa = KI
    KLa = KL

    # zmenseni zasahu vnitrni smycky po locknuti
    [pem, mai] = signal.lfilter(bma, ama, [pe], 0, mai)
    pem = pem[0]
    if n > 20:
        if np.abs(pem) < 1e-3 or locked:
            if not locked:
                print(f'locked {n}, {n*Ts:0.2f}s')

            locked = True
            KIa = KI / 5
            KLa = KL / 5

        if np.abs(pem) > 0.1:
            if locked:
                print(f'unlocked {n}, {n*Ts:0.2f}s')
            locked = False


    # Loop Filter
    integ_ = KIa*pe + integ_
    vtune_ = integ_ + KLa*pe

    vtune_ = min(max(vtune_, -1), 1)

    # sledovaci signaly si ulozime zvlast, at se nam neplete, co se uklada a co ne
    phase_error[n] = pe
    vtune[n] = vtune_
    u[n] = u_
    arr[n] = arr_
    timpsc[n] = timpsc_
    y[n] = Inco
    yq[n] = Qnco


# obnova signalu
# % u = timval / 2^timbits
du = np.diff(u, prepend=u[0])
# du = np.diff(u, append=u[-1])
du[du < 0] = 1 + du[du < 0] # oprava preteceni
dus = np.cumsum(du) + u[0] # ta predchozi diferenciace odstranila vychozi hodnotu
f = interp1d(t, dus, bounds_error=False)
dusi = f(ti)
Vrec  = np.sin(2 * pi * dus)  # sin nebo cos, podle toho, na co optimalizujem, I nebo Q
Vreci = np.sin(2 * pi * dusi)

# plt.close("all")
figs = list(map(plt.figure, plt.get_fignums()))
for f in figs:
    f.clear()

arr_mean = np.mean(arr[N//2:])
arr_var = np.var(arr[N//2:])
vtune_mean = np.mean(vtune[N//2:])
vtune_var = np.var(vtune[N//2:])

plt.figure(1)
psd(Vr, 1024*2, fs)
psd(y[N//2:], 1024*2, fs)
# psd(Vr[0:N//4], 512, fs)
plt.legend(['Vr', 'y'])
plt.title(f'hilbert = {use_hilbert}')

fig, axs = plt.subplots(2, num=2)
axs[0].plot(t, Vr)
shift = 1
# axs[0].plot(t[shift:], y[:-shift])
axs[0].plot(t, y)
axs[0].legend(['Vr', 'y'])
axs[0].set_xlim(t[0], np.take(t, t.size*0.08))
# axs[0].set_xlim(np.take(t, t.size*0.9), t[-1])
axs[1].plot(t, vtune)
axs[1].plot(t, phase_error)
axs[1].grid('on')
axs[1].legend([f'vtune {vtune_mean}, {vtune_var}', 'error'])


plt.figure(3)
plt.plot(t, Vr)
plt.plot(t, Vrec)
plt.plot(ti, Vreci)
plt.plot(t, np.mod(dus, 1), 'x--')
# plt.plot(t, yy)
plt.gca().set_xlim(np.take(t, t.size*0.99), t[-1])
# plt.gca().set_xlim(t[0], np.take(t, t.size*0.05))
plt.legend(['Vr', 'Vrec', 'Vreci', 'y'])

print(arr_mean, arr_var, arr_var/arr_mean)
