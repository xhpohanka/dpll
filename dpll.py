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
use_prescaler = False

# f_xtaln = 8e6
f_xtaln = 32e6
fref = 59.4
fs = 200
timbits = 12
fnco = 52
fmin = 35

iq = 20  # interpolation ratio

# zakladni frekvence MCU (zatizena chybou)
f_base = (f_xtaln * 1.00)
fs = f_base / (f_xtaln/fs) # nepresna vzorkovacka
Ts = 1/fs

# vstupni signal vyrobime s vyssi vzorkovackou a pak podsamplujeme
# podsamplovani potrebuje orez frekvenci nad Nyqitsem, ze ano a do toho si oriznem i zbytecny sum
# vnese nam to fazovej posuv, kterej se s FIR filtrem resi jednoduse
over_ratio = 5
inf_taps = 11
b_in, a_in = signal.firwin(inf_taps, 60, fs=(fs * over_ratio)), 1
# kompenzace fazovyho posunu od vstupniho filtru
f_shift = (inf_taps - 1) // 2 // over_ratio

# pokus s vstupnim filtrem IIR
# zatim nevim, jak se vyporadat s fazovym posunem
# napad pouzit stejnej filtr na signal NCO nevychazi, protoze tenhle filter je nadesignovanej na vyssi fs
# b_in, a_in = signal.ellip(5, 1, 50, [65], fs=(fs*over_ratio))
# inf_i = signal.lfilter_zi(b_in, a_in)
# f_shift = 0

# testovaci signal vygenerujem, nebo pouzijem nejakej zmerenej
if 1:
    N = int(5.5 / Ts)

    n = np.arange(N)
    t = n * Ts
    ti = np.arange(N * iq) * Ts / iq


    ff = np.ones(N * over_ratio) * fref
    fref_err = 1.1
    # ff[N * 4 // 5:] = fref * fref_err
    # ff[N * 3 // 5:N * 4 // 5] = np.linspace(1, fref_err, N*1//5) * fref

    phi = np.cumsum(2 * pi * ff/(fs*over_ratio))

    # ref signal
    A = .5
    Vro = np.sin(phi + pi/3)
    Vr = A * Vro + .05293585 * np.random.randn(len(phi))

    # vstupni signal je potreba podvzorkovat, ale predtim samozrejme oriznout nehezke frekvence
    Vrf = Vr
    Vrf = signal.lfilter(b_in, a_in, Vr)

    Vrf = Vrf[::over_ratio]
    Vr = Vr[::over_ratio]
    Vro = Vro[::over_ratio]

    Vr = quant(Vr, 10)
    Vrf = quant(Vrf, 10)

else:
    Vr = np.loadtxt('grid_200fs5.txt', delimiter=';')
    Vr = Vr[-1, :] * 2.5
    Vro = Vr
    # Vr = Vr - np.mean(Vr)
    Vr = Vr / 2**12
    # Vr = Vr / np.max(Vr) * 0.3
    N = len(Vr)
    A = Vr.max()

    n = np.arange(N)
    t = n * Ts
    ti = np.arange(N * 10) * Ts / 10
    iq = 10


def get_kx(Ts, fn, Knco, zeta=1.0, kp=2.0):
    # vypocet konstant, viz dsp related Part 2
    # KP = phase detector gain = 2
    # wn loop natural freq (rad/s), 2*pi * fn, fn
    # zeta = 1, loop damping
    # Knco - nco gain, typically << 1,

    wn = 2*pi * fn

    KL = 2*zeta*wn*Ts / (kp * Knco)
    KI = wn**2 * Ts**2 / (kp * Knco)

    return KL, KI

Knco = .1
fn = 1
kp = 1

KL, KI = get_kx(Ts, fn, Knco, kp=kp)


# KL = 0.1005 /1
# KI = 5.0532e-03
# KL = KL * 0.6
# KI = KI * 1.3

u = np.zeros(N)
y = np.zeros(N)
yq = np.zeros(N)
phase_error = np.zeros(N)
vtune = np.zeros(N)
timval = np.zeros(N)
timpsc  = np.zeros(N)
arr  = np.zeros(N)
pemsig  = np.zeros(N)

psc = np.fix(f_xtaln / 2**timbits / fmin)  # tady je potreba pouzit minimalni uvazovanou frekvenci
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
[bl, al] = signal.butter(6, 50, fs=fs)
# [bl, al] = signal.cheby1(10, 2, 50, fs=fs)
# [bl, al] = signal.ellip(4, 1, 100, 40, fs=fs)
# [bl, al] = signal.ellip(5, 0.5, 10, [80, 120], 'bandstop', fs=fs)
# [bl, al] = [signal.firwin(10, 40, fs=fs), 1]

# Vr = signal.filtfilt(bl, al, Vr)

# band stop
[bbs, a_bs] = signal.butter(11, np.array([60, 80]), 'bs', fs=fs)

# ewma
c = 20/fs
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
        timval_ = timval_ + fix(f_xtaln / fs / timpsc_)
        timval_ = np.mod(timval_, arr_)
        u_ = timval_ / arr_
        u_ = quant(u_, timbits)

    else:
        u_ = u_ + (fnco * Ts + vtune_ * Knco);
        u_ = np.mod(u_, 1)
        u_ = quant(u_, 23)

    # kvadraturni signal vnitrni frekvence
    Inco = np.sin(2 * pi * u_)
    Qnco = np.cos(2 * pi * u_)

    x = Vrf[n]

    y[n], yq[n] = Inco, Qnco
    Inco_act, Qnco_act = y[n - f_shift], yq[n - f_shift]

    if use_hilbert:
        # tohle proste neumim rozchodit tak, aby se to chytalo na nulovou fazi
        [Qr_, hbdi] = signal.lfilter(hbd, 1, [x], 0, hbdi)
        [Ir_, hbi]  = signal.lfilter(hb,  1, [x], 0, hbi)
        Qr_ = Qr_[0]
        Ir_ = Ir_[0]

        # Phase Detector
        pe = -Ir_ * Qnco_act + Qr_ * Inco_act

    else:
        pe = +x * Qnco_act
        #+ x * Inco

        # odfiltrovani druhe harmonicke
        [pe, lpi] = signal.lfilter(bl, al, [pe], 0, lpi)
        # [pe, bsi] = signal.lfilter(bbs, a_bs, [pe], 0, bsi)
        pe = pe[0]

    KIa = KI
    KLa = KL
    Kncoa = Knco

    # zmenseni zasahu vnitrni smycky po locknuti
    [pem, mai] = signal.lfilter(bma, ama, [pe], 0, mai)
    pem = pem[0]
    if n > 20:
        if np.abs(pem) < 0.005 or locked:
            # Kncoa = Knco / 2
            # KLa, KIa = get_kx(Ts, fn, Kncoa, kp=A)
            KLa, KIa = get_kx(Ts, fn / 8, Knco, kp=kp)
            # KLa, KIa = KL/10, KI/10

            if not locked:
                print(f'locked {n}, {n*Ts:0.2f}s')
                print(KIa, KLa)

            locked = True

        if np.abs(pem) > 0.02:
            KIa, KLa = KI, KL

            if locked:
                print(f'unlocked {n}, {n*Ts:0.2f}s')
                print(KIa, KLa)

            locked = False


    # Loop Filter
    integ_ = KIa*pe + integ_
    vtune_ = integ_ + KLa*pe
    vtune_ = vtune_ * Kncoa

    vtune_ = min(max(vtune_, -1), 1)

    # opravna freq
    ffix = vtune_ / Ts
    if use_prescaler:
        timpsc_ = fix(f_xtaln / arr_ / (fnco + ffix))
    else:
        arr_ = fix(f_xtaln / timpsc_ / (fnco + ffix))
        if arr_ > 2**timbits - 1:
            arr_ = 2**timbits - 1

        # arr_ = np.mod(arr_, 2**timbits)

    # sledovaci signaly si ulozime zvlast, at se nam neplete, co se uklada a co ne
    phase_error[n] = pe
    vtune[n] = vtune_
    u[n] = u_
    arr[n] = arr_
    timpsc[n] = timpsc_
    pemsig[n] = pem


# obnova signalu
# % u = timval / 2^timbits
dus = np.unwrap(u, period=1)
f = interp1d(t, dus, bounds_error=False)
dusi = f(ti)
Vrec  = np.sin(2 * pi * dus)  # sin nebo cos, podle toho, na co optimalizujem, I nebo Q
Vreci = np.sin(2 * pi * dusi) * A

# plt.close("all")
figs = list(map(plt.figure, plt.get_fignums()))
for f in figs:
    f.clear()

timpsc_mean = np.mean(timpsc[N//2:])
timpsc_var = np.var(timpsc[N//2:])
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
axs[0].set_title(f'f_ref = {fref:.2f}, f_s = {fs}')
# axs[0].set_xlim(np.take(t, t.size*0.9), t[-1])
axs[1].plot(t, vtune)
axs[1].plot(t, phase_error)
axs[1].plot(t, pemsig)
# axs[1].plot(t, (ff - fref) / (fref * fref_err - fref))
axs[1].grid('on')
axs[1].legend([f'vtune {vtune_mean:.6f}, {vtune_var:.6f}', 'pe', 'pem'])


Vroi = signal.resample(Vr, len(Vr)*iq)

plt.figure(3)
plt.plot(t, Vro)
plt.plot(ti, Vroi, 'x')
plt.plot(t, Vrec)
plt.plot(ti, Vreci)
plt.plot(ti, np.mod(dusi, 1), '--')
# plt.plot(t, yy)
plt.gca().set_xlim(np.take(t, t.size*0.99), t[-1])
# plt.gca().set_xlim(t[0], np.take(t, t.size*0.05))
plt.legend(['Vr', 'Vroi', 'Vrec', 'Vreci', 'y'])


plt.figure(4)
plt.plot(ti, Vroi)
plt.plot(ti, Vreci)
plt.plot(ti, Vroi - Vreci)
plt.plot(t, phase_error)
plt.plot(t, pemsig, '-')
plt.plot(t, np.diff(pemsig, prepend=pemsig[0]), '-')
plt.legend(['Vroi', 'Vreci', 'amplitude error', 'pe', 'pem'])


plt.figure(5)
plt.plot(t, arr)
# plt.plot(t, timpsc)
plt.legend(['arr', 'psc'])


step = 1/fref/2 / 256
t_uncert = 1 / (f_xtaln/psc) * np.sqrt(arr_var)


print(f'arr mean {arr_mean}, arr var {arr_var}, arr sd {np.sqrt(arr_var)}, var/mean {arr_var/arr_mean}')
print(f'time uncertainity {t_uncert:.3e} ~ {t_uncert/step:.2f} of minimal dimmer step')
