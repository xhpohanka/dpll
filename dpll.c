#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include <stdio.h>

#define USE_PSC

#define F_PLL       32e6

#define PI          ((float) M_PI)
#define F_NCO       55
#define TIM_BITS    10
#define ADC_MAX     (1 << 12)
#define F_S         256

#if F_S == 256
# define KL          0.07853981633974483
# define KI          0.003084251375340424 // fs = 256
#elif F_S == 2560
# define KI          3.0842513753404247e-05
# define KL          0.007853981633974483 // fs = 2560
#endif


struct filter_desc {
    const float *b;
    const float *a;
    const int len;
    float *w;
};

// dolni propust butterworth, 4rad, 60Hz, fs = 256
static struct filter_desc low_pass = {
#if F_S == 256
    .b = (float []){ 0.07673978, 0.30695911, 0.46043867, 0.30695911, 0.07673978 },
    .a = (float []){ 1.0f, -0.2440988, 0.50486789, -0.05175419, 0.01882155 },
#elif F_S == 2560
    .b = (float []){ 2.44195653e-05, 9.76782612e-05, 1.46517392e-04, 9.76782612e-05, 2.44195653e-05 }, // fs = 2560
    .a = (float []){ 1., -3.6153527 ,  4.91831572, -2.98287135,  0.68029904 },
#endif
    .len = 5,
    .w = (float [5]) {}
};

// jednoducha implementace IIR v direct form II transposed
float filter_df2t(struct filter_desc *fd, float x)
{
    float y = fd->b[0] * x + fd->w[0];
    for (int i = 0; i < fd->len - 1; i++) {
        fd->w[i] = fd->b[i + 1] * x + fd->w[i + 1] - fd->a[i + 1] * y;
    }

    return y;
}

float dpll_step(int sample, int timval, int reload_val)
{
    static float integ = 0;
    static float avg = 0;
    static int locked = 0;
    float ki = KI;
    float kl = KL;

    float x = (float) sample / ADC_MAX;
    float phase = (float) timval / reload_val;

    //float Itim = sinf(2 * PI * phase);
    float Qtim = cosf(2 * PI * phase);

    // phase detector
    float pe = x * Qtim;
    pe = filter_df2t(&low_pass, pe);

    // moving average of pe (ewma)
    // if small enough we can reduce action
    float k1 = 20.0 / F_S;
    avg = k1 * pe + (1 - k1) * avg;

    // bacha nechodi na vyssich fs
    if (fabsf(avg) < 0.001) {
        if (locked > 5) {
            ki = ki / 5;
            kl = kl / 5;
        } else {
            locked++;
        }
    }
    else if (fabsf(avg) > 0.01) {
        locked = 0;
    }

    // PLL loop filter
    integ = ki * pe + integ;
    float tune = integ + kl * pe;

    return tune;
}

void dpll_fix(float tune, int *psc, int *arr)
{
    float ffix = tune * F_S;
#if defined(USE_PSC)
    *psc = roundf(F_PLL / *arr / (F_NCO + ffix));
#else
    *arr = roundf(F_PLL / *psc / (F_NCO + ffix));
#endif
}

static int timer(int fs, int arr, int psc)
{
    static int val = 0;
    int v = val;

    val += F_PLL / fs / psc;
    val = val % arr;

    return v;
}

int main()
{
    int psc;
    int arr;

    psc = F_PLL / F_NCO / (1 << TIM_BITS);
    if (psc < 1)
        psc = 1;
    arr = F_PLL / F_NCO / psc;

    for (int i = 0; i < 200 * F_S / F_NCO; ++i) {
        int t = timer(F_S, arr, psc);
        float s = 0.85 * sinf(2 * PI * i / (float) F_S * 65 + 2.75);
        s += 0.02 * rand() / RAND_MAX;

        // simulace chyby
        if (i == 100 * F_S / F_NCO)
            for (int k = 0; k < 40; k++)
                timer(F_S, arr, psc);

        float tune = dpll_step(s * ADC_MAX, t, arr);

        // bacha rekonstruovat musme pred opravou timeru
        float s_rec = 1 * sinf(2 * PI * t / arr);

        dpll_fix(tune, &psc, &arr);

        printf("%.3f %.3f %d %d %.3f\n",
               s * 1000, tune * 1000, psc, arr, s_rec * 1000);
    }

    return 0;
}
