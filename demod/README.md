# `demod/` — coherent binary-pulsar demodulation

This directory searches a **single** PRESTO `.fft` observation for a pulsar in a
binary system (specifically a double-neutron-star, NS–NS, orbit) by
**demodulating** — undoing — the line-of-sight (LOS) acceleration the orbit
imposes on the spin period. A coherent harmonic sum needs the pulse phase to stay
coherent over the whole observation; an orbiting pulsar's period is stretched and
squeezed by the LOS Doppler, which destroys that coherence unless it is removed
first.

The pipeline is:

1. **Derive a grid** of trial orbital/kinematic parameters
   (`nsns_grid.py`) that is dense enough to phase-connect the orbit to a chosen
   tolerance.
2. **Demodulate** the time series at every grid point
   (`demod_grid.jl` → `demod_dat.jl` → `resample.jl`).
3. **Search** each demodulated series for the pulsar
   (`coherent_search.jl`, not part of this directory).
4. **Combine** the per-point candidate lists (`combine_cohout.py`).

`nsns_grid.py` has three grid models, selected with `-mode`:

* **`-mode ajs`** (default): an accel/jerk/snap **polynomial** grid. The orbit's
  LOS velocity is approximated by a cubic over the observation; the grid covers
  the cubic coefficients. This is the general case — it works for eccentric and
  relativistic orbits, but the cubic cannot track the true orbit exactly, so
  there is a residual ("truncation") phase error that must fit inside the budget.
* **`-mode circular`**: a pure-Keplerian **circular-orbit** grid in
  `(ω_b, x, A_T)`. The circular demod removes the exact non-polynomial Roemer
  track, so at the true parameters the residual is zero and there is no
  truncation gate; the only error is grid-point mismatch.
* **`-mode hybrid`**: circular below `p_break` and ajs at/above it, combining the
  two regimes (Section 6). The exact circular demod is the better short-orbit
  search; the cubic is the general, cheaper long-orbit search. `p_break` is
  derived from where the ajs coverage collapses, or set with `-p_break`.

  The ajs coverage is **not monotonic** in `p_o`: it dips near `p_o ~ T_obs`
  (one orbit per observation — the cubic's worst case), then recovers for long
  orbits (a slow arc a cubic tracks easily) and is high for very short ones (the
  integrated velocity excursion is bounded by `a_p sin i`, small there). The
  switch therefore has to clear the whole inadequate band: `p_break` is the
  shortest scanned `p_o` such that every `p_o` at/above it has mean `e=0` ajs
  coverage ≥ `-break_coverage` (default 60%), i.e. the next ladder point above
  the largest still-inadequate `p_o`.

---

## 1. Notation

Everything below uses these symbols. Code names are in `monospace`.

| symbol | code | meaning |
|---|---|---|
| $c$ | `C` | speed of light, $299792458\ \mathrm{m\,s^{-1}}$ |
| $G$ | `G` | gravitational constant, $6.67430\times10^{-11}$ |
| $M_\odot$ | `SOLAR_MASS` | solar mass, $1.989\times10^{30}\ \mathrm{kg}$ |
| $T$ | `t_obs` | observation length $= N\,dt$ [s] |
| $N$ | `N` | number of time-series bins (from `.inf`) |
| $dt$ | `dt` | time-series bin width [s] (from `.inf`) |
| $p_0$ | `p0` | pulsar spin period [s] (use the **minimum** of the search range) |
| $\dot p$ | `pdot` | intrinsic period derivative [s/s] (in the truth) |
| $\tau$ | `tau` | time measured from the anchor: $\tau = t-t_\text{anchor}$ |
| $p_o$ | `p_o` | orbital period [yr] (scanned; internally $p_{o,s}=p_o\,Y\!E\!A\!R_S$ [s]) |
| $e$ | `e` | orbital eccentricity (scanned) |
| $\sin i$ | `sin_i` | sine of orbital inclination (scanned) |
| $\omega_\text{peri}$ | `omega_peri` | argument of periastron [rad] (scanned) |
| $A_T$ | `A_T` | mean anomaly at epoch [rad] (fixed at 0 in the ajs scan) |
| $m_c$ | `companion_mass` | companion mass [$M_\odot$], default 1.4 |
| $m_p$ | `pulsar_mass` | pulsar mass [$M_\odot$], default 1.4 |
| $M$ | | total mass $m_c+m_p$ [$M_\odot$] |
| $a_\text{tot}$ | | total semi-major axis [m] |
| $a_p$ | | pulsar's semi-major axis [m] |
| $a_p\sin i$ | `a_psini` | projected pulsar semi-major axis [m] |
| $x$ | `x` | $a_p\sin i/c$ [light-seconds] (circular mode) |
| $\omega_b$ | `omega_b` | orbital angular frequency $2\pi/p_{o,s}$ [rad/s] |
| $v_\text{los}$ | `v_los` | LOS orbital velocity [m/s] |
| $v_\text{true}$ | `true_velocity` | observed LOS velocity [m/s] |
| $v_\text{model}$ | | trial cubic LOS velocity [m/s] |
| $v_0$ | `v0` | constant velocity term (absorbed by the period search) |
| $a,j,s$ | `a,j,s` | LOS accel [m/s²], jerk [m/s³], snap [m/s⁴] |
| $\varepsilon$ | `phase_tol_cycles` | total phase-drift budget [cycles], default 0.1 |
| $\varepsilon_m$ | `phase_tol_m` | the same budget as a path length [m], $\varepsilon_m=\varepsilon\,c\,p_0$ |
| $\Lambda$ | `trunc_phase` | the segment's own integrated truncation [m] |
| $\Lambda_\text{max}$ | `trunc_max` | worst kept segment's truncation [m] |
| $B_m$ | `budget_m` | budget left for grid mismatch [m], $B_m=\varepsilon_m-\Lambda_\text{max}$ |
| $L_a,L_j,L_s$ | `PHASE_LEVERS` | lever arms [s², s³, s⁴] (Section 4.5) |
| $D_x$ | `spacings` | grid spacing on axis $x$ |
| $R_x=[\text{lo}_x,\text{hi}_x]$ | `ranges` | coefficient range the grid must cover |
| $\epsilon_x$ | `eps` | phase share (m) granted to axis $x$ |
| | `mask` | which of accel/jerk/snap are in the model (`-terms`) |
| | `active` | which axes span more than one cell (not saturated) |
| $p_\text{break}$ | `p_break` | hybrid: period [yr] splitting circular (<) from ajs (≥) |

The default scan parameters (from `build_parser`, `main`, and the module
constants):

```
-p0              1.0                 (use the smallest period searched)
-pdot            1e-15
-p_o             12 log-spaced points from 2 min to 1 day [yr]
-e               0.0 0.1 0.3 0.5 0.7 0.9
-sin_i           0.1 0.3 0.5 0.7 0.9
-omega_peri      0  pi/8  pi/4  3pi/8  pi/2
-A_T             0.0
-companion_mass  1.4   -pulsar_mass 1.4   (GR on)
-phase_tol_cycles 0.1
-drop_pct        10.0
-n_phase         120     -nsamp 4000     -n_mc 300
-anchor          midpoint
-mode            ajs     (ajs | circular | hybrid)
-break_coverage  60.0    (hybrid auto-p_break threshold [%])
-p_break         None    (hybrid: manual switch period [yr])
```

> The module docstring says "30 minutes out to 10 days"; that is stale. The
> actual default ladder is 2 minutes to 1 day, as set by `_NSNS_PO_MIN_D`,
> `_NSNS_PO_MAX_D` and matching the `-p_o` help text.

---

## 2. The physical model

### 2.1 Observed period and LOS velocity

An orbiting pulsar's pulses arrive at

$$
t_\text{arr}(t) \;=\; t + \Delta(t),
$$

where $t$ is the pulse's proper (emission) time and $\Delta$ is the varying
Roemer-type propagation delay. To first order the observed period is the
intrinsic period stretched by the LOS Doppler factor,

$$
p_\text{obs}(t) \;=\; p_\text{intr}(t)\Bigl(1+\frac{v_\text{los}(t)}{c}\Bigr),
\qquad
p_\text{intr}(t) = p_0 + \dot p\,t .
$$

The code's `observed_period(t, p0, pdot, ω_b, a_psini, A_T, e, ω_peri, m_c, m_p, gr)`
computes exactly this, with $v_\text{los}$ from `los_velocity`. With `gr=True`
(default) `los_velocity` folds in Einstein delay, Shapiro delay, periastron
advance and orbital decay (the Damour–Deruelle terms); with `-no_gr` only the
Keplerian Roemer term is kept.

The **truth velocity** used for the fit is then

$$
v_\text{true}(t) \;=\; c\left(\frac{p_\text{obs}(t)}{p_0}-1\right)
\quad\text{(code: \texttt{true\_velocity})},
$$

i.e. the apparent LOS velocity relative to the nominal period $p_0$. Because
$p_\text{intr}=p_0+\dot p\,t$, the intrinsic spin-down enters as a linear ramp in
velocity — a **pseudo-acceleration** $c\,\dot p/p_0$ — which the cubic fit simply
absorbs into its $a$ coefficient.

For a **circular** orbit ($e=0$, Roemer only, no GR) the delay is a pure sinusoid,

$$
\Delta(t) = x\,\sin(\omega_b t + A_T),
\qquad
\frac{v_\text{los}(t)}{c} = \dot\Delta(t) = x\,\omega_b\cos(\omega_b t+A_T),
$$

where $x=a_p\sin i/c$ is in light-seconds (so $\Delta$ is in seconds). This is the
track the `circular` mode removes exactly (`circular_voc` in `resample.jl`).

### 2.2 Why phase error is measured in metres

Demodulating means remapping the time axis so the pulses become periodic. If the
demodulated series has a residual LOS-velocity error $\delta v(\tau)$, then over a
time $\mathrm{d}\tau$ the pulse-arrival time drifts by
$\mathrm{d}(\Delta t) = \delta v\,\mathrm{d}\tau / c$, and the accumulated **spin
phase** error in cycles is that delay divided by $p_0$:

$$
\Delta\phi \;=\; \frac{1}{c\,p_0}\int \delta v\,\mathrm{d}\tau .
$$

So a **path-length** error of $\Lambda$ metres is $\Lambda/(c\,p_0)$ cycles, and a
phase budget of $\varepsilon$ cycles is a path budget of

$$
\boxed{\;\varepsilon_m = \varepsilon\,c\,p_0\;}
\qquad\text{(code: \texttt{phase\_tol\_m}).}
$$

This is the bridge between the physics (velocities, metres) and the grid budget
(cycles). The code measures the integrated truncation `integrated_trunc` as

$$
\Lambda \;=\; \max_{\tau}\left|\int_{\tau_\text{first}}^{\tau}
\delta v(\tau')\,\mathrm{d}\tau'\right| ,
$$

i.e. the maximum absolute cumulative integral of the residual velocity, starting
at the **first sample** of the span (`np.concatenate(([0.0], cumulative_trapezoid(...)))`).
For the `midpoint` anchor the first sample is at $\tau=-T/2$.

---

## 3. The demodulation operator (`resample.jl`)

A grid point is applied by remapping the time coordinate:

$$
u(\tau) \;=\; \int_0^{\tau}\frac{\mathrm{d}t'}{1+v_\text{inj}(t')/c},
$$

where $v_\text{inj}$ is the **injected** velocity (the sign that, when injected
into the time axis, cancels the observed drift). The resampled series at new
index $u$ takes the old sample at the time $\tau$ solving $u(\tau)=u$. The
resampler is exact in the sense that no interpolation is used: sample $i$ maps to
integer index $\operatorname{round}(u(\tau_i)/dt)$; the map must be monotonic
(checked), gaps scatter to zero, and the de-meaning in `demod_dat.jl` removes the
DC spike that those zero holes would otherwise make.

**Polynomial (ajs) remap** (`_exact_remap_poly`). Expanding
$1/(1+x)=\sum_n(-x)^n$ with $x=v_\text{inj}/c$ and integrating term by term gives

$$
u(\tau) \;=\; \tau - \frac{1}{c}\!\int v_\text{inj}
      + \frac{1}{c^2}\!\int v_\text{inj}^2
      - \frac{1}{c^3}\!\int v_\text{inj}^3 + \cdots
$$

The resampler substitutes $v_\text{inj} = -v$, where
$v(\tau) = v_0 + a\tau + \tfrac{j}{2}\tau^2 + \tfrac{s}{6}\tau^3$ is the cubic with
the **trial coefficients as given** (`_velocity_poly`). The sign flip cancels the
series alternation, so all terms add:

$$
u(\tau) \;=\; \tau + \frac{1}{c}\!\int v
      + \frac{1}{c^2}\!\int v^2
      + \frac{1}{c^3}\!\int v^3 + \cdots
$$

Orders are added until the next order's peak contribution over the span is below
`RESAMPLE_TOL_SAMPLES = 0.1` samples (up to `RESAMPLE_MAX_ORDER = 8`). Because
$|v|/c$ is tiny, convergence is essentially immediate.

**Non-polynomial (circular) remap** (`resample_ts_shift_voc`). A circular-orbit
Doppler is not polynomial in $\tau$, so `_integrate_voc_remap` integrates
$1/(1+\text{voc}(t'))$ numerically by cumulative trapezoid on
$[\min(0,\tau),\max(0,\tau)]$, quadrupling the sample count (from 4000, capped at
2 000 000) until one refinement changes the result by less than 0.1 samples.

**Anchor.** `demod_dat.jl` anchors the polynomial at the **midpoint** of the input
series (`reference_mjd = epoch + 0.5 N dt`), which is why the grid's default
`-anchor midpoint` matches it. For the circular model, `A_T` is the mean anomaly
at the **start** (the `.inf` epoch, matching `pulsegen_gr.py -anchor start`); the
code shifts the midpoint back to the start by `t_offset_s = T/2`
(`0.5*N*dt`). The ajs grid's `midpoint`/`start` choice must match the anchor the
demod uses, or the fitted coefficients refer to the wrong origin. `start` is kept
only for comparison.

---

## 4. Grid sizing, model `ajs`

This is the derivation of the polynomial grid. It proceeds in eight steps.

### 4.1 Step 1 — fit each segment's truth with a cubic

For every scanned orbit $(p_o,\sin i,e,\omega_\text{peri})$ and every segment
start phase $t_0$, the code fits the model

$$
v_\text{model}(\tau) \;=\; v_0 + a\,\tau + \frac{j}{2}\tau^2 + \frac{s}{6}\tau^3
$$

to $v_\text{true}(t)$ over the span $[t_0,\,t_0+T]$, sampled at `nsamp` points
(`fit_velocity_model`). The anchor is

$$
t_\text{anchor} =
\begin{cases}
t_0, & \texttt{anchor}=\texttt{start},\\[2pt]
t_0 + T/2, & \texttt{anchor}=\texttt{midpoint},
\end{cases}
\qquad \tau = t - t_\text{anchor}.
$$

The design matrix has a column of ones ($v_0$, always fitted) followed by the
columns $\tau^{k+1}/k!$ for the masked axes ($k=0,1,2$ for accel, jerk, snap;
`FACTORIALS = [1,2,6]`). Columns are divided by their max abs value for
conditioning, `lstsq` solves, and the coefficients are un-scaled. The fit returns
the max instantaneous residual `trunc` (m/s, used only for reporting) and the
**integrated** truncation

$$
\Lambda \;=\; \texttt{integrated\_trunc}(\text{resid},\tau)
       \;=\; \max_\tau\Bigl|\int_{\tau_\text{first}}^{\tau}\text{resid}\,\mathrm{d}\tau'\Bigr|
       \quad [\text{m}],
$$

the accumulated path-length error of the best cubic against the truth
(`integrated_trunc`). In cycles this is $\Lambda/(c\,p_0)$.

Why is $v_0$ always fitted but never gridded? A constant velocity offset is just a
constant shift of the apparent spin frequency, which the coherent search's period
axis absorbs. Only the $\tau$-dependent part of the mismatch costs phase
coherence. Hence the grid covers only $(a,j,s)$.

### 4.2 Step 2 — feasibility gate

A segment is **feasible** iff its own best-fit cubic already accumulates less than
the budget (`mark_feasible`):

$$
\Lambda < \varepsilon_m \qquad (\texttt{trunc\_phase} < \texttt{phase\_tol\_m}).
$$

No grid density can repair a truncation larger than the budget — the cubic is the
best possible template on that segment — so infeasible segments are simply not
claimed by the grid.

### 4.3 Step 3 — caps

`-max_accel`, `-max_jerk`, `-max_snap` cap the searched coefficients. A feasible
segment is discarded by `apply_caps` if its best-fit $|a|$, $|j|$ or $|s|$ exceeds
its cap. The defaults are derived by `peak_kinematics(p_min_s, e, sin_i, m_c, m_p)`
from the **shortest** orbit in the scan: it samples the true
`los_acceleration`/`los_jerk`/`los_snap` (central differences of `los_velocity`) on
`n_t = 4000` points over one orbital period, maximised over `n_omega = 36` values
of $\omega_\text{peri}$, using the largest scanned $e$ and $\sin i$. This "just
covers the tightest binary requested".

### 4.4 Step 4 — `-drop_pct`

Of the remaining feasible segments, the worst-phase-error `-drop_pct` percent are
discarded (`apply_drop`), pooled over **all** orbits:

* sort the feasible segments by $\Lambda$ ascending (cheapest first),
* keep $n_\text{keep} = \max\!\big(\operatorname{round}(n_\text{feas}(1-\text{drop\_pct}/100)),\,1\big)$,
* the rest stay `feasible` but lose `kept`.

Dropping the worst segments is what lets the grid be coarse: the worst **kept**
segment now sets the residual budget. Dropped phases are simply not claimed (the
plots report the claimed fraction so this cost is explicit).

### 4.5 Step 5 — lever arms

Let the grid point differ from the true best-fit coefficients by
$\delta a,\delta j,\delta s$, each at most half a grid spacing. The extra velocity
is $\delta v = \delta a\,\tau + \tfrac{\delta j}{2}\tau^2 + \tfrac{\delta s}{6}\tau^3$,
and the extra path length accumulated from the span start is

$$
E(\tau) \;=\; \int_{\tau_\text{first}}^{\tau}\delta v\,\mathrm{d}\tau'
\;=\; \delta a\,\frac{\tau^2-\tau_\text{first}^2}{2}
    + \delta j\,\frac{\tau^3-\tau_\text{first}^3}{6}
    + \delta s\,\frac{\tau^4-\tau_\text{first}^4}{24}.
$$

The worst case over the span bounds the path error by

$$
|E| \;\le\; |\delta a|\,L_a + |\delta j|\,L_j + |\delta s|\,L_s,
$$

where the **lever arms** are

$$
L_a = \max_\tau\left|\frac{\tau^2-\tau_\text{first}^2}{2}\right|,\quad
L_j = \max_\tau\left|\frac{\tau^3-\tau_\text{first}^3}{6}\right|,\quad
L_s = \max_\tau\left|\frac{\tau^4-\tau_\text{first}^4}{24}\right|.
$$

The starting point $\tau_\text{first}$ is the first sample of the span. Evaluated
for each anchor:

**`start`** anchor, $\tau_\text{first}=0$, $\tau\in[0,T]$:

$$
L_a = \max_{0\le\tau\le T}\frac{\tau^2}{2} = \frac{T^2}{2},\qquad
L_j = \frac{T^3}{6},\qquad
L_s = \frac{T^4}{24}.
$$

**`midpoint`** anchor, $\tau_\text{first}=-T/2$, $\tau\in[-T/2,T/2]$:

* Accel: $\dfrac{\tau^2-T^2/4}{2}$ is $0$ at the ends and $-T^2/8$ at the
  centre, so $L_a = T^2/8$.
* Jerk: $\dfrac{\tau^3+T^3/8}{6}$ ranges from $0$ to $+T^3/24$, so
  $L_j = T^3/24$.
* Snap: $\dfrac{\tau^4-T^4/16}{24}$ is $0$ at the ends and $-T^4/384$ at the
  centre, so $L_s = T^4/384$.

These are exactly the entries of `PHASE_LEVERS`:

$$
\texttt{start}:[T^2/2,\;T^3/6,\;T^4/24],\qquad
\texttt{midpoint}:[T^2/8,\;T^3/24,\;T^4/384].
$$

A unit coefficient offset on an axis therefore costs at most $L_x$ metres of path
error, i.e. $L_x/(c\,p_0)$ cycles.

### 4.6 Step 6 — splitting the leftover budget

The worst kept segment already spends $\Lambda_\text{max}$ metres of the budget.
What remains for grid mismatch is

$$
B_m = \varepsilon_m - \Lambda_\text{max}
\qquad(\texttt{budget\_m}),
$$

which is positive because every kept segment passed the feasibility gate. The
grid is chosen by `allocate_spacings(B_m, [L_a,L_j,L_s], ranges, mask)`:

1. Start with `active = mask` (axes present in the model).
2. **Saturated axes (fixed cost).** An axis that is in the mask but already
   deactivated ("**saturated**", below) is covered by a **single** trial, placed at
   the range midpoint (`grids_from_spec`). Its worst-case offset is half the range,
   so it costs a *fixed* amount of path error
   $$
   F = \sum_{x\ \text{saturated}} \frac{\text{hi}_x-\text{lo}_x}{2}\,L_x .
   $$
   Axes that are off (not in `mask`) cost nothing.
3. If $B_m - F \le 0$ the budget cannot even cover the fixed costs → return
   infeasible (`None`).
4. **Equal split.** The remaining budget is split *equally* over the
   $n_\text{act}$ still-active axes:
   $$
   \epsilon_x \;=\; \frac{B_m - F}{n_\text{act}} \quad\text{(same for every active axis).}
   $$
   Each active axis may then be off by at most half a spacing, and the resulting
   path error is at most $\epsilon_x$:
   $$
   \frac{D_x}{2}\,L_x = \epsilon_x
   \quad\Longrightarrow\quad
   \boxed{\,D_x = \frac{2\,\epsilon_x}{L_x}\,}.
   $$
   This is the line `spac[x] = 2.0*eps/levers[x]`.
5. With these spacings, add up the per-axis contributions: the active axes
   contribute at most $\sum_\text{active}\epsilon_x = B_m-F$, and the saturated
   axes contribute at most $F$, giving a total grid-mismatch path error of at most
   $B_m$. Together with the truncation $\Lambda_\text{max}$ this is at most
   $\varepsilon_m$ — the chosen budget is never exceeded. (The sum of per-axis
   worst cases is an upper bound, so the true worst case is usually smaller; Step 8
   verifies it.)

**Saturation.** If an axis's whole range fits inside a single cell,
$\text{hi}_x-\text{lo}_x \le D_x = 2\epsilon_x/L_x$, then giving it more than one
trial is pointless; the axis is **saturated**. Its spacing is set to the full
range, it is moved out of `active` (so it costs exactly $F$ above and gets a single
midpoint trial), and the loop re-runs — recomputing $\epsilon_x$ over the remaining
active axes. Because each pass can only saturate more axes, at most 3 passes are
needed; the loop runs up to 4 and returns `None` only if it somehow cannot
converge (or the budget is exhausted).

### 4.7 Step 7 — ranges, trial counts, concrete grid

The grid must cover every kept segment's best-fit coefficients, so the ranges are
the min/max over the kept segments (`coeff_ranges`):

$$
R_x = [\text{lo}_x,\text{hi}_x] = \bigl[\min_\text{kept} x,\ \max_\text{kept} x\bigr].
$$

An active axis then gets

$$
n_x = \Bigl\lceil \frac{\text{hi}_x-\text{lo}_x}{D_x}\Bigr\rceil + 1
$$

trials at $\text{lo}_x + D_x\cdot\{0,1,\dots,n_x-1\}$ (`grids_from_spec`). The `+1`
means the last point reaches at least $\text{hi}_x$ (it may overshoot by up to one
spacing), so **every** value in the range is within $D_x/2$ of a grid point — the
half-spacing assumption used in Step 5. A saturated or off axis gets exactly one
trial (the range midpoint, or 0). The total template count is
$\prod_x n_x$ (`n_templates`).

### 4.8 Step 8 — validation

The analytical bound is conservative, so the code verifies it at the **worst kept
segment** (`worst = max kept trunc_phase`). It refits that segment to get its true
residual `resid_v` and axis `tau`, then adds the grid-quantisation velocity

$$
q(\tau) = \delta a\,\tau + \frac{\delta j}{2}\tau^2 + \frac{\delta s}{6}\tau^3,
\qquad \delta x \in \pm D_x/2,
$$

and computes the exact accumulated phase
$\texttt{integrated\_trunc}(\text{resid}_v+q,\tau)/(c\,p_0)$. It checks all **8
corners** of the cell $\delta x=\pm D_x/2$ (worst-case signs), plus `n_mc` random
draws uniform in $[-D_x/2,D_x/2]^3$. The guarantee holds iff

$$
\max_\text{corners}\bigl(\text{integrated cycles}\bigr) \le \varepsilon\,(1+10^{-9}).
$$

The printed worst corner is the number to trust; the Monte-Carlo max/median show
the typical offset.

### 4.9 Worked example (ajs)

A reduced scan on the bundled `inj.inf` (`T=1509.95 s`, `p0=1 s`,
`-n_phase 24 -nsamp 1000`, defaults otherwise) gives, after the gate, caps and
drop:

```
Phase budget: 0.0336 cycles (1.006e+07 m), split equally over 3 active axes
  axis      spacing    range_lo     range_hi  n_trials  eps [cyc]
 accel   2.3539e+01  -2.0131e+03   2.0933e+03     176     0.0112
  jerk   4.6768e-02  -3.4502e+00   4.8213e+00     178     0.0112
  snap   4.9557e-04  -2.8294e-02   2.7268e-02     114     0.0112
Total templates: 3.571e+06
Validation ... worst grid corner : 0.0855 cycles ... guarantee holds: True
```

Check the arithmetic against the formulas (midpoint anchor, $T=1509.9494$ s,
$c\,p_0=2.99792458\times10^8$ m):

* $L_a=T^2/8=2.84993\times10^5$, $L_j=T^3/24=1.43442\times10^8$,
  $L_s=T^4/384=1.35369\times10^{10}$.
* $B_m=1.006\times10^7$ m over 3 axes ⟹ $\epsilon_x=3.353\times10^6$ m
  $=0.01118$ cycles.
* $D_a=2\epsilon/L_a=23.539$, $D_j=2\epsilon/L_j=0.046768$,
  $D_s=2\epsilon/L_s=4.9557\times10^{-4}$ — match the output.
* $n_a=\lceil(2093.3+2013.1)/23.539\rceil+1=176$, etc.

(The numbers move with `-n_phase`/`-nsamp` because the set of kept segments
changes; the formulas do not.)

---

## 5. Grid sizing, model `circular`

### 5.1 Why there is no truncation gate

The circular demod removes the exact Roemer track
$\Delta=x\sin(\omega_b t + A_T)$, which is not a polynomial. At the true
$(x,\omega_b,A_T)$ the residual is zero, so there is no residual "truncation" to
gate on. Every scanned orbit is feasible, and `-drop_pct` is the only cost
control. The only error at a grid point is the **parameter mismatch** between
truth and grid.

### 5.2 The residual spin phase

The orbit-induced residual spin phase (cycles) — the quantity the budget bounds —
is the difference of the delays over $p_0$:

$$
\Phi(t) \;=\; \frac{x}{p_0}\sin(\omega_b t + A_T),
\qquad t\in[0,T],
$$

with $t$ the **absolute** time since the observation start (because $A_T$ is
anchored at the start; `demod_dat.jl` maps the midpoint anchor back to the start).
Equivalently $\Phi = (1/p_0)\int (v/c)\,\mathrm{d}t$.

### 5.3 First-order perturbation: the three amplitudes

A grid point differs from the truth by $(\delta x,\delta A_T,\delta\omega_b)$. To
first order,

$$
\delta\Phi(t) \;=\;
\underbrace{\frac{\delta x}{p_0}}_{\textstyle \alpha}\sin\varphi
\;+\;\frac{x}{p_0}\bigl(\delta A_T + t\,\delta\omega_b\bigr)\cos\varphi,
\qquad \varphi=\omega_b t + A_T .
$$

The worst-case offsets are half a spacing, so the code defines the three
amplitudes

$$
\alpha \;=\; \frac{D_x/2}{p_0},\qquad
\beta \;=\; \frac{x_\text{lever}\,(D_{A_T}/2)}{p_0},\qquad
\gamma \;=\; \frac{x_\text{lever}\,(D_{\omega}/2)\,T}{p_0},
$$

and writes

$$
\delta\Phi(t) \;=\; \alpha\sin\varphi + \Bigl(\beta+\gamma\,\frac{t}{T}\Bigr)\cos\varphi .
$$

Two points on the definitions:

* The $A_T$ and $\omega_b$ terms carry the **grid point's** $x$, so they are
  evaluated at $x_\text{lever}=x[-1]$, the top of the (padded) $x$ grid — a
  covered orbit near the top can snap up to that padded point (Section 5.5).
* The $\omega_b$ lever is the **full** $T$, not $T/2$, because $t$ runs over
  $[0,T]$ from the start.

### 5.4 The joint bound and the volume-optimal split

At a fixed $t$ we must bound $|\alpha\sin\varphi + B\cos\varphi|$ over the unknown
orbital phase $\varphi$, where $B=\beta+\gamma\,t/T$. The identity

$$
\max_\varphi\bigl|\alpha\sin\varphi + B\cos\varphi\bigr| = \sqrt{\alpha^2+B^2}
$$

(for any $\varphi$) and $B\le\beta+\gamma$ (both non-negative, worst at $t=T$) give
the exact worst case

$$
\max_{t,\varphi}|\delta\Phi| = \sqrt{\alpha^2+(\beta+\gamma)^2}.
$$

Requiring this to be within the budget gives the **joint constraint**

$$
\boxed{\;\alpha^2 + (\beta+\gamma)^2 \;\le\; \varepsilon^2\;}.
$$

We now choose $\alpha,\beta,\gamma$ to maximise the cell **volume**
$D_x\,D_{A_T}\,D_\omega$. Since each spacing is proportional to its amplitude
($D_x\propto\alpha$, $D_{A_T}\propto\beta$, $D_\omega\propto\gamma$), this is
maximising $\alpha\beta\gamma$ subject to $\alpha^2+(\beta+\gamma)^2=\varepsilon^2$.

1. For fixed $\alpha$ and fixed $s=\beta+\gamma$, the product
   $\beta\gamma$ is maximised when $\beta=\gamma=s/2$ (AM–GM).
2. So maximise $\alpha\,(s/2)^2$ subject to $\alpha^2+s^2=\varepsilon^2$. Set
   $f=\alpha s^2$ and use a Lagrange multiplier:
   $$
   \frac{\partial f}{\partial\alpha}=s^2=2\lambda\alpha,\qquad
   \frac{\partial f}{\partial s}=2\alpha s=2\lambda s
   \;\Longrightarrow\; \alpha=\lambda,\quad s=\alpha\sqrt2 .
   $$
   Then $\alpha^2+s^2 = 3\alpha^2=\varepsilon^2$.

Therefore

$$
\alpha_\text{bud} = \frac{\varepsilon}{\sqrt3},\qquad
\beta_\text{bud}=\gamma_\text{bud}=\frac{\varepsilon}{\sqrt6}
\quad(\text{code: \texttt{a\_bud}, \texttt{bc\_bud}}).
$$

(The old per-axis split $\alpha=\beta=\gamma=\varepsilon/3$, which only satisfies
the triangle inequality, gives a cell volume smaller by a factor
$\frac{27}{6\sqrt3}\approx 2.6$.)

### 5.5 Spacings, ranges and the concrete grid

Inverting the amplitude definitions ($\delta = D/2$ at the worst corner) gives the
spacings used in the code:

$$
D_x = 2\,\alpha_\text{bud}\,p_0,\qquad
D_{A_T} = \frac{2\,\beta_\text{bud}\,p_0}{x_\text{lever}},\qquad
D_\omega = \frac{2\,\gamma_\text{bud}\,p_0}{x_\text{lever}\,T}.
$$

The axes (`derive_circular_grid`):

* **x axis.** Over the kept orbits, $x\in[x_\text{min},x_\text{max}]$. The grid is
  $\texttt{x} = x_\text{min} + D_x\cdot\{0,\dots,\lceil(x_\text{max}-x_\text{min})/D_x\rceil\}$,
  padded to the next step above $x_\text{max}$ so every covered orbit is within
  $D_x/2$ of a grid point. The lever for the other axes is then
  $x_\text{lever}=x[-1]$ — the **top of the padded grid**, not $x_\text{max}$ —
  because a covered orbit at $x_\text{max}$ can snap to that padded top point.
* **$\omega_b$ axis.** Uniform in $\omega_b=2\pi/p_{o,s}$ (not in $p_o$: the phase
  error is linear in $\delta\omega_b$, and it is $\omega_b$ that keeps the $\cos$
  lever constant), over
  $[\omega_\text{lo},\omega_\text{hi}]=[2\pi/p_{o,\max},\,2\pi/p_{o,\min}]$ on the
  kept orbits, same padding.
* **$A_T$ axis.** Uniform over $[0,2\pi)$ with
  $\lceil 2\pi/D_{A_T}\rceil$ points (no `+1`; the axis is periodic and wraps).
* Counts $n_{\omega_b},n_x,n_{A_T}$ and total $n_\omega n_x n_{A_T}$.

`write_circular_outputs` writes the CSV in the demod's own coordinates: `pb` in
**days** ($2\pi/\omega_b/86400$), `x` in light-seconds, `at` $=A_T$ in radians at
the `.inf` epoch.

### 5.6 `-drop_pct` in circular mode

There is no truncation gate, so `apply_drop_circular` ranks orbits by
$x=a_p\sin i/c$ and drops the largest-$x$ `drop_pct` percent, keeping at least
one. Every grid lever is monotonic in $x$ (the $A_T$ and $\omega_b$ spacings are
proportional to $1/x_\text{lever}$, hence the trial counts grow with $x$), so the
highest-$x$ orbits are the most expensive to cover; dropping them first buys the
largest template reduction. The dropped orbits are not claimed, and the coverage
plot reports exactly which survive.

### 5.7 Validation

`test_circular_grid.py` Monte-Carlos the guarantee directly, without
linearisation. It imports the grid code (so it cannot drift), samples random
circular orbits uniformly in the covered ranges (log-uniform $p_o$, uniform
$\sin i$, uniform $A_T$), snaps each coordinate to the nearest grid value, and
computes the **exact** residual spin phase
$R(t)=\frac{x_t}{p_0}\sin(\omega_t t+A_t)-\frac{x_g}{p_0}\sin(\omega_g t+A_g)$ on a
fine time grid. It asserts

* the worst $\max_t|R|$ over all Monte-Carlo trials is within the budget,
* the deterministic worst cell corner is within budget on both the linear bound
  $\sqrt{\alpha^2+(\beta+\gamma)^2}$ **and** the exact nonlinear $R$,
* a random cell-corner scan (cell midpoints, which uniform sampling almost never
  hits) is within budget.

Run it with `-drop_pct 0` to verify the fundamental spacing over the full range,
or with the desired `-drop_pct` to check the kept box (it reports how many samples
fell outside).

### 5.8 Worked example (circular)

On the bundled `inj.inf` with `-drop_pct 50` (fully default otherwise), the
derivation above gives exactly:

```
p_o = 0.001388-0.1662 d (342 periods, uniform in omega_b)
x   = 0.008568-0.2956 lt-s (4)
A_T = 0-2pi (28)
joint phase budget 0.1 cyc (a^2+(b+c)^2 <= tol^2; a=0.05774, b=c=0.04082)
d(omega_b)=1.5233e-04 rad/s   d(x)=1.1547e-01 lt-s   d(A_T)=2.3001e-01 rad
Total templates: 3.830e+04
```

Check: $\alpha_\text{bud}=0.1/\sqrt3=0.057735$,
$\beta_\text{bud}=0.1/\sqrt6=0.040825$;
$D_x=2(0.057735)=0.11547$ lt-s. The $x$ grid top is
$x[-1]=0.008568+3(0.11547)=0.35498$, so
$D_{A_T}=2(0.040825)/0.35501=0.23001$ rad and
$D_\omega=0.23001/1509.95=1.5233\times10^{-4}$ rad/s. Counts:
$\lceil 2\pi/0.23001\rceil=28$, and
$n_\omega=\lceil(0.052360-0.00043746)/1.5233\times10^{-4}\rceil+1=342$.

---

## 6. Grid sizing, model `hybrid`

Hybrid runs **one** full-ladder ajs scan (the machinery of §4), uses it to place
`p_break`, then builds *two* sub-grids: a circular grid (§5) for the periods
below `p_break`, and an ajs grid (§4) for the scanned records at/above it.

### 6.1 Placing `p_break`

For each scanned `p_o`, `ajs_coverage_by_p_o` computes the **mean feasibility**
over the `e = 0` records, pooled across the scanned `sin_i × omega_peri` combos:

$$
\text{cov}(p_o) \;=\;
\frac{\#\{\text{e=0 segments at } p_o \text{ that pass the phase gate and the caps}\}}
     {\#\{\text{e=0 segments at } p_o\}} .
$$

This is measured **before** `-drop_pct`: the drop is a cost knob, and the break
should reflect physics, not the cost budget. Feasibility here is the same
`trunc_phase < phase_tol_m` gate plus the caps (§4.2–4.3), since a segment that
fails either cannot be claimed.

Because coverage dips near `p_o ~ T_obs` and recovers for longer orbits, the
switch must clear the entire inadequate band. With the `-p_o` ladder sorted
ascending, let `bad` be the ladder points with `cov < -break_coverage/100`. Then

$$
p_\text{break} =
\begin{cases}
p_{\text{ladder}[0]}, & \text{no } p_o \text{ is bad (all ajs)}, \\[3pt]
p_{\text{ladder}[i_\text{bad}+1]}, & i_\text{bad} = \text{index of the last bad } p_o, \\[3pt]
p_{\max}\,(1+10^{-9}), & \text{the longest } p_o \text{ is still bad (all circular).}
\end{cases}
$$

Orbits are then split by `p_o < p_break` (circular) and `p_o ≥ p_break` (ajs).
A manual `-p_break` (in yr) overrides the search entirely.

### 6.2 The two sub-grids

* **Circular side** (`derive_circular_grid` with `p_o` filtered to `< p_break`):
  unchanged from §5, except the period ladder is restricted. `-drop_pct` still
  drops the highest-`x` orbits of this side.
* **Ajs side** (`ajs_grid_from_scan` on the records with `p_o ≥ p_break`): the
  gate, `-drop_pct`, ranges and allocation of §4 are applied to that subset only.
  The default caps are re-derived from the **shortest ajs-side orbit** (near
  `p_break`) rather than the global shortest, because the tighter binaries are
  now the circular side's job; this shrinks the `(a,j,s)` ranges and the template
  count. An explicit `-max_accel/...` still wins.

### 6.3 Outputs

Hybrid writes `<outstem>_circular.csv` (`pb,x,at`) and `<outstem>_ajs.csv`
(`accel,jerk,snap`), plus one combined `<outstem>.yaml` with `mode: hybrid`,
`p_break`, the per-`p_o` coverage, and both sub-specs nested under `circular:`
and `ajs:`. `demod_grid.jl` accepts both CSVs in one invocation (it reads each
file's own header), and `run_nsns_sweep.sh MODE=hybrid` reads each point's model
from its CSV header so both filename patterns coexist.

### 6.4 Worked example (hybrid)

On the bundled `inj.inf` with `-n_phase 24 -nsamp 1000` (T = 1509.95 s), the ajs
coverage is:

```
      p_o [yr]    p_o [d]   ajs coverage
   3.80257e-06  0.0013889          83.3%
   6.91568e-06   0.002526          60.0%
   1.25774e-05  0.0045939          40.0%   <- dip
   2.28744e-05  0.0083549          28.3%   <- worst (~p_o ~ T_obs)
   4.16013e-05   0.015195          93.3%   <- recovers
   7.56597e-05   0.027635         100.0%
   ... (all longer periods 100%)
```

The last `p_o` below 60% is 0.008355 d, so `p_break = 0.015195 d`: circular
covers 2 min–12 min (4 ladder points, 11115 templates), ajs covers 22 min–1 d
(28800 scanned segments, caps from the `p_break` orbit). The ajs side's
`trunc_max` is 0.0272 cycles (vs 0.0664 for the all-ajs grid), because the hard
short orbits are gone.

---

## 7. Outputs

Every mode writes machine-readable files for the sweep:

* **`<outstem>.yaml`** — the full spec: ranges, spacings, active/caps, tolerances,
  counts. Circular adds `mode`, `n_omega_b/n_x/n_A_T/n_trials`; ajs adds `terms`,
  `anchor`, `trunc_max`, `budget_m`, `eps`, `active`, `guarantee_ok`; hybrid adds
  `p_break`, `break_coverage`, `coverage_by_p_o` and nests both sub-specs under
  `circular:` and `ajs:`.
* **`<outstem>.csv`** — every concrete trial, `%.10g`, one per line, **LF** line
  endings (bash builds output filenames from these literal strings):
  * ajs header `accel,jerk,snap`, one row per $(a,j,s)$;
  * circular header `pb,x,at`, one row per $(p_b,x,A_T)$.
  * Hybrid writes **two** files, `<outstem>_circular.csv` and `<outstem>_ajs.csv`.
* Plots under `-plot_subdir` (default `grid_size_plots/`):
  `corner_feasible_fraction.png`, `claimed_fraction.png` (ajs), and
  `circular_coverage.png` (circular; hybrid writes `hybrid_circular_coverage.png`).
  With `-plot`, ajs also writes one phase / best-fit figure per orbit.

`grids_from_spec` and `write_grid_outputs` (ajs) build the Cartesian product and
write it; `write_circular_outputs` does the same for circular, converting
$\omega_b\to p_b$ (days) for the CSV; hybrid calls the ajs and circular writers
with explicit per-model CSV paths.

---

## 8. Directory reference

| file | what it does |
|---|---|
| `nsns_grid.py` | **Grid derivation** (both modes), YAML/CSV/plots. |
| `resample.jl` | The exact time-remap resamplers (polynomial and non-polynomial). |
| `demod_dat.jl` | Apply the remap to one `.dat`/`.inf`; CLI + library `demod_file`. |
| `demod_grid.jl` | Batch: demodulate one observation at every row of a grid CSV, in one Julia process. |
| `pulsegen_gr.py` | Inject a full Damour–Deruelle binary pulsar into a `.dat`/`.inf` (+ `_truth.yaml`). |
| `run_demod_search.sh` | End-to-end: inject → derive grid → nearest grid point → demod → search. |
| `test_circular_demod.sh` | Same, for circular mode, with assertions on recovery. |
| `run_nsns_sweep.sh` | Full sweep: derive grid → demodulate **every** point → FFT → search all; CPU-parallel or one GPU invocation. |
| `combine_cohout.py` | Merge all per-point `.cohout` into one candidate list (optional plot + truth crosshair). |
| `test_circular_grid.py` | MCMC / adversarial validation of the circular grid's phase guarantee. |
| `nsns_grid.yaml`, `nsns_grid.csv` | Example derived grid. |
| `inj.dat`, `inj.inf`, `demod_out/`, `grid_size_plots/`, `plots/`, `nsns_sweep*/` | Test observation and outputs. |

The orbit and DD-model helpers (`los_velocity`, `los_acceleration`,
`los_jerk`, `los_snap`, `observed_period`, `semi_major_axis_sini`, `pk_coeffs`)
live in the sibling `FFA_stacking/orbit_simulation/simulate_orbit_accel_jerk.py`,
located via `$FFA_REPO` (default `/home/fadong/Documents/FFA_stacking`).

---

## 9. Running it

```sh
# One-shot: inject a binary, derive the grid, demodulate at the nearest point,
# and search.  All knobs are environment variables (see the script header).
bash demod/run_demod_search.sh
SNR=50 MODE=circular PB_DAYS=0.041666666666666664 bash demod/run_demod_search.sh

# Full sweep over every grid point (grid -> demod all -> realfft/rednoise/search).
bash demod/run_nsns_sweep.sh
MODE=circular DROPPCT=50 bash demod/run_nsns_sweep.sh
MODE=hybrid bash demod/run_nsns_sweep.sh                    # circular + ajs
MODE=hybrid P_BREAK=0.02 bash demod/run_nsns_sweep.sh       # manual p_break [yr]
MODE=hybrid PO="0.0000038 0.00001 0.1 100" bash demod/run_nsns_sweep.sh
GRID_CSV=/path/to/grid.csv bash demod/run_nsns_sweep.sh   # skip derivation

# Grid only:
python demod/nsns_grid.py -inf OBS.inf -p0 1.0 -phase_tol_cycles 0.1
python demod/nsns_grid.py -inf OBS.inf -mode circular -drop_pct 50
python demod/nsns_grid.py -inf OBS.inf -mode hybrid          # auto p_break
python demod/nsns_grid.py -inf OBS.inf -mode hybrid -p_break 0.02 -break_coverage 60

# Validate the circular guarantee:
python demod/test_circular_grid.py -inf OBS.inf -drop_pct 0 -n_mc 2000
```

Requires Julia with the `CoherentSearch.jl` project, a Python with
numpy/scipy/matplotlib/yaml, PRESTO's `realfft`/`rednoise` on `PATH`, and the
sibling `FFA_stacking` checkout.

---

## 10. Tests

* `test_circular_grid.py` — verifies the circular phase-tolerance guarantee
  (Section 5.7). Exits non-zero on failure so it can gate a run.
* `test_circular_demod.sh` — end-to-end circular inject → grid → demod → search,
  asserting the pulsar is recovered near $1/p_0$ above a threshold S/N.
* `run_demod_search.sh` — end-to-end ajs (and circular) smoke test.
