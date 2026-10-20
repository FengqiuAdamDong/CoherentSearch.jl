<!--
Math-notation note: GitHub runs its Markdown pass BEFORE MathJax, and Markdown
treats a backslash before ASCII punctuation (\, \; \! \_ \{ \} \#) as an escape,
stripping one backslash. So every such TeX macro here is written with a DOUBLE
backslash (\\, \\; \\_ ...) so that exactly one survives into MathJax. Row
breaks in cases/aligned are written \cr, not \\[Npt] (which the same pass
mangles). Do not "tidy" these backslashes away, and re-check with the GitHub
Markdown API if you add new math.

GitHub's math renderer also BLOCKS certain TeX macros outright ("The following
macros are not allowed"). The operator-name macro is one: use \mathrm (or
\text) instead. Plain/AMS macros (\frac, \dfrac, \tfrac, \boxed, \underbrace,
\textstyle, \lbrace, \rbrace, spacing macros) render fine.
-->

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
2. **Demodulate, FFT, deredden and search** each grid point, either through the
   file-based shell pipeline (`demod_grid.jl` → `demod_dat.jl` → PRESTO's
   `realfft`/`rednoise` → `coherent_search.jl`) or entirely in memory with
   `bin/coherent_search.jl --demod-grid` (Section 11), which writes only the
   `.cohout` files.
3. **Combine** the per-point candidate lists (`combine_cohout.py`).

`nsns_grid.py` has three grid models, selected with `-mode`:

* **`-mode ajs`** (default): an accel/jerk/snap **polynomial** grid. The orbit's
  LOS velocity is approximated by a cubic over the observation; the grid covers
  the cubic coefficients. This is the general case — it works for eccentric and
  relativistic orbits, but the cubic cannot track the true orbit exactly, so
  there is a residual ("truncation") phase error that must fit inside the budget.
* **`-mode circular`**: a pure-Keplerian **circular-orbit** grid in
  `(ω_b, x, A_T)`. The circular demod removes the exact non-polynomial Roemer
  track, so at the true parameters the residual is zero and there is no
  truncation gate; the only error is grid-point mismatch. The budget bounds
  the mismatch phase the spin-frequency search **cannot** absorb (§5.6), so
  orbits not much shorter than the observation get much coarser steps.
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
| $c$ | `C` | speed of light, $299792458\ \mathrm{m\\,s^{-1}}$ |
| $G$ | `G` | gravitational constant, $6.67430\times10^{-11}$ |
| $M_\odot$ | `SOLAR_MASS` | solar mass, $1.989\times10^{30}\ \mathrm{kg}$ |
| $T$ | `t_obs` | observation length $= N\\,dt$ [s] |
| $N$ | `N` | number of time-series bins (from `.inf`) |
| $dt$ | `dt` | time-series bin width [s] (from `.inf`) |
| $p_0$ | `p0` | pulsar spin period [s] (use the **minimum** of the search range) |
| $\dot p$ | `pdot` | intrinsic period derivative [s/s] (in the truth) |
| $\tau$ | `tau` | time measured from the anchor: $\tau = t-t_\text{anchor}$ |
| $p_o$ | `p_o` | orbital period [yr] (scanned; internally $p_{o,s}=p_o\\,Y\\!E\\!A\\!R_S$ [s]) |
| $e$ | `e` | orbital eccentricity (scanned) |
| $\sin i$ | `sin_i` | sine of orbital inclination (scanned) |
| $\omega_\text{peri}$ | `omega_peri` | argument of periastron [rad] (scanned) |
| $A_T$ | `A_T` | mean anomaly [rad] at the observation **midpoint** (circular model); in the ajs scan, `-A_T` (default 0) at the scan's t = 0, immaterial there since t0 sweeps a full orbit |
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
| $\varepsilon_m$ | `phase_tol_m` | the same budget as a path length [m], $\varepsilon_m=\varepsilon\\,c\\,p_0$ |
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
-p_o             12 log-spaced points from 4 h to 1 day [yr]
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
-circ_strict     off     (circular/hybrid: bound the raw max|R|, §5.6)
-no_write        off     (derive + report only; write no CSV/YAML/plots)
```

> The actual default ladder is 4 hours to 1 day, as set by `_NSNS_PO_MIN_D`,
> `_NSNS_PO_MAX_D` and matching the `-p_o` help text.

---

## 2. The physical model

### 2.1 Observed period and LOS velocity

An orbiting pulsar's pulses arrive at

$$
t_\text{arr}(t) \\;=\\; t + \Delta(t),
$$

where $t$ is the pulse's proper (emission) time and $\Delta$ is the varying
Roemer-type propagation delay. To first order the observed period is the
intrinsic period stretched by the LOS Doppler factor,

$$
p_\text{obs}(t) \\;=\\; p_\text{intr}(t)\Bigl(1+\frac{v_\text{los}(t)}{c}\Bigr),
\qquad
p_\text{intr}(t) = p_0 + \dot p\\,t .
$$

The code's `observed_period(t, p0, pdot, ω_b, a_psini, A_T, e, ω_peri, m_c, m_p, gr)`
computes exactly this, with $v_\text{los}$ from `los_velocity`. With `gr=True`
(default) `los_velocity` folds in Einstein delay, Shapiro delay, periastron
advance and orbital decay (the Damour–Deruelle terms); with `-no_gr` only the
Keplerian Roemer term is kept.

The **truth velocity** used for the fit is then

$$
v_\text{true}(t) \\;=\\; c\left(\frac{p_\text{obs}(t)}{p_0}-1\right)
\quad\text{(code: \texttt{true\\_velocity})},
$$

i.e. the apparent LOS velocity relative to the nominal period $p_0$. Because
$p_\text{intr}=p_0+\dot p\\,t$, the intrinsic spin-down enters as a linear ramp in
velocity — a **pseudo-acceleration** $c\\,\dot p/p_0$ — which the cubic fit simply
absorbs into its $a$ coefficient.

For a **circular** orbit ($e=0$, Roemer only, no GR) the delay is a pure sinusoid,

$$
\Delta(t) = x\\,\sin\!\bigl(\omega_b (t-T/2) + A_T\bigr),
\qquad
\frac{v_\text{los}(t)}{c} = \dot\Delta(t) = x\\,\omega_b\cos\!\bigl(\omega_b (t-T/2)+A_T\bigr),
$$

with $t$ measured from the observation start and $A_T$ the mean anomaly at the
**midpoint** (Section 3); $x=a_p\sin i/c$ is in light-seconds (so $\Delta$ is in
seconds). This is the track the `circular` mode removes exactly (`circular_voc` in
`src/demod.jl`).

### 2.2 Why phase error is measured in metres

Demodulating means remapping the time axis so the pulses become periodic. If the
demodulated series has a residual LOS-velocity error $\delta v(\tau)$, then over a
time $\mathrm{d}\tau$ the pulse-arrival time drifts by
$\mathrm{d}(\Delta t) = \delta v\\,\mathrm{d}\tau / c$, and the accumulated **spin
phase** error in cycles is that delay divided by $p_0$:

$$
\Delta\phi \\;=\\; \frac{1}{c\\,p_0}\int \delta v\\,\mathrm{d}\tau .
$$

So a **path-length** error of $\Lambda$ metres is $\Lambda/(c\\,p_0)$ cycles, and a
phase budget of $\varepsilon$ cycles is a path budget of

$$
\boxed{\\;\varepsilon_m = \varepsilon\\,c\\,p_0\\;}
\qquad\text{(code: \texttt{phase\\_tol\\_m}).}
$$

This is the bridge between the physics (velocities, metres) and the grid budget
(cycles). The code measures the integrated truncation `integrated_trunc` as

$$
\Lambda \\;=\\; \max_{\tau}\left|\int_{\tau_\text{first}}^{\tau}
\delta v(\tau')\\,\mathrm{d}\tau'\right| ,
$$

i.e. the maximum absolute cumulative integral of the residual velocity, starting
at the **first sample** of the span (`np.concatenate(([0.0], cumulative_trapezoid(...)))`).
For the `midpoint` anchor the first sample is at $\tau=-T/2$.

---

## 3. The demodulation operator (`src/demod.jl`)

A grid point is applied by remapping the time coordinate:

$$
u(\tau) \\;=\\; \int_0^{\tau}\frac{\mathrm{d}t'}{1+v_\text{inj}(t')/c},
$$

where $v_\text{inj}$ is the **injected** velocity (the sign that, when injected
into the time axis, cancels the observed drift). The resampled series at new
index $u$ takes the old sample at the time $\tau$ solving $u(\tau)=u$. The
resampler is exact in the sense that no interpolation is used: sample $i$ maps to
integer index $\mathrm{round}(u(\tau_i)/dt)$; the map must be monotonic
(checked), gaps scatter to zero, and the de-meaning in `demodulate_series` removes the
DC spike that those zero holes would otherwise make.

**Polynomial (ajs) remap** (`_exact_remap_poly`). Expanding
$1/(1+x)=\sum_n(-x)^n$ with $x=v_\text{inj}/c$ and integrating term by term gives

$$
u(\tau) \\;=\\; \tau - \frac{1}{c}\\!\int v_\text{inj}
      + \frac{1}{c^2}\\!\int v_\text{inj}^2
      - \frac{1}{c^3}\\!\int v_\text{inj}^3 + \cdots
$$

with $v_\text{inj} = v$, where
$v(\tau) = v_0 + a\tau + \tfrac{j}{2}\tau^2 + \tfrac{s}{6}\tau^3$ is the cubic with
the **trial coefficients as given** (`_velocity_poly`), positive receding (the
grid fits $v = c\,(p_\text{obs}/p_0-1)$). The code builds $w=-v$ and sums
$\tau+\sum_n\int w^n/c^n$, which is the same alternating series:

$$
u(\tau) \\;=\\; \int_0^\tau\frac{d\tau'}{1+v/c}
      \\;=\\; \tau - \frac{1}{c}\\!\int v
      + \frac{1}{c^2}\\!\int v^2
      - \frac{1}{c^3}\\!\int v^3 + \cdots
$$

A receding pulsar's clock runs slow by $(1+v/c)$; dividing it out cancels the
drift. The circular remap (`circular_voc`) uses the same sign.

Orders are added until the next order's peak contribution over the span is below
`RESAMPLE_TOL_SAMPLES = 0.1` samples (up to `RESAMPLE_MAX_ORDER = 8`). Because
$|v|/c$ is tiny, convergence is essentially immediate.

**Non-polynomial (circular) remap** (`resample_ts_shift_voc`). A circular-orbit
Doppler is not polynomial in $\tau$, so `_integrate_voc_remap` integrates
$1/(1+\text{voc}(t'))$ numerically by cumulative trapezoid on
$[\min(0,\tau),\max(0,\tau)]$, quadrupling the sample count (from 4000, capped at
2 000 000) until one refinement changes the result by less than 0.1 samples.

**Anchor.** `demodulate_series` anchors the polynomial at the **midpoint** of the input
series (`reference_mjd = epoch + 0.5 N dt`), which is why the grid's default
`-anchor midpoint` matches it. For the circular model, `A_T` is the mean anomaly
at that same **midpoint** (obs start + `T/2`): the circular grid, the grid
driver and `pulsegen_gr.py` all use `anchor="midpoint"` by default, and the
remap's `tau` is already measured from the midpoint, so `t_offset_s = 0`. The
legacy `start` anchor (the `.inf` epoch) shifts `A_T` back by `t_offset_s =
T/2` (`0.5*N*dt`) and is kept only for injections/comparison. The polynomial demod is **always**
midpoint-anchored (`demodulate_series` ignores `anchor` there), so an ajs grid
derived with `-anchor start` is for comparison only and is not valid demod
input: its coefficients refer to the wrong origin. The midpoint is also what keeps the
circular `omega_b` lever at `T/2` instead of `T` (Section 5.4), halving the
dominant axis.

---

## 4. Grid sizing, model `ajs`

This is the derivation of the polynomial grid. It proceeds in eight steps.

### 4.1 Step 1 — fit each segment's truth with a cubic

For every scanned orbit $(p_o,\sin i,e,\omega_\text{peri})$ and every segment
start phase $t_0$, the code fits the model

$$
v_\text{model}(\tau) \\;=\\; v_0 + a\\,\tau + \frac{j}{2}\tau^2 + \frac{s}{6}\tau^3
$$

to $v_\text{true}(t)$ over the span $[t_0,\\,t_0+T]$, sampled at `nsamp` points
(`fit_velocity_model`). The anchor is

$$
t_\text{anchor} =
\begin{cases}
t_0, & \texttt{anchor}=\texttt{start},\cr
t_0 + T/2, & \texttt{anchor}=\texttt{midpoint},
\end{cases}
\qquad \tau = t - t_\text{anchor}.
$$

The design matrix has a column of ones ($v_0$, always fitted) followed by the
columns $\tau^{k+1}/k!$ for the masked axes ($k=0,1,2$ for accel, jerk, snap;
`FACTORIALS = [1,2,6]`). Columns are divided by their max abs value for
conditioning, `lstsq` solves, and the coefficients are un-scaled. The fit returns
the max instantaneous residual `trunc` (m/s; stored in the record, not used downstream) and the
**integrated** truncation

$$
\Lambda \\;=\\; \texttt{integrated\\_trunc}(\text{resid},\tau)
       \\;=\\; \max_\tau\Bigl|\int_{\tau_\text{first}}^{\tau}\text{resid}\\,\mathrm{d}\tau'\Bigr|
       \quad [\text{m}],
$$

the accumulated path-length error of the best cubic against the truth
(`integrated_trunc`). In cycles this is $\Lambda/(c\\,p_0)$.

Why is $v_0$ always fitted but never gridded? A constant velocity offset is just a
constant shift of the apparent spin frequency, which the coherent search's period
axis absorbs. Only the $\tau$-dependent part of the mismatch costs phase
coherence. Hence the grid covers only $(a,j,s)$.

### 4.2 Step 2 — feasibility gate

A segment is **feasible** iff its own best-fit cubic already accumulates less than
the budget (`mark_feasible`):

$$
\Lambda < \varepsilon_m \qquad (\texttt{trunc\\_phase} < \texttt{phase\\_tol\\_m}).
$$

No grid density can repair a truncation larger than the budget, so infeasible segments are simply not
claimed by the grid.
The cubic is the least-squares fit to the segment's *velocity*, not the
minimax fit on $\Lambda$, so the gate is conservative: it can reject a segment
that some other cubic would pass, never the reverse.

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
* keep $n_\text{keep} = \max\\!\big(\mathrm{round}(n_\text{feas}(1-\text{drop\\_pct}/100)),\\,1\big)$,
* the rest stay `feasible` but lose `kept`.

Dropping the worst segments is what lets the grid be coarse: the worst **kept**
segment now sets the residual budget. Dropped phases are simply not claimed (the
plots report the claimed fraction so this cost is explicit).

### 4.5 Step 5 — lever arms

Let the grid point differ from the true best-fit coefficients by
$\delta a,\delta j,\delta s$, each at most half a grid spacing. The extra velocity
is $\delta v = \delta a\\,\tau + \tfrac{\delta j}{2}\tau^2 + \tfrac{\delta s}{6}\tau^3$,
and the extra path length accumulated from the span start is

$$
E(\tau) \\;=\\; \int_{\tau_\text{first}}^{\tau}\delta v\\,\mathrm{d}\tau'
\\;=\\; \delta a\\,\frac{\tau^2-\tau_\text{first}^2}{2}
    + \delta j\\,\frac{\tau^3-\tau_\text{first}^3}{6}
    + \delta s\\,\frac{\tau^4-\tau_\text{first}^4}{24}.
$$

The worst case over the span bounds the path error by

$$
|E| \\;\le\\; |\delta a|\\,L_a + |\delta j|\\,L_j + |\delta s|\\,L_s,
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
\texttt{start}:[T^2/2,\\;T^3/6,\\;T^4/24],\qquad
\texttt{midpoint}:[T^2/8,\\;T^3/24,\\;T^4/384].
$$

A unit coefficient offset on an axis therefore costs at most $L_x$ metres of path
error, i.e. $L_x/(c\\,p_0)$ cycles.

### 4.6 Step 6 — splitting the leftover budget

The worst kept segment already spends $\Lambda_\text{max}$ metres of the budget.
What remains for grid mismatch is

$$
B_m = \varepsilon_m - \Lambda_\text{max}
\qquad(\texttt{budget\\_m}),
$$

which is positive because every kept segment passed the feasibility gate. The
grid is chosen by `allocate_spacings(B_m, [L_a,L_j,L_s], ranges, mask)`:

1. Start with `active = mask` (axes present in the model).
2. **Saturated axes (fixed cost).** An axis that is in the mask but already
   deactivated ("**saturated**", below) is covered by a **single** trial, placed at
   the range midpoint (`grids_from_spec`). Its worst-case offset is half the range,
   so it costs a *fixed* amount of path error
   $$
   F = \sum_{x\ \text{saturated}} \frac{\text{hi}_x-\text{lo}_x}{2}\\,L_x .
   $$
   Axes that are off (not in `mask`) cost nothing.
3. If $B_m - F \le 0$ the budget cannot even cover the fixed costs → return
   infeasible (`None`).
4. **Equal split.** The remaining budget is split *equally* over the
   $n_\text{act}$ still-active axes:
   $$
   \epsilon_x \\;=\\; \frac{B_m - F}{n_\text{act}} \quad\text{(same for every active axis).}
   $$
   Each active axis may then be off by at most half a spacing, and the resulting
   path error is at most $\epsilon_x$:
   $$
   \frac{D_x}{2}\\,L_x = \epsilon_x
   \quad\Longrightarrow\quad
   \boxed{\\,D_x = \frac{2\\,\epsilon_x}{L_x}\\,}.
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

trials at $\text{lo}_x + D_x\cdot\\{0,1,\dots,n_x-1\\}$ (`grids_from_spec`). The `+1`
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
q(\tau) = \delta a\\,\tau + \frac{\delta j}{2}\tau^2 + \frac{\delta s}{6}\tau^3,
\qquad \delta x \in \pm D_x/2,
$$

and computes the exact accumulated phase
$\texttt{integrated\\_trunc}(\text{resid}_v+q,\tau)/(c\\,p_0)$. It checks all **8
corners** of the cell $\delta x=\pm D_x/2$ (worst-case signs), plus `n_mc` random
draws uniform in $[-D_x/2,D_x/2]^3$. The guarantee holds iff

$$
\max_\text{corners}\bigl(\text{integrated cycles}\bigr) \le \varepsilon\\,(1+10^{-9}).
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
$c\\,p_0=2.99792458\times10^8$ m):

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
gate on. Every orbit is feasible and **nothing is dropped** (`-drop_pct` is
ajs-only): the cost knob is the budget $\varepsilon$ itself, and the count
scales as $\varepsilon^{-3}$ in every band. The only error at a grid point is
the **parameter mismatch** between truth and grid.

### 5.2 The residual spin phase

The orbit-induced residual spin phase (cycles) — the quantity the budget bounds —
is the difference of the delays over $p_0$:

$$
\Phi(t) \\;=\\; \frac{x}{p_0}\sin\!\bigl(\omega_b (t - T/2) + A_T\bigr),
\qquad t\in[0,T],
$$

with $t$ the **absolute** time since the observation start and $A_T$ the mean
anomaly at the **midpoint**, matching `demodulate_series`'s default
`anchor="midpoint"` (Section 3).
Equivalently $\Phi = (1/p_0)\int (v/c)\\,\mathrm{d}t$.

### 5.3 First-order perturbation: the three amplitudes

A grid point differs from the truth by $(\delta x,\delta A_T,\delta\omega_b)$.
With $t'=t-T/2\in[-T/2,T/2]$, to first order,

$$
\delta\Phi(t) \\;=\\;
\underbrace{\frac{\delta x}{p_0}}_{\textstyle \alpha}\sin\varphi
\\;+\\;\frac{x}{p_0}\bigl(\delta A_T + t'\\,\delta\omega_b\bigr)\cos\varphi,
\qquad \varphi=\omega_b t' + A_T .
$$

The worst-case offsets are half a spacing, so the code defines the three
amplitudes

$$
\alpha \\;=\\; \frac{D_x/2}{p_0},\qquad
\beta \\;=\\; \frac{x_\text{lever}\\,(D_{A_T}/2)}{p_0},\qquad
\gamma \\;=\\; \frac{x_\text{lever}\\,(D_{\omega}/2)\,(T/2)}{p_0},
$$

and writes

$$
\delta\Phi(t) \\;=\\; \alpha\sin\varphi + \Bigl(\beta+\gamma\\,\frac{2t'}{T}\Bigr)\cos\varphi .
$$

Two points on the definitions:

* The $A_T$ and $\omega_b$ terms carry $x$, so they are evaluated at
  $x_\text{lever}=x_j+D_x/2$, the largest $x$ that snaps to row $x_j$
  (Section 5.5). Each row is spaced for its own lever, not the grid top.
* The $\omega_b$ lever is $T/2$, not $T$, because $A_T$ is anchored at the
  observation **midpoint**, so $|t'|\le T/2$. Anchoring at the start would
  double the $\omega_b$ axis for the same budget (Section 5.4).

### 5.4 The joint bound and the volume-optimal split

At a fixed $t'$ we must bound $|\alpha\sin\varphi + B\cos\varphi|$ over the unknown
orbital phase $\varphi$, where $B=\beta+\gamma\\,2t'/T$. The identity

$$
\max_\varphi\bigl|\alpha\sin\varphi + B\cos\varphi\bigr| = \sqrt{\alpha^2+B^2}
$$

(for any $\varphi$) and $B\le\beta+\gamma$ (both non-negative, worst at
$|t'|=T/2$) give the exact worst case

$$
\max_{t',\varphi}|\delta\Phi| = \sqrt{\alpha^2+(\beta+\gamma)^2}.
$$

Requiring this to be within the budget gives the **joint constraint**

$$
\boxed{\\;\alpha^2 + (\beta+\gamma)^2 \\;\le\\; \varepsilon^2\\;}.
$$

We now choose $\alpha,\beta,\gamma$ to maximise the cell **volume**
$D_x\\,D_{A_T}\\,D_\omega$. Since each spacing is proportional to its amplitude
($D_x\propto\alpha$, $D_{A_T}\propto\beta$, $D_\omega\propto\gamma$), this is
maximising $\alpha\beta\gamma$ subject to $\alpha^2+(\beta+\gamma)^2=\varepsilon^2$.

1. For fixed $\alpha$ and fixed $s=\beta+\gamma$, the product
   $\beta\gamma$ is maximised when $\beta=\gamma=s/2$ (AM–GM).
2. So maximise $\alpha\\,(s/2)^2$ subject to $\alpha^2+s^2=\varepsilon^2$. Set
   $f=\alpha s^2$ and use a Lagrange multiplier:
   $$
   \frac{\partial f}{\partial\alpha}=s^2=2\lambda\alpha,\qquad
   \frac{\partial f}{\partial s}=2\alpha s=2\lambda s
   \\;\Longrightarrow\\; \alpha=\lambda,\quad s=\alpha\sqrt2 .
   $$
   Then $\alpha^2+s^2 = 3\alpha^2=\varepsilon^2$.

Therefore

$$
\alpha_\text{bud} = \frac{\varepsilon}{\sqrt3},\qquad
\beta_\text{bud}=\gamma_\text{bud}=\frac{\varepsilon}{\sqrt6}
\quad(\text{code: \texttt{a\\_bud}, \texttt{bc\\_bud}}).
$$

(The old per-axis split $\alpha=\beta=\gamma=\varepsilon/3$, which only satisfies
the triangle inequality, gives a cell volume smaller by a factor
$\frac{27}{6\sqrt3}\approx 2.6$.)

### 5.5 Spacings, period bands and the concrete grid

Inverting the amplitude definitions ($\delta = D/2$ at the worst corner) gives the
raw-bound spacings (`_band_rows`), each multiplied by the band's stretch
$s_x, s_A, s_\omega\ge1$ from §5.6 (all 1 under `-circ_strict`):

$$
D_x = 2\\,\alpha_\text{bud}\\,p_0\\,s_x,\qquad
D_{A_T} = \frac{B\\,p_0\\,s_A}{x_\text{lever}},\qquad
D_\omega = \frac{B\\,p_0\\,s_\omega}{x_\text{lever}\\,(T/2)},\qquad
B=\beta_\text{bud}+\gamma_\text{bud}=\varepsilon\sqrt{2/3}.
$$

The grid (`derive_circular_grid`):

* **Period bands.** The `-p_o` ladder is split into one band per ladder period,
  edges at the geometric midpoints between neighbours. Each band is its own
  sub-grid, so the coarse steps a long-period band can afford are not forced on
  the short ones.
* **x rows.** In a band, $x$ runs over the physical range
  $x = k\\,\sin i\\,\omega_b^{-2/3}$ for $\sin i\in$ `-sin_i` and $\omega_b$ in the
  band, in steps of $D_x$ (rows $x_j$). $x$, not $\sin i$, is the coordinate,
  because it decouples the amplitude from the period ($a_p\propto P_b^{2/3}$).
* **$\omega_b$ per row.** Uniform in $\omega_b=2\pi/p_{o,s}$ (the phase error is
  linear in $\delta\omega_b$), but only over the $\omega_b$ some $\sin i$ in range
  can reach for that row's $x$ — the physical band, not a box.
* **$(\omega_b, A_T)$ checkerboard.** The $(A_T,\omega)$ cell is the **diamond**
  $|\beta'|+|\gamma'|\le B$, which tiles on a checkerboard: steps $D_{A_T}$ and
  $D_\omega$, keeping only points with $(i_\omega+i_{A_T})$ even. Half the points
  of the inscribed rectangle for the same guarantee. $n_{A_T}=2\lceil\pi/D_{A_T}\rceil$
  is even so the parity survives the $2\pi$ wrap; every consumer applies the
  rule (`write_circular_csv`, `test_circular_grid.py`).
* $A_T$ is the mean anomaly at the observation **midpoint**; the demod is driven
  with `anchor="midpoint"` to match.

`write_circular_outputs` writes the CSV in the demod's own coordinates: `pb` in
**days** ($2\pi/\omega_b/86400$), `x` in light-seconds, `at` $=A_T$ in radians at
the observation **midpoint**. `circular_share_plot` writes
`circular_template_share.png`: the share of templates per $(p_o,\sin i)$ (each
template counted towards the `-sin_i` value nearest its implied
$\sin i = x\\,\omega_b^{2/3}/k$) and the count per band. It is written even under
`-no_write`.

### 5.6 What the spin-frequency search absorbs (the stretch)

§5.4 bounds the **raw** $\max_t|\delta\Phi|$. The demodulated series is then
searched over spin frequency, which takes part of the residual for free:

* a **constant** phase offset does not change the folded power;
* a **linear** drift $c_1 t$ is a spin-frequency offset $c_1$, which the
  frequency search finds.

So the residual that costs S/N is what is left after the best line,
$\min_{c_0,c_1}\max_t|R(t)-c_0-c_1t'|$. For short orbits ($\omega_bT/2\gg1$) $R$
oscillates many times and a line removes almost nothing. When the observation
covers less than about one orbit, a grid offset mostly looks like a wrong spin
phase and period. Expanding about the midpoint,

$$
\frac{x}{p_0}\sin(\omega_b t'+A_T)\approx\frac{x}{p_0}\Bigl[\sin A_T+\omega_bt'\cos A_T-\tfrac12(\omega_bt')^2\sin A_T+\dots\Bigr],
$$

so only the curvature and higher terms survive, of relative size
$\sim(\omega_bT/2)^2/2$. The raw bound over-counts by exactly that, and those are
the most expensive bands (the count per log-period grows $\propto P_b$).

`search_absorbed_max(R, t')` evaluates the surviving residual as the better of no
line and the least-squares line. Both are lines the search can land on, so it
bounds the minimax line from above. `_band_stretch` then sets each band's
multipliers:

1. **Seed.** $L_A=\max_{A_T,\omega_b}$ `search_absorbed_max`$(\cos\varphi)$ (the
   $x$ and $A_T$ perturbations, $\sin\varphi$ and $\cos\varphi$, give the same set
   over all $A_T$), and $L_\omega$ the same for $(2t'/T)\cos\varphi$. Start from
   $s_x=s_A=1/L_A$, $s_\omega=1/L_\omega$, clipped to $[1, 20]$.
2. **Certify.** `_cell_worst` evaluates the **exact, nonlinear** residual on the
   stretched cell boundary — truth half an $x$ step either side of a row (at both
   ends of the band's $x$ range), on the $(A_T,\omega)$ diamond edge in all four
   quadrants, scanned over $A_T$ and five $\omega_b$ across the band, with the
   $A_T$ step as actually built ($2\pi/n_{A_T}$, never more than $\pi$; $\sin$ is
   periodic, so a larger nominal step would certify nothing).
3. **Shrink** all three multipliers by $0.98\\,\varepsilon/\text{worst}$ (floored
   at 1) and re-certify, up to 8 times. The loop stops there without a final
   check; `test_circular_grid.py` (§5.7) is what certifies the result.

The floor at 1 means no band is ever denser than the raw-bound grid, which §5.4
already guarantees. The cap at 20 keeps the absorbed drift modest; past it the
orbit is ajs territory anyway. The volume-optimal $\alpha/B$ split of §5.4 is
kept and only shrunk uniformly, not re-optimised.

The report prints each band's stretch and the **raw** $\max|R|$ at its cell
boundary, the drift the search absorbs. It shows up as a spin-frequency offset
of up to $2\,$raw$/T$ Hz ($2\,$raw Fourier bins) between the reported and true
spin frequency. The YAML records `search_absorbs_linear` and each band's
`stretch_x_AT_omega`.

**`-circ_strict`** turns this off and bounds the raw $\max|R|$. Use it if the
demodulated series is folded at a **fixed** spin frequency rather than searched.

Measured, $T=19209$ s, $p_0=0.01$ s, $\varepsilon=0.2$, the 9-point ladder
`np.geomspace(0.03, 1.26, 9)` d, i.e.
`-p_o 8.214e-05 0.0001311 0.0002091 0.0003336 0.0005323 0.0008493 0.001355 0.002162 0.00345`
(yr). Every band with $\omega_bT/2 > 1.12$ (here, below 0.63 d) stays at
stretch 1:

| band [d] | $\omega_bT/2$ | stretch $x/A_T/\omega$ | templates, raw bound | templates, absorbed |
|---|---|---|---|---|
| 0.03–0.63 (7 bands) | 1.12–23.3 | 1/1/1 | 1.686e10 | 1.686e10 |
| 0.63–1.00 | 0.70–1.12 | 1.88/1.88/1.16 | 1.054e10 | 2.584e9 |
| 1.00–1.26 | 0.55–0.70 | 4.49/4.49/1.63 | 7.443e9 | 2.276e8 |
| **total** | | | **3.484e10** | **1.967e10** |

### 5.7 Validation

`test_circular_grid.py` Monte-Carlos the guarantee directly, without
linearisation. It imports the grid code (so it cannot drift), samples random
circular orbits uniformly in the covered ranges (log-uniform $p_o$, uniform
$\sin i$, uniform $A_T$), snaps each to the nearest point of its band (nearest
$x$ row, then the L1-nearest checkerboard point), and computes the **exact**
residual spin phase
$R(t)=\frac{x_t}{p_0}\sin(\omega_t t'+A_t)-\frac{x_g}{p_0}\sin(\omega_g t'+A_g)$ on
a fine time grid. It scores `search_absorbed_max(R)` (the raw $\max|R|$ under
`-circ_strict`) as a fraction of the band budget, and asserts that

* the worst over all Monte-Carlo trials is within budget,
* each band's deterministic worst corner (top row, half a step on every axis,
  $A_T$ scanned) is within budget on the exact nonlinear $R$,
* a random scan of cell-boundary points (which uniform sampling almost never
  hits), restricted to truths inside the band, is within budget.

It passes in both modes on the bundled `inj.inf` config (worst 0.986 of budget;
0.996 with `-circ_strict`) and on $T=19209$ s, $p_0=0.01$ s, $\varepsilon=0.2$
(worst 0.9998; 0.9999 with `-circ_strict`).

### 5.8 Worked example (circular)

On the bundled `inj.inf` (T = 1509.95 s, $p_0=1$ s, $\varepsilon=0.1$, default
4 h–1 d ladder), `python nsns_grid.py -inf inj.inf -mode circular` gives:

```
       p_o band [d]   templates   share   stretch x/A_T/omega raw max|R|
  0.1667-0.1808       2.200e+01    2.6%  19.24  19.24   4.46        1.56
  0.1808-0.2128       2.200e+01    2.6%  20.00  20.00   5.02        1.67
  0.2128-0.2504       2.200e+01    2.6%  20.00  20.00   5.89        1.76
  0.2504-0.2947       2.200e+01    2.6%  20.00  20.00   6.91        1.72
  0.2947-0.3469       2.200e+01    2.6%  20.00  20.00   8.12         1.7
  0.3469-0.4082       2.200e+01    2.6%  19.04  19.04   9.08        1.65
  0.4082-0.4805       4.800e+01    5.7%  17.51  17.51   9.82        1.58
  0.4805-0.5655       5.000e+01    5.9%  15.75  15.75  10.38        1.45
  0.5655-0.6655       8.400e+01   10.0%  13.89  13.89  10.77        1.28
  0.6655-0.7832       1.340e+02   15.9%  10.69  10.69   9.75        1.01
  0.7832-0.9218       1.980e+02   23.5%   8.77   8.77   8.77       0.853
  0.9218-1            1.980e+02   23.5%   9.04   9.04   9.04        0.88
              total   8.440e+02
```

against **7.015e+04** with `-circ_strict` (83x). The raw drift tops out at
1.76 cycles, i.e. a spin-frequency offset of up to 3.5 Fourier bins (2.3 mHz)
absorbed by the search.

---

## 6. Grid sizing, model `hybrid`

Hybrid runs **one** full-ladder ajs scan (the machinery of §4), uses it to place
`p_break`, then builds *two* sub-grids: a circular grid (§5) for the periods
below `p_break`, and an ajs grid (§4) for the scanned records at/above it.

### 6.1 Placing `p_break`

For each scanned `p_o`, `ajs_coverage_by_p_o` computes the **mean feasibility**
over the `e = 0` records, pooled across the scanned `sin_i × omega_peri` combos:

$$
\text{cov}(p_o) \\;=\\;
\frac{\\#\\{\text{e=0 segments at } p_o \text{ that pass the phase gate and the caps}\\}}
     {\\#\\{\text{e=0 segments at } p_o\\}} .
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
p_{\text{ladder}[0]}, & \text{no } p_o \text{ is bad (all ajs)}, \cr
p_{\text{ladder}[i_\text{bad}+1]}, & i_\text{bad} = \text{index of the last bad } p_o, \cr
p_{\max}\\,(1+10^{-9}), & \text{the longest } p_o \text{ is still bad (all circular).}
\end{cases}
$$

Orbits are then split by `p_o < p_break` (circular) and `p_o ≥ p_break` (ajs).
A manual `-p_break` (in yr) overrides the search entirely.

### 6.2 The two sub-grids

* **Circular side** (`derive_circular_grid` with `p_o` filtered to `< p_break`):
  unchanged from §5, except the period ladder is restricted. Nothing is
  dropped here; `-drop_pct` applies to the ajs side only. The §5.6 stretch only
  helps bands with $\omega_bT/2\lesssim1$ ($P_b\gtrsim\pi T$; ≳ 0.63 d at
  $T=19209$ s), so it barely changes a circular side whose `p_break` is below
  that, and gives the §5.6 savings when `p_break` lies above it.
  The top circular band ends at the last circular ladder period, not at
  `p_break`: orbits between that period and `p_break` are deliberately claimed
  by neither sub-grid.
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
and `ajs:`. `--demod-grid` accepts both CSVs in one invocation (it reads each
file's own header), and `run_nsns_sweep.sh MODE=hybrid` reads each point's model
from its CSV header so both filename patterns coexist.

### 6.4 Worked example (hybrid)

On the bundled `inj.inf` with `-n_phase 24 -nsamp 1000` (T = 1509.95 s), the ajs
coverage is:

```
      p_o [yr]    p_o [d]   ajs coverage
   0.000456308    0.16667         100.0%
   0.000537031    0.19615         100.0%
   0.000632034    0.23085         100.0%
   0.000743844    0.27169         100.0%
   0.000875432    0.31975         100.0%
     0.0010303    0.37632         100.0%
    0.00121256    0.44289         100.0%
    0.00142707    0.52124         100.0%
    0.00167953    0.61345         100.0%
    0.00197664    0.72197         100.0%
    0.00232632    0.84969         100.0%
    0.00273785          1         100.0%
```

With the default 4 h floor every scanned orbit is ≳ 9.5 × T_obs, so the
coverage dip near `p_o ~ T_obs` described in §6.1 falls below the ladder and
every point clears `-break_coverage`. The auto switch therefore keeps `p_break`
at the first ladder period (0.000456308 yr) and there is **no circular
sub-grid**: the whole band is gridded ajs. (A longer observation pushes `T_obs`
up toward 4 h and the split reappears.) The ajs side then runs on all 43200
scanned segments, caps from the 0.16667 d orbit (`|accel| ≤ 1.071e+04 m/s²`,
`|jerk| ≤ 203.7 m/s³`, `|snap| ≤ 7.531 m/s⁴`); `trunc_max` is 0.0000 cycles, so
only the accel axis stays active and the grid collapses to 3 templates.

---

## 7. Outputs

Every mode writes machine-readable files for the sweep:

* **`<outstem>.yaml`** — the spec; its keys depend on the mode:
  * ajs: ranges, spacings, `active`, caps, tolerances, `terms`, `anchor`,
    `trunc_max`, `budget_m`, `eps`, `guarantee_ok` and counts. There is no
    `mode` key; a spec without one is ajs.
  * circular: `mode`, tolerances, `A_T_anchor` (`midpoint`), `lattice`,
    `search_absorbs_linear`, per-band `bands` (edges, `stretch_x_AT_omega`,
    `n_trials`) and `n_trials`. It has no ranges/spacings/caps: the grid is
    per band and per row.
  * hybrid: `mode`, `p_break`, `break_coverage`, `coverage_by_p_o`, and both
    sub-specs nested under `circular:` and `ajs:`.
* **`<outstem>.csv`** — every concrete trial, `%.10g`, one per line, **LF** line
  endings (bash builds output filenames from these literal strings):
  * ajs header `accel,jerk,snap`, one row per $(a,j,s)$;
  * circular header `pb,x,at`, one row per $(p_b,x,A_T)$.
  * Hybrid writes **two** files, `<outstem>_circular.csv` and `<outstem>_ajs.csv`.
* Plots under `-plot_subdir` (default `grid_size_plots/`):
  `corner_feasible_fraction.png`, `claimed_fraction.png` (ajs only),
  `circular_template_share.png` (circular) and
  `hybrid_circular_template_share.png` (hybrid, which writes no ajs plots).
  The share plot is written even under `-no_write`.
  With `-plot`, ajs also writes one phase / best-fit figure per orbit.

`grids_from_spec` and `write_grid_outputs` (ajs) build the Cartesian product and
write it; `write_circular_outputs` does the same for circular, converting
$\omega_b\to p_b$ (days) for the CSV. Hybrid calls `write_circular_csv` for
`<outstem>_circular.csv` and `write_grid_outputs` for `<outstem>_ajs.csv` (with
its YAML skipped), then writes the combined YAML itself.

---

## 8. Directory reference

| file | what it does |
|---|---|
| `nsns_grid.py` | **Grid derivation** (ajs, circular, hybrid), YAML/CSV/plots. |
| `src/demod.jl` (package) | The exact time-remap resamplers, PRESTO's `.fft` packing, `presto_deredden!` and `demodulate_series`. Was `demod/resample.jl`; moved into the package so the in-memory driver uses the same code. |
| `demod_dat.jl` | On-disk wrapper: apply the remap to one `.dat`/`.inf`; CLI + `demod_file`. |
| `demod_grid.jl` | Batch, on disk: demodulate one observation at every row of a grid CSV, in one Julia process. |
| `src/demodsearch.jl` (package) | The in-memory driver behind `bin/coherent_search.jl --demod-grid` (Section 11). |
| `pulsegen_gr.py` | Inject a full Damour–Deruelle binary pulsar into a `.dat`/`.inf` (+ `_truth.yaml`). |
| `run_demod_search.sh` | End-to-end: inject → derive grid → nearest grid point → demod → search. |
| `test_circular_demod.sh` | Same, for circular mode, with assertions on recovery. *Untracked (local only).* |
| `run_nsns_sweep.sh` | Full sweep: derive grid → demodulate **every** point → FFT → search all; CPU-parallel or one GPU invocation. |
| `combine_cohout.py` | Merge all per-point `.cohout` into one candidate list (optional plot + truth crosshair). |
| `test_circular_grid.py` | Monte Carlo / adversarial validation of the circular grid's phase guarantee (search-absorbed residual; `-circ_strict` for raw). |
| `nsns_grid.yaml`, `nsns_grid.csv` | Example derived grid (local only, untracked). |
| `inj.dat`, `inj.inf`, `demod_out/`, `grid_size_plots/`, `plots/`, `nsns_sweep*/` | Test observation and outputs (local only: `inj.dat` is gitignored, the rest untracked; `run_nsns_sweep.sh`'s default `INF` points at `inj.inf`, so set `INF` on a fresh clone). |

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
MODE=circular bash demod/run_nsns_sweep.sh
MODE=hybrid bash demod/run_nsns_sweep.sh                    # circular + ajs
MODE=hybrid P_BREAK=0.02 bash demod/run_nsns_sweep.sh       # manual p_break [yr]
MODE=hybrid PO="0.0000038 0.00001 0.1 100" bash demod/run_nsns_sweep.sh
GRID_CSV=/path/to/grid.csv bash demod/run_nsns_sweep.sh   # skip derivation

# Grid only:
python demod/nsns_grid.py -inf OBS.inf -p0 1.0 -phase_tol_cycles 0.1
python demod/nsns_grid.py -inf OBS.inf -mode circular             # -circ_strict: raw bound
python demod/nsns_grid.py -inf OBS.inf -mode hybrid          # auto p_break
python demod/nsns_grid.py -inf OBS.inf -mode hybrid -p_break 0.02 -break_coverage 60

# Validate the circular guarantee:
python demod/test_circular_grid.py -inf OBS.inf -n_mc 2000
```

The in-memory driver replaces steps 2 and 3 for a whole sweep -- one process, one
read of the `.dat`, and only `.cohout` written (**no** `realfft`/`rednoise`
needed at run time):

```sh
julia --project=. -t auto bin/coherent_search.jl OBS.dat \
      --demod-grid GRID.csv --outdir OUT --threshold 6
```

`--demod-grid` may be repeated (hybrid grids emit one CSV per model).  It rejects
`-o`, `--plot` and `--plotstem`; plot afterwards from the `.cohout` files with
`bin/plot_candidates.jl`.  Section 11 has the details.

Requires Julia with the `CoherentSearch.jl` project, a Python with
numpy/scipy/matplotlib/yaml, PRESTO's `realfft`/`rednoise` (shell pipeline
only), and the sibling `FFA_stacking` checkout. The scripts do **not** search
`PATH`: `PYTHON`, `REALFFT` and `REDNOISE` default to `/home/fadong/anaconda3/bin/`,
so set them on another host. `run_nsns_sweep.sh` searches on the CPU by default;
`GPU=1` (exactly) switches to `--gpu`.

---

## 10. Tests

* `test_circular_grid.py` — verifies the circular phase-tolerance guarantee
  (Section 5.7). Exits non-zero on failure so it can gate a run.
* `test_circular_demod.sh` (untracked, local only) — end-to-end circular inject → grid → demod → search,
  asserting the pulsar is recovered near $1/p_0$ above a threshold S/N.
* `run_demod_search.sh` — end-to-end ajs (and circular) smoke test.

---

## 11. In-memory demodulation: `bin/coherent_search.jl --demod-grid`

The shell pipeline writes and re-reads about five files per grid point -- a
`.dat`, an `.inf`, a `.fft`, and a `_red.fft` plus its `.inf` -- and every one of
them is a pure intermediate.  A 220-point sweep writes ~1100 files, reads them
all back, and pays a Julia process start-up per point.

`--demod-grid` does the whole chain inside one process, per grid point:

```
raw .dat held in RAM  ->  exact time remap  ->  PRESTO realfft  ->  rednoise
                      ->  coherent search     ->  <stem>_demod_..._red.cohout
```

The observation is read **once**; peak memory is the raw series plus one
demodulated series plus one amplitude array plus a bounded FFT-plan cache,
independent of the grid size.  The remap is evaluated on the fly rather than
through N-long `tau`/`polyval`/index temporaries, so a point allocates ~8 bytes
per sample (the de-meaned copy and the remapped series) instead of ~88; at the
192M-sample fixture that is ~1.4 GiB rather than ~16 GiB of garbage per grid
point.  Nothing but the `.cohout` files is written.

```sh
julia --project=. -t auto bin/coherent_search.jl OBS.dat \
      --demod-grid GRID.csv --outdir OUT --threshold 6
# hybrid grids: repeat the flag, one CSV per model
julia --project=. -t auto bin/coherent_search.jl OBS.dat \
      --demod-grid circgrid.csv --demod-grid ajsgrid.csv --outdir OUT
```

* Every search option applies as it would to an `.fft` file: the driver reuses
  the CLI's own `search_one`, so `--nharms`, `--maxdecim`, `--ncands`,
  `--metricstats`, `--sigma`, `--precision`, `--gpu` and the duty-cycle
  measurement all behave identically.
* `-o`/`--outputfilenm`, `--plot` and `--plotstem` are **rejected** -- each names
  a single output a sweep would clobber, and deferred plotting would hold every
  grid point's amplitudes.  Plot afterwards from the `.cohout` files with
  `bin/plot_candidates.jl`.
* Output is `<stem>_demod_a<A>_j<J>_s<S>_red.cohout` (ajs) or
  `<stem>_demod_pb<Pb>_x<X>_at<AT>_red.cohout` (circular), where the parameter
  text is the **literal CSV string**.  The `_red` infix is deliberate:
  `combine_cohout.py`'s `MODELS` regexes end in `_red\.cohout$`, so the existing
  sift/combine tooling works unchanged.
* `--rednoise-startwidth` / `--rednoise-endwidth` / `--rednoise-endfreq` pass
  PRESTO's rednoise knobs (defaults 6 / 100 / 6.0).
* **`--demod-concurrency N` searches N grid points at once** (0 = one slot per
  Julia thread, 1 = the purely sequential driver).  Each concurrent search is
  pinned to `nthreads()/N` threads.  This is the faster choice: a demodulation is
  serial (~25 ms during which every worker otherwise idles) and the search's own
  chunk parallelism flattens well below `nthreads()` — measured `-t 16`,
  `--nharms 60`, 64 points: **14.6 s sequential vs 10.2 s concurrent (1.43x)**,
  with the optimum at N ≈ `nthreads()` and the default within 3% of it.  At
  `-t 32` on the same 64 points the curve is flat from 8 to 32 slots
  (10.3/10.4/**10.0**/10.2/10.3 s), so the default is not a knife edge.  With
  `N > 1` the demodulated points are searched on tasks; the `.cohout` files are
  written first-in-first-out as slots are reclaimed, i.e. in grid order. `--gpu`
  forces `N = 1` (the device is one resource).
  - **Each concurrent slot owns a separate `SearchCache`.**  Workspaces are
    indexed by task index inside `_search_region!`, so two searches sharing one
    would have their chunks writing into the same workspaces — a silent data
    race.  A per-slot pool fixes that and also amortises FFTW planning across the
    sweep.  Candidates are bit-identical to the sequential run (pinned by
    `test/test_demod.jl`).
  - **The observation is still read once, and Julia starts once.**  The parallel
    axis is *inside* this process, deliberately not one process per grid point:
    that alternative was measured at 1.06–1.13x (better scaling, but it re-reads
    the `.dat` and re-pays ~0.85 s/point of start-up) and it costs real file
    I/O.  See below.
* The FFTW plan for each demodulated length uses the same switch as PRESTO's
  `fftwcall()`: `FFTW_ESTIMATE` above 16384 complex points and `FFTW_MEASURE`
  at or below it.  Real demodulations are far above the threshold, so they pay
  only `ESTIMATE`; for `N = 6e6` that is 15 ms rather than the 35.5 s a
  `MEASURE` plan costs.  The plan is cached per demodulated length because each
  grid point lands on its own `N'` (varying by hundreds of samples), but the
  cache holds a **byte budget**, not one plan per length forever: FFTW's
  twiddle tables live outside the Julia heap at roughly 4–8 bytes per sample
  per plan (the fallback path keeps the planned array too), so an unbounded
  per-`N` cache pinned ~0.75 GB per grid point on the 192M-sample fixture and
  a long sweep ran out of RAM.  Plans are now destroyed as the budget is
  exceeded (see `PRESTO_FFT_PLAN_CACHE_BYTES` in `src/demod.jl`), which bounds
  peak memory independently of the grid size at the cost of re-planning a point
  whose length is not currently cached.

**Why not one process per grid point.**  `demod/run_nsns_sweep.sh` does exactly
that (and `bin/parallel_search.py` for `.fft` files), and it does scale better
per core.  It is the wrong trade here: measured against the in-process
concurrent driver it wins only 1.06–1.13x, because it re-reads the `.dat` and
re-pays Julia's start-up, plan building and wisdom import for every point
(~0.85 s against a ~1.8 s point, where the in-process driver's fixed cost is
paid once).  The `.dat` read itself is negligible (~0.11 ms warm for a 600 kB
series) — it is the start-up that costs.

**Agreement with the file-based chain.**  The same demodulated series run both
ways now produces the same candidate file, bit for bit: the in-memory
`realfft` port matches PRESTO's `.fft` bytes, the rednoise port matches the
`_red.fft` bytes, and the search reads that dereddened array.  For a real
demodulated observation (`N/2 > 16384`, PRESTO's deterministic `ESTIMATE`
branch) `test/test_demod.jl` asserts the full `.cohout` equality against
`demod_dat.jl` -> `realfft` -> `rednoise` -> `coherent_search`.

The underlying ports, so a failure can be localised:

| step | port | agreement |
|---|---|---|
| real FFT (`presto_fft_amps`) | `realfft()` (`fastffts.c`), including its N/2-point complex FFT and double trig recursion | **bytes identical** for `N/2 > 16384`; below that PRESTO itself uses timing-dependent `MEASURE` plans, so only numerically equivalent |
| rednoise (`presto_deredden!`) | `dered_engine` (`presto/src/misc_utils.c`) | **bytes identical** |

Byte identity needs the same `libfftw3f` PRESTO links.  The port finds it from
the `realfft` executable's `ldd` output (or `Libdl.find_library`), and
`COHERENT_FFTW_LIB` overrides the choice; `CoherentSearch.presto_fftw_exact()`
reports whether a match was found.  Without it the chain still runs through
FFTW.jl's library with PRESTO's exact unpacking, but is not byte-identical.

Two details of the rednoise port are load-bearing and were measured against the
shipped binaries, not the C source's intent:

* **The median is the LOWER order statistic.**  `src/median.c` calls
  `gsl_stats_float_median`, which is documented to average the two central values
  for even `n` -- but the `libpresto.so` that `rednoise` actually links returns
  `sorted[(n+1)/2]`.  Verified by calling `libpresto.so:median` directly on sorted
  data (n = 6, 8, 100).  The mean-of-two form puts every block's median ~7% high
  on a log-distributed power spectrum, so the normalisation is 7% too strong and
  every amplitude -- and every S/N -- comes out ~7% low, silently.
* **`invsqrtf` is the Quake reciprocal-sqrt**, bit-exact (`0x5f3759df`, one Newton
  step), not `1/sqrt` -- the two differ by ~0.1%.

Both are pinned by `test/test_demod.jl`; the `$REDNOISE`-gated test compares
against PRESTO itself, so a future PRESTO whose `median` really did interpolate
would fail loudly rather than quietly shift every S/N by 7%.
