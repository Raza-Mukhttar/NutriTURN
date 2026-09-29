"""Regenerate Figures 4, 5 and 6 with the defects the authors flagged, and check for overlap.

Figure 4  the two panel headings collide at this width, and panel B cites "Eq. (3)" from an
          earlier draft; the equation is (2) in the current manuscript. Fix: widen slightly, put
          each heading over its own axes with room, and renumber.
Figure 5  the H=10 panel shows fractional year ticks, and the H=3 legend sits on the
          events/units labels. Fix: integer ticks from the actual origins, legend above the panels.
Figure 6  the "observed 0.948" label overlaps the legend in the first panel. Fix: legend above the
          panels, observed labels in reserved headroom.

Every figure is checked programmatically for text-on-text and text-on-mark collisions.

  python analyses/nfx_figs.py
"""

# ---- NutriTURN portable path bootstrap (added by the release build) ----
import os as _os, sys as _sys
_NT = _os.path.abspath(__file__)
for _ in range(5):
    _NT = _os.path.dirname(_NT)
    if _os.path.isdir(_os.path.join(_NT, 'data')) and _os.path.isdir(_os.path.join(_NT, 'scripts')):
        break
for _p in ('lib', 'experiments', 'pipeline', 'figures', ''):
    _q = _os.path.join(_NT, 'scripts', _p)
    if _q not in _sys.path:
        _sys.path.insert(0, _q)
# ------------------------------------------------------------------------
import os, sys, glob, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nfx_common as N

GP = N.GPTPRO
ASTRA = _NT
C_A, C_B, C_OBS = '#2D6CA2', '#C9A227', '#B3402F'
INK, MUTED = '#1a1a19', '#5b5b57'


def overlap_report(fig, name):
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    boxes = []
    for ax in fig.axes:
        for t in ax.texts:
            boxes.append((f'{name}:{t.get_text()[:20]}', t.get_window_extent(r)))
        if ax.get_legend() is not None:
            boxes.append((f'{name}:axes-legend', ax.get_legend().get_window_extent(r)))
    for lg in fig.legends:
        boxes.append((f'{name}:fig-legend', lg.get_window_extent(r)))
    clash = 0
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes[i][1].overlaps(boxes[j][1]):
                clash += 1
                print(f'   OVERLAP {boxes[i][0]} <-> {boxes[j][0]}')
    print(f'   {name}: {clash} text/legend collisions')
    return clash


def legend_clear(ax, leg, name):
    """Count scatter points and line samples that fall inside the legend's box."""
    fig = ax.figure
    fig.canvas.draw()
    bb = leg.get_window_extent(fig.canvas.get_renderer())
    inside = lambda q: np.sum((q[:, 0] >= bb.x0) & (q[:, 0] <= bb.x1) &
                              (q[:, 1] >= bb.y0) & (q[:, 1] <= bb.y1))
    bad = 0
    for col in ax.collections:
        off = col.get_offsets()
        if len(off):
            bad += int(inside(ax.transData.transform(off)))
    t = np.linspace(0, 1, 400)[:, None]
    for ln in ax.lines:
        xy = ax.transData.transform(ln.get_xydata())
        for i in range(len(xy) - 1):
            bad += int(inside(xy[i] + t * (xy[i + 1] - xy[i])))
    print(f'   {name}: {bad} marks under the legend')
    return bad


def fig4():
    """Analytic versus plug-in simulation, headings separated and the equation renumbered.

    Panel A recomputes both series the way the original generator does: the two analytic
    probabilities are plotted against a Binomial simulation frequency that is drawn here,
    not read from a column.  source_scores.csv carries bb_oracle but no simulation column.
    """
    from scipy.stats import binom as _binom
    import csv as _csv
    rows = list(_csv.DictReader(open(os.path.join(GP, 'results', 'issue1', 'source_scores.csv'))))
    npl = np.array([float(r['n_plus']) for r in rows])
    nmi = np.array([float(r['n_minus']) for r in rows])
    m = np.array([int(float(r['m_signed'])) for r in rows])
    oracle = np.array([float(r['bb_oracle']) for r in rows])
    s0 = np.array([float(r['operational_pre_sign']) for r in rows])
    th = np.divide(npl, np.maximum(npl + nmi, 1))
    rng = np.random.default_rng(3)
    freqB = np.full(len(rows), np.nan)
    plug = np.full(len(rows), np.nan)
    for i in range(len(rows)):
        if m[i] == 0:
            continue
        k = rng.binomial(m[i], th[i], size=1000)
        freqB[i] = float(np.mean(np.sign(2 * k - m[i]) != s0[i]))
        kk = np.arange(m[i] + 1)
        pk = _binom.pmf(kk, m[i], th[i])
        plug[i] = float(pk[np.sign(2 * kk - m[i]) != s0[i]].sum())
    ok = np.isfinite(freqB) & np.isfinite(plug) & np.isfinite(oracle)
    if ok.sum() < 0.9 * len(rows):
        raise SystemExit(f'fig4 panel A: only {ok.sum()} of {len(rows)} units plottable')
    r1 = float(np.corrcoef(plug[ok], freqB[ok])[0, 1])
    r2 = float(np.corrcoef(oracle[ok], freqB[ok])[0, 1])
    plt.rcParams.update({'font.size': 6.5, 'axes.edgecolor': MUTED, 'xtick.color': MUTED,
                         'ytick.color': MUTED, 'text.color': INK})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(3.9, 2.35), constrained_layout=True)
    a1.scatter(freqB[ok], plug[ok], s=3, color=C_A, alpha=.6,
               label=f'plug-in Binomial, r = {r1:.3f}')
    a1.scatter(freqB[ok], oracle[ok], s=3, color=C_B, alpha=.6,
               label=f'flat-prior Beta-Binomial, r = {r2:.3f}')
    drawn = sum(int(c.get_offsets().shape[0]) for c in a1.collections)
    if drawn < 2 * ok.sum():
        raise SystemExit(f'fig4 panel A drew {drawn} points, expected {2 * ok.sum()}')
    print(f'   fig4 panel A: {ok.sum()} units, {drawn} points, '
          f'r(plug-in) = {r1:.3f}, r(Beta-Binomial) = {r2:.3f}')
    a1.plot([0, 1], [0, 1], color=C_OBS, lw=0.9)
    a1.set_xlabel('simulated frequency'); a1.set_ylabel('analytic probability')
    a1.set_xlim(-0.04, 1.04); a1.set_ylim(-0.03, 1.46)
    a1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    a1.axhline(1.06, color=MUTED, lw=0.4, alpha=0.45)
    legA = a1.legend(fontsize=4.9, frameon=False, loc='upper left',
                     bbox_to_anchor=(0.0, 1.0), borderaxespad=0.25,
                     handletextpad=0.5, labelspacing=0.45)
    a1.set_title('A. operational event', fontsize=6.5, loc='left', pad=3)
    sys.path.insert(0, os.path.join(GP, 'scripts'))
    import pro_common as pc
    xs = np.linspace(0, 0.95, 40)
    for (n_, m_), ls in zip([(60, 30), (60, 100), (60, 300), (20, 100), (200, 100)],
                            ['-', '-', '-', ':', '--']):
        ys = []
        for x in xs:
            h = int(round(n_ * (1 + x) / 2)); l = n_ - h
            ys.append(pc.bb_change_prob(h, l, m_))
        a2.plot(xs, ys, ls, lw=1.0, label=f'n={n_}, m={m_}',
                color=C_A if n_ == 60 else C_B)
    a2.set_xlabel('raw pre-cutoff margin $|r_t|$')
    a2.set_ylabel(r'$\pi^{\mathrm{raw}}(m)$')
    a2.set_ylim(-0.03, 1.46)
    a2.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    a2.axhline(1.06, color=MUTED, lw=0.4, alpha=0.45)
    legB = a2.legend(fontsize=4.6, frameon=False, loc='upper left',
                     bbox_to_anchor=(0.0, 1.0), borderaxespad=0.25, ncol=2,
                     columnspacing=1.0, handletextpad=0.5, labelspacing=0.4)
    a2.set_title('B. raw-majority probability, Eq. (2)', fontsize=6.5, loc='left', pad=3)
    for ax in (a1, a2):
        ax.tick_params(labelsize=5)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
    c = overlap_report(fig, 'fig4')
    c += legend_clear(a1, legA, 'fig4 panel A')
    c += legend_clear(a2, legB, 'fig4 panel B')
    if c:
        raise SystemExit(f'fig4: {c} overlapping elements remain')
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(N.FIGURES, f'fig4_analytic_vs_simulation.{ext}'),
                    dpi=220, bbox_inches='tight')
    plt.close(fig); return c


def fig5():
    """Calendar cohort: integer year ticks, legend clear of the per-origin labels."""
    J = json.load(open(os.path.join(GP, 'results', 'issue5', 'origin_cohort.json')))
    census = J['by_origin']          # per-origin AUROCs live here, not in the census block
    plt.rcParams.update({'font.size': 7, 'axes.edgecolor': MUTED, 'xtick.color': MUTED,
                         'ytick.color': MUTED, 'text.color': INK})
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 2.9), constrained_layout=True)
    series = [('bb_deployable_auroc_cond', 'stationary', C_A, 'o'),
              ('margin_auroc_cond', 'margin rule', C_OBS, 's'),
              ('two_var_auroc_cond', 'two-variable', C_B, '^'),
              ('compact7_auroc_cond', 'Compact-7', MUTED, 'd')]
    for ax, H in zip(axes, (3, 5, 10)):
        rows = [r for r in census if int(r.get('horizon', 0)) == H]
        rows.sort(key=lambda r: int(r['origin']))
        if not rows:
            continue
        xs = [int(r['origin']) for r in rows]
        for key, lab, col, mk in series:
            ys = [r.get(key) for r in rows]
            if not any(v is not None for v in ys):
                continue
            ax.plot(xs, ys, marker=mk, ms=3.5, lw=1.2, color=col,
                    label=lab if H == 3 else None)
        ax.set_xticks(xs)                                   # integer year ticks, the real origins
        ax.set_xticklabels([str(x) for x in xs], fontsize=6)
        ax.set_title(f'H = {H} years', fontsize=8)
        ax.set_xlabel('origin year', fontsize=7.5)
        ax.tick_params(labelsize=6.5)
        ax.set_ylim(0.55, 1.03)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        for r, x in zip(rows, xs):                          # events/units labels below the curves
            ax.annotate(f"{r.get('events_among_accrued','')}/{r.get('accrued','')}", xy=(x, 0.575),
                        ha='center', va='bottom', fontsize=5.2, color=MUTED)
    axes[0].set_ylabel('AUROC among accrued units', fontsize=7.5)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='outside upper center', ncol=4, frameon=False, fontsize=7.5)
    drawn = sum(len(ax.lines) for ax in axes)
    if drawn == 0:
        raise SystemExit('fig5 drew no data; refusing to emit an empty figure')
    print(f'   fig5: {drawn} plotted series across three panels')
    c = overlap_report(fig, 'fig5')
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(N.FIGURES, f'fig5_calendar_cohort.{ext}'),
                    dpi=220, bbox_inches='tight')
    plt.close(fig); return c


def fig6():
    """Null A and Null B under the primary annotation, legend clear of the observed labels."""
    draws = {k: np.load(os.path.join(N.RESULTS, f'E2_draws_primary_{k}.npz')) for k in ('A', 'B')}
    obs = json.load(open(os.path.join(N.RESULTS, 'E2_null_primary.json')))['observed']
    panels = [('two_var', 'two-variable AUROC', obs['two_var_auroc'], '{:.3f}'),
              ('margin', 'margin-rule AUROC', obs['margin_auroc'], '{:.3f}'),
              ('events', 'number of events', obs['events'], '{:.0f}')]
    plt.rcParams.update({'font.size': 8, 'axes.edgecolor': MUTED, 'xtick.color': MUTED,
                         'ytick.color': MUTED, 'text.color': INK})
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), constrained_layout=True)
    for ax, (key, lab, o, fmt) in zip(axes, panels):
        vA = draws['A'][key]; vB = draws['B'][key]
        vA = vA[np.isfinite(vA)]; vB = vB[np.isfinite(vB)]
        lo = min(vA.min(), vB.min(), o); hi = max(vA.max(), vB.max(), o)
        pad = (hi - lo) * 0.06 or 0.01
        bins = np.linspace(lo - pad, hi + pad, 30)
        nA, _, _ = ax.hist(vA, bins=bins, color=C_A, alpha=.75, edgecolor='white', linewidth=.3)
        nB, _, _ = ax.hist(vB, bins=bins, color=C_B, alpha=.55, edgecolor='white', linewidth=.3,
                           hatch='///')
        top = max(nA.max(), nB.max())
        ax.set_ylim(0, top * 1.34)                          # reserved headroom for the label
        ax.axvline(o, color=C_OBS, lw=1.7)
        x0, x1 = ax.get_xlim(); right = o > (x0 + x1) / 2
        ax.annotate(f'observed {fmt.format(o)}', xy=(o, top * 1.12),
                    xytext=(-5 if right else 5, 0), textcoords='offset points',
                    ha='right' if right else 'left', va='center', fontsize=7.5, color=C_OBS)
        ax.set_xlabel(lab, fontsize=8.5); ax.tick_params(labelsize=7)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel('draws', fontsize=8.5)
    handles = [Patch(facecolor=C_A, alpha=.75, edgecolor='white', label='Null A: within-claim permutation'),
               Patch(facecolor=C_B, alpha=.55, edgecolor='white', hatch='///', label='Null B: i.i.d. label draws'),
               Line2D([0], [0], color=C_OBS, lw=1.7, label='observed')]
    fig.legend(handles=handles, loc='outside upper center', ncol=3, frameon=False, fontsize=7.5)
    c = overlap_report(fig, 'fig6')
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(N.FIGURES, f'fig6_null_audit.{ext}'), dpi=220, bbox_inches='tight')
    plt.close(fig); return c


if __name__ == '__main__':
    tot = 0
    for fn in (fig4, fig5, fig6):
        try:
            tot += fn()
        except Exception as e:
            print(f'   {fn.__name__} FAILED: {type(e).__name__}: {e}')
    print(f'\ntotal collisions across regenerated figures: {tot}')
