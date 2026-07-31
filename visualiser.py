"""
Solvent recovery superstructure — ensemble inference interface.

Run with:  streamlit run app.py

A single view onto the deep-ensemble predictor for solvent recovery
flowsheets (superstructure formulation after Chea et al., 2020). For a
user-defined feed stream and a chosen path through the superstructure, the
app shows the ensemble's predictive distributions for feasibility,
recovery, purity, and cost, decomposes the uncertainty into its aleatoric
and epistemic parts, and — where ground truth is available — reports how
each prediction fared.

Intended audience: readers with a chemistry background but no ML
background. Every statistical readout carries a plain-language caption,
and the "Reading the numbers" panel at the bottom explains the concepts
once, in full.

Model interface
---------------
``evaluate.manual_eval(name, stream, model_type="ensemble", **indices)``
returns a dict with two keys:

``"predicted"``
    A ``ModelOutputDist`` with one attribute per output head
    (``feasibility``, ``recovery``, ``purity``, ``cost_per_kg``). Each
    attribute is a dict with three entries:

    - ``"dist"``: a ``torch.distributions`` object for the head —
      Bernoulli for feasibility, Normal (moment-matched from the ensemble
      mixture) for the continuous heads.
    - ``"epistemic"``: variance attributable to disagreement between
      ensemble members (reducible with more data).
    - ``"aleatoric"``: mean predicted noise variance (irreducible).

``"true"``
    Ground truth for the same stream, with bare-float / 0-d-tensor fields
    matching the head names. ``NaN``/``None`` means unlabelled.
"""

from __future__ import annotations

import math
import random

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from config import FLAGSHIP_MODEL_NAME

# =====================================================================
# Project imports (with a standalone fallback for the stream builder)
# =====================================================================

MODEL_AVAILABLE = True
IMPORT_ERROR = ""

try:
    from solvent_recovery_data_generator.solvents import (list_solvents, list_salts,
                                                          SOLVENT_DATA, SALT_DATA)
except Exception:
    # Fallback registry so the stream builder runs standalone.
    SOLVENT_DATA = {
        "methanol": ("67-56-1", 32.04, 792, 2.53, 1100.0, 64.7, -0.77),
        "ethanol": ("64-17-5", 46.07, 789, 2.44, 841.0, 78.4, -0.31),
        "1-propanol": ("71-23-8", 60.10, 803, 2.39, 686.0, 97.2, 0.25),
        "isopropanol": ("67-63-0", 60.10, 786, 2.32, 664.0, 82.5, 0.05),
        "1-butanol": ("71-36-3", 74.12, 810, 2.39, 582.0, 117.7, 0.88),
        "tert-butanol": ("75-65-0", 74.12, 786, 3.04, 527.0, 82.4, 0.35),
        "acetone": ("67-64-1", 58.08, 790, 2.16, 518.0, 56.1, -0.24),
        "2-butanone": ("78-93-3", 72.11, 805, 2.20, 443.0, 79.6, 0.29),
        "mibk": ("108-10-1", 100.16, 802, 2.09, 358.0, 116.5, 1.31),
        "ethyl acetate": ("141-78-6", 88.11, 902, 1.94, 366.0, 77.1, 0.73),
        "isopropyl acetate": ("108-21-4", 102.13, 872, 1.99, 331.0, 88.6, 1.02),
        "butyl acetate": ("123-86-4", 116.16, 882, 1.94, 309.0, 126.1, 1.78),
        "thf": ("109-99-9", 72.11, 889, 1.72, 410.0, 66.0, 0.46),
        "2-methyltetrahydrofuran": ("96-47-9", 86.13, 854, 1.78, 375.0, 80.2, 1.10),
        "1,4-dioxane": ("123-91-1", 88.11, 1033, 1.74, 406.0, 101.1, -0.27),
        "dichloromethane": ("75-09-2", 84.93, 1327, 1.19, 330.0, 39.6, 1.25),
        "chloroform": ("67-66-3", 119.38, 1489, 0.96, 247.0, 61.2, 1.97),
        "toluene": ("108-88-3", 92.14, 876, 1.71, 401.6, 110.6, 2.73),
        "benzene": ("71-43-2", 78.11, 876, 1.74, 394.0, 80.1, 2.13),
        "p-xylene": ("106-42-3", 106.17, 861, 1.72, 340.0, 138.4, 3.15),
        "n-hexane": ("110-54-3", 86.18, 655, 2.26, 335.0, 68.7, 3.76),
        "n-heptane": ("142-82-5", 100.20, 684, 2.24, 318.0, 98.4, 4.50),
        "cyclohexane": ("110-82-7", 84.16, 779, 1.84, 358.0, 80.7, 3.44),
        "n-pentane": ("109-66-0", 72.15, 626, 2.32, 358.0, 36.1, 3.39),
        "acetonitrile": ("75-05-8", 41.05, 786, 2.23, 727.0, 81.6, -0.34),
        "dmf": ("68-12-2", 73.09, 944, 2.06, 578.0, 153.0, -1.01),
        "dmso": ("67-68-5", 78.13, 1101, 1.96, 677.0, 189.0, -1.35),
        "dmac": ("127-19-5", 87.12, 940, 2.06, 500.0, 165.1, -0.77),
        "nmp": ("872-50-4", 99.13, 1028, 1.79, 493.0, 202.0, -0.38),
        "pyridine": ("110-86-1", 79.10, 982, 1.70, 449.0, 115.2, 0.65),
        "diethyl ether": ("60-29-7", 74.12, 713, 2.34, 358.0, 34.6, 0.89),
        "mtbe": ("1634-04-4", 88.15, 740, 2.16, 337.0, 55.2, 0.94),
        "1,2-dimethoxyethane": ("110-71-4", 90.12, 867, 1.42, 418.6, 85.0, -0.21),
        "ethylene glycol": ("107-21-1", 62.07, 1113, 2.36, 846.0, 197.3, -1.36),
        "glycerol": ("56-81-5", 92.09, 1261, 2.43, 976.0, 290.0, -1.76),
        "acetic acid": ("64-19-7", 60.05, 1049, 2.05, 390.0, 117.9, -0.17),
        "anisole": ("100-66-3", 108.14, 995, 1.72, 360.0, 153.7, 2.11),
        "chlorobenzene": ("108-90-7", 112.56, 1106, 1.34, 325.0, 131.7, 2.84),
        "nitromethane": ("75-52-5", 61.04, 1137, 1.74, 560.0, 101.2, -0.33),
        "water": ("7732-18-5", 18.02, 1000, 4.18, 2260.0, 100.0, -1.38),
    }
    SALT_DATA = {
        "sodium chloride": ("7647-14-5", 58.44, 2160),
        "sodium sulfate": ("7757-82-6", 142.04, 2664),
        "calcium chloride": ("10043-52-4", 110.98, 2150),
        "magnesium sulfate": ("7487-88-9", 120.37, 2660),
        "potassium chloride": ("7447-40-7", 74.55, 1984),
        "sodium carbonate": ("497-19-8", 105.99, 2540),
        "potassium carbonate": ("584-08-7", 138.21, 2430),
        "sodium bicarbonate": ("144-55-8", 84.01, 2200),
        "ammonium sulfate": ("7783-20-2", 132.14, 1769),
    }

    def list_solvents():
        return sorted(SOLVENT_DATA.keys())

    def list_salts():
        return sorted(SALT_DATA.keys())



from models import StreamComposition
from evaluate import manual_eval


# =====================================================================
# Superstructure vocabulary
# =====================================================================

# Human-readable names for each technology index, per stage, following the
# superstructure of Chea et al. (2020). Index 0 always means the stream
# bypasses the stage untouched.
STAGE_TECHNOLOGIES: dict[str, list[str]] = {
    "Solids removal": ["bypass", "sedimentation", "centrifugation",
                       "filtration"],
    "Recovery": ["bypass", "distillation", "pervaporation",
                 "aqueous two-phase extraction"],
    "Purification": ["bypass", "distillation", "pervaporation",
                     "ultrafiltration"],
    "Refinement": ["bypass", "distillation", "pervaporation",
                   "ultrafiltration", "microfiltration"],
}

STAGE_ORDER = list(STAGE_TECHNOLOGIES.keys())

# Keyword arguments expected by manual_eval, in stage order.
STAGE_KWARGS = ["solid_removal_idx", "recovery_idx",
                "purification_idx", "refinement_idx"]


# =====================================================================
# Page config and visual identity
# =====================================================================

st.set_page_config(
    page_title="Solvent recovery — superstructure inference",
    page_icon="◇",
    layout="wide",
)

# Palette: cool laboratory greys, with a single signal colour per epistemic
# state. Deliberately not a dashboard-blue; these are ink/graphite tones
# with a copper accent borrowed from distillation column cladding.
INK = "#1B1E23"
GRAPHITE = "#4A5058"
MIST = "#9AA3AD"
PAPER = "#FAFAF8"
RULE = "#DFE1E0"
COPPER = "#B0662F"
ALEATORIC = "#7C8B99"   # irreducible — muted, you cannot fix this
EPISTEMIC = "#B0662F"   # reducible — copper, this is where more data helps
FEASIBLE = "#3D6B52"
INFEASIBLE = "#8C3A3A"
TRUTH = "#1B4F72"       # measured reality — the only ink that isn't a guess

st.markdown(
    f"""
    <style>
      .stApp {{ background: {PAPER}; }}
      html, body, [class*="css"] {{
        font-family: "Inter", "Helvetica Neue", -apple-system, sans-serif;
      }}
      h1, h2, h3 {{ color: {INK}; letter-spacing: -0.015em; }}
      .eyebrow {{
        font-family: "SF Mono", "JetBrains Mono", monospace;
        font-size: 0.68rem; letter-spacing: 0.14em; text-transform: uppercase;
        color: {MIST}; margin-bottom: 0.2rem;
      }}
      .readout {{
        font-family: "SF Mono", "JetBrains Mono", monospace;
        font-size: 2.0rem; font-weight: 500; color: {INK};
        line-height: 1.1;
      }}
      .readout-unit {{ font-size: 0.9rem; color: {MIST}; font-weight: 400; }}
      .truth-line {{
        font-family: "SF Mono", "JetBrains Mono", monospace;
        font-size: 0.95rem; color: {TRUTH}; font-weight: 500;
        margin-top: 0.15rem;
      }}
      .subtle {{ color: {GRAPHITE}; font-size: 0.85rem; }}
      .mono-note {{
        font-family: "SF Mono", "JetBrains Mono", monospace;
        font-size: 0.72rem; color: {MIST};
      }}
      .flowpath {{
        font-family: "SF Mono", "JetBrains Mono", monospace;
        font-size: 0.85rem; color: {INK}; line-height: 2;
      }}
      .flowpath .bypassed {{ color: {MIST}; }}
      .flowpath .arrow {{ color: {MIST}; padding: 0 0.35rem; }}
      .flowpath .stage-tag {{
        font-size: 0.62rem; letter-spacing: 0.1em; text-transform: uppercase;
        color: {MIST}; display: block; line-height: 1;
      }}
      hr {{ border: none; border-top: 1px solid {RULE}; margin: 1.2rem 0; }}
      .stSlider label, .stSelectbox label {{
        font-size: 0.8rem !important; color: {GRAPHITE} !important;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)


# =====================================================================
# Small numeric helpers
# =====================================================================

def as_float(value) -> float | None:
    """Coerce a bare float / 0-d tensor to ``float``, or ``None``.

    Ground-truth fields may be floats, 0-d tensors, ``NaN``, or ``None``.
    ``None`` means unlabelled — a masked head legitimately carries it, and
    every call site needs that distinction preserved.
    """
    if value is None:
        return None
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:  # noqa: BLE001
            return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def gaussian_curve(mu: float, sigma: float, n: int = 400):
    """Sample the Normal(mu, sigma) density over ±4σ for plotting."""
    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, n)
    y = np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))
    return x, y


def normal_cdf(x: float, mu: float, sigma: float) -> float:
    """Φ((x − μ)/σ): the predicted probability of an outcome below ``x``."""
    return 0.5 * (1.0 + math.erf((x - mu) / (sigma * math.sqrt(2.0))))


def pit_reading(pit: float) -> tuple[str, str]:
    """Plain-language verdict for a single PIT value, plus a colour.

    One point cannot diagnose calibration — this only says whether this
    particular outcome landed somewhere the model considered plausible.
    """
    tail = min(pit, 1.0 - pit)
    if tail < 0.025:
        return "in the far tail — the model did not expect this", INFEASIBLE
    if tail < 0.10:
        return "toward the tail", COPPER
    return "well inside the predicted range", FEASIBLE


# =====================================================================
# Figures
# =====================================================================

def distribution_figure(mu: float, sigma: float, label: str, unit: str,
                        clip01: bool = False, truth: float | None = None):
    """Predictive density with the central 90% interval shaded.

    The shaded band is [μ − 1.6449σ, μ + 1.6449σ]: under the plotted
    Normal, exactly 90% of the probability lies inside it. Whether the true
    value actually lands there 90% of the time is a property of
    calibration, which the caption in the page footer addresses.
    """
    x, y = gaussian_curve(mu, sigma)

    # Widen the window if the truth falls outside ±4σ, so a badly missed
    # point stays visible instead of silently sliding off the axis.
    if truth is not None:
        lo = min(x[0], truth - 0.5 * sigma)
        hi = max(x[-1], truth + 0.5 * sigma)
        if lo < x[0] or hi > x[-1]:
            x = np.linspace(lo, hi, 500)
            y = (np.exp(-0.5 * ((x - mu) / sigma) ** 2)
                 / (sigma * math.sqrt(2 * math.pi)))

    if clip01:
        # Fractions live on [0, 1]; trim the tails but never the truth.
        mask = (x >= -0.02) & (x <= 1.02)
        if truth is not None:
            mask |= np.isclose(x, truth, atol=(x[1] - x[0]))
        x, y = x[mask], y[mask]

    fig = go.Figure()

    lo90, hi90 = mu - 1.6449 * sigma, mu + 1.6449 * sigma
    band = (x >= lo90) & (x <= hi90)
    fig.add_trace(go.Scatter(
        x=x[band], y=y[band], fill="tozeroy", mode="none",
        fillcolor="rgba(176,102,47,0.13)", hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines", line=dict(color=INK, width=1.8),
        hovertemplate=f"{label}: %{{x:.3f}} {unit}<extra></extra>",
        showlegend=False,
    ))
    fig.add_vline(x=mu, line=dict(color=COPPER, width=1.4, dash="dot"))

    if truth is not None:
        fig.add_vline(
            x=truth,
            line=dict(color=TRUTH, width=2.2),
            annotation_text="true",
            annotation_position="top",
            annotation_font=dict(size=9, color=TRUTH),
        )

    fig.update_layout(
        height=170,
        margin=dict(l=8, r=8, t=8, b=24),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            showgrid=False, zeroline=False, showline=True,
            linecolor=RULE, tickfont=dict(size=10, color=MIST),
            title=dict(text=unit, font=dict(size=9, color=MIST)),
        ),
        yaxis=dict(visible=False),
        hoverlabel=dict(bgcolor=PAPER, font_size=11),
    )
    return fig


def variance_split_bar(aleatoric: float, epistemic: float):
    """Horizontal stack: what you can't fix vs what more data would fix."""
    total = aleatoric + epistemic
    if total <= 0:
        return None
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[aleatoric], y=[""], orientation="h", marker_color=ALEATORIC,
        name="Aleatoric (noise in the process)",
        hovertemplate="Aleatoric: %{x:.4g}<extra></extra>", width=0.5,
    ))
    fig.add_trace(go.Bar(
        x=[epistemic], y=[""], orientation="h", marker_color=EPISTEMIC,
        name="Epistemic (model disagreement)",
        hovertemplate="Epistemic: %{x:.4g}<extra></extra>", width=0.5,
    ))
    fig.update_layout(
        barmode="stack", height=58,
        margin=dict(l=8, r=8, t=4, b=4),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, showticklabels=False),
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.6, x=0,
            font=dict(size=10, color=GRAPHITE), bgcolor="rgba(0,0,0,0)",
        ),
        bargap=0.1,
    )
    return fig


# =====================================================================
# Panels
# =====================================================================

def head_panel(col, title: str, unit: str, head: dict,
               truth: float | None = None, clip01: bool = False,
               fmt: str = "{:.3f}") -> None:
    """Render one continuous output head of the ensemble.

    ``head`` is the per-head dict from ``ModelOutputDist``:
    ``{"dist", "epistemic", "aleatoric"}``.

    Reading order is deliberate — the error comes first, because that is
    what you actually want to know; the variance split comes last, because
    it explains rather than reports.
    """
    dist = head["dist"]
    mu = float(dist.mean)
    sigma = float(dist.stddev)
    ale = float(head["aleatoric"])
    epi = float(head["epistemic"])

    with col:
        st.markdown(f'<div class="eyebrow">{title}</div>',
                    unsafe_allow_html=True)
        st.markdown(
            f'<div class="readout">{fmt.format(mu)}'
            f'<span class="readout-unit"> ± {sigma:.3f}</span></div>',
            unsafe_allow_html=True,
        )

        if truth is not None:
            err = mu - truth
            z = err / sigma if sigma > 0 else float("nan")
            st.markdown(
                f'<div class="truth-line">true {fmt.format(truth)}'
                f'<span class="readout-unit">&nbsp;&nbsp;err {err:+.3f}'
                f' &nbsp;·&nbsp; {z:+.2f}σ</span></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<div class="truth-line">&nbsp;</div>',
                        unsafe_allow_html=True)

        st.plotly_chart(
            distribution_figure(mu, sigma, title, unit,
                                clip01=clip01, truth=truth),
            use_container_width=True,
            config={"displayModeBar": False},
        )

        # PIT: where the true value sits inside the predicted distribution
        # (0 = far below everything the model expected, 1 = far above,
        # 0.5 = dead centre). See "Reading the numbers" for the full story.
        if truth is not None and sigma > 0:
            pit = normal_cdf(truth, mu, sigma)
            verdict, colour = pit_reading(pit)
            st.markdown(
                f'<div class="mono-note">PIT = {pit:.3f} &nbsp;·&nbsp; '
                f'<span style="color:{colour}">{verdict}</span></div>',
                unsafe_allow_html=True,
            )

        frac = epi / (ale + epi) if (ale + epi) > 0 else 0.0
        st.markdown(
            f'<div class="mono-note">σ² = {ale + epi:.4g} &nbsp;·&nbsp; '
            f'{frac:.0%} reducible</div>',
            unsafe_allow_html=True,
        )
        bar = variance_split_bar(ale, epi)
        if bar is not None:
            st.plotly_chart(bar, use_container_width=True,
                            config={"displayModeBar": False})


def feasibility_panel(head: dict, t_feas: float | None) -> None:
    """Render the feasibility gate: verdict, truth check, doubt split."""
    p = float(head["dist"].probs)
    ale = float(head["aleatoric"])
    epi = float(head["epistemic"])
    total = ale + epi

    gate_col, unc_col = st.columns([1, 2])

    with gate_col:
        colour = FEASIBLE if p >= 0.5 else INFEASIBLE
        verdict = "feasible" if p >= 0.5 else "infeasible"
        st.markdown('<div class="eyebrow">Feasibility</div>',
                    unsafe_allow_html=True)
        st.markdown(
            f'<div class="readout" style="color:{colour}">{p:.1%}'
            f'<span class="readout-unit"> {verdict}</span></div>',
            unsafe_allow_html=True,
        )
        if t_feas is not None:
            true_label = "feasible" if t_feas >= 0.5 else "infeasible"
            correct = (p >= 0.5) == (t_feas >= 0.5)
            # Brier score for one point: squared error on the probability.
            # Rewards being right *and* being appropriately confident.
            brier = (p - t_feas) ** 2
            mark = FEASIBLE if correct else INFEASIBLE
            st.markdown(
                f'<div class="truth-line">true {true_label}'
                f'<span class="readout-unit">&nbsp;&nbsp;'
                f'<span style="color:{mark}">'
                f'{"correct" if correct else "wrong"}</span>'
                f' &nbsp;·&nbsp; Brier {brier:.3f}</span></div>',
                unsafe_allow_html=True,
            )

    with unc_col:
        st.markdown('<div class="eyebrow">Where the doubt comes from</div>',
                    unsafe_allow_html=True)
        bar = variance_split_bar(ale, epi)
        if bar is not None:
            st.plotly_chart(bar, use_container_width=True,
                            config={"displayModeBar": False})
        st.markdown(
            f'<div class="mono-note">H = {total:.3f} nats &nbsp;·&nbsp; '
            f'{(epi / total if total > 0 else 0):.0%} from '
            f'ensemble disagreement</div>',
            unsafe_allow_html=True,
        )


# =====================================================================
# Sidebar: feed stream, operating point, superstructure
# =====================================================================

SOLVENTS = list_solvents()
SALTS = list_salts()

# One entry per widget: (session-state key, default). Declared up front so
# the randomiser and the widgets never drift apart.
_DEFAULT_TARGET = ("2-methyltetrahydrofuran"
                   if "2-methyltetrahydrofuran" in SOLVENTS else SOLVENTS[0])
_DEFAULT_SALT = ("sodium bicarbonate"
                 if "sodium bicarbonate" in SALTS else SALTS[0])

WIDGET_DEFAULTS: dict[str, object] = {
    "target_name": _DEFAULT_TARGET,
    "target_kgph": 34.0,
    "solvent2_name": "acetone" if "acetone" in SOLVENTS else SOLVENTS[0],
    "solvent2_kgph": 0.0,
    "salt_name": _DEFAULT_SALT,
    "salt_kgph": 0.0,
    "water_kgph": 0.0,
    "solids_kgph": 0.0,
    "temperature_C": 25.0,
    "stage_0": 0,
    "stage_1": 0,
    "stage_2": 0,
    "stage_3": 0,
}

for key, default in WIDGET_DEFAULTS.items():
    st.session_state.setdefault(key, default)


def randomise_inputs() -> None:
    """Draw a random feed stream, operating point, and flowsheet.

    Runs as an ``on_click`` callback, i.e. before the widgets are built on
    the rerun, so writing to ``st.session_state`` here is the supported way
    to move every control at once. Flows are rounded to the sliders' step
    sizes; the secondary solvent, salt, water, and solids are each absent
    half the time so that sparse streams appear as often as busy ones.
    """
    target = random.choice(SOLVENTS)
    others = [s for s in SOLVENTS if s != target] or SOLVENTS

    def maybe(lo: float, hi: float, step: float) -> float:
        if random.random() < 0.5:
            return 0.0
        return round(random.uniform(lo, hi) / step) * step

    st.session_state.update({
        "target_name": target,
        "target_kgph": round(random.uniform(5.0, 150.0) * 2) / 2,
        "solvent2_name": random.choice(others),
        "solvent2_kgph": maybe(0.5, 100.0, 0.5),
        "salt_name": random.choice(SALTS),
        "salt_kgph": maybe(0.1, 20.0, 0.1),
        "water_kgph": maybe(0.5, 150.0, 0.5),
        "solids_kgph": maybe(0.1, 20.0, 0.1),
        "temperature_C": float(random.randint(-10, 150)),
        **{f"stage_{i}": random.randrange(len(STAGE_TECHNOLOGIES[stage]))
           for i, stage in enumerate(STAGE_ORDER)},
    })


with st.sidebar:
    st.button("Randomise everything", on_click=randomise_inputs,
              use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">Feed stream</div>',
                unsafe_allow_html=True)

    st.selectbox("Target solvent", SOLVENTS, key="target_name")
    st.slider("Target flow", 0.0, 200.0, step=0.5, format="%.1f kg/h",
              key="target_kgph")

    st.selectbox("Second solvent", SOLVENTS, key="solvent2_name")
    st.slider("Second solvent flow", 0.0, 200.0, step=0.5,
              format="%.1f kg/h", key="solvent2_kgph")

    st.selectbox("Salt", SALTS, key="salt_name")
    st.slider("Salt flow", 0.0, 50.0, step=0.1, format="%.1f kg/h",
              key="salt_kgph")

    st.slider("Water", 0.0, 200.0, step=0.5, format="%.1f kg/h",
              key="water_kgph")
    st.slider("Solids", 0.0, 50.0, step=0.1, format="%.1f kg/h",
              key="solids_kgph")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">Operating point</div>',
                unsafe_allow_html=True)
    st.slider("Temperature", -20.0, 200.0, step=1.0, format="%.0f °C",
              key="temperature_C")

    st.markdown('<div class="eyebrow">Superstructure</div>',
                unsafe_allow_html=True)
    for i, stage in enumerate(STAGE_ORDER):
        st.selectbox(
            stage,
            options=range(len(STAGE_TECHNOLOGIES[stage])),
            format_func=lambda idx, s=stage: (
                f"{idx} · {STAGE_TECHNOLOGIES[s][idx]}"),
            key=f"stage_{i}",
        )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">Checkpoint</div>',
                unsafe_allow_html=True)
    ensemble_name = st.text_input("Ensemble", value=FLAGSHIP_MODEL_NAME)


# Pull widget state into plain names for the rest of the script.
target_name = st.session_state["target_name"]
target_kgph = st.session_state["target_kgph"]
solvent2_name = st.session_state["solvent2_name"]
solvent2_kgph = st.session_state["solvent2_kgph"]
salt_name = st.session_state["salt_name"]
salt_kgph = st.session_state["salt_kgph"]
water_kgph = st.session_state["water_kgph"]
solids_kgph = st.session_state["solids_kgph"]
temperature_C = st.session_state["temperature_C"]
stage_indices = [st.session_state[f"stage_{i}"]
                 for i in range(len(STAGE_ORDER))]


# =====================================================================
# Header: composition and flowsheet path
# =====================================================================

st.markdown('<div class="eyebrow">Superstructure optimisation · '
            'solvent recovery</div>', unsafe_allow_html=True)
st.markdown(f"### {target_name}")

total_flow = (target_kgph + solvent2_kgph + water_kgph
              + salt_kgph + solids_kgph)

if total_flow <= 0:
    st.warning("No feed. Raise at least one flow above zero.")
    st.stop()

components = [
    (target_name, target_kgph, INK),
    (solvent2_name, solvent2_kgph, GRAPHITE),
    ("water", water_kgph, "#6E8BA3"),
    (salt_name, salt_kgph, MIST),
    ("solids", solids_kgph, "#C4B7A6"),
]

comp_fig = go.Figure()
for name, flow, colour in components:
    if flow <= 0:
        continue
    comp_fig.add_trace(go.Bar(
        x=[flow / total_flow], y=[""], orientation="h",
        marker_color=colour, name=name,
        hovertemplate=f"{name}: {flow:.1f} kg/h "
                      f"({flow / total_flow:.1%})<extra></extra>",
        width=0.42,
    ))
comp_fig.update_layout(
    barmode="stack", height=76,
    margin=dict(l=0, r=0, t=0, b=0),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
               range=[0, 1]),
    yaxis=dict(showgrid=False, showticklabels=False),
    showlegend=True,
    legend=dict(orientation="h", yanchor="top", y=-0.1, x=0,
                font=dict(size=10, color=GRAPHITE),
                bgcolor="rgba(0,0,0,0)"),
    bargap=0.05,
)
st.plotly_chart(comp_fig, use_container_width=True,
                config={"displayModeBar": False})

# The chosen path through the superstructure, written out as the flowsheet
# it actually describes. Bypassed stages stay visible but greyed, so the
# reader sees both what the stream goes through and what it skips.
path_cells = []
for stage, idx in zip(STAGE_ORDER, stage_indices):
    tech = STAGE_TECHNOLOGIES[stage][idx]
    css = "bypassed" if idx == 0 else ""
    path_cells.append(
        f'<span class="{css}"><span class="stage-tag">{stage}</span>'
        f'{tech}</span>'
    )
arrow = '<span class="arrow">→</span>'
st.markdown(
    f'<div class="flowpath">'
    + f' {arrow} '.join(path_cells)
    + f'</div>',
    unsafe_allow_html=True,
)

st.markdown(
    f'<div class="mono-note">{total_flow:.1f} kg/h total &nbsp;·&nbsp; '
    f'{temperature_C:.0f} °C &nbsp;·&nbsp; '
    f'superstructure {stage_indices}</div>',
    unsafe_allow_html=True,
)

st.markdown("<hr>", unsafe_allow_html=True)


# =====================================================================
# Inference
# =====================================================================

if not MODEL_AVAILABLE:
    st.info(
        f"Stream builder only — the model modules did not import.\n\n"
        f"`{IMPORT_ERROR}`\n\n"
        f"Run this from the project root so `models` and `evaluate` are "
        f"importable, and the prediction panels will appear here."
    )
    st.stop()

if not st.button("Run inference", type="primary"):
    st.markdown('<div class="subtle">Set the feed and operating point, '
                'then run inference.</div>', unsafe_allow_html=True)
    st.stop()

stream = StreamComposition(
    target_name=target_name,
    target_kgph=target_kgph,
    solvent2_name=solvent2_name,
    solvent2_kgph=solvent2_kgph,
    salt_name=salt_name,
    salt_kgph=salt_kgph,
    water_kgph=water_kgph,
    solids_kgph=solids_kgph,
)

stage_kwargs = dict(zip(STAGE_KWARGS, stage_indices))

try:
    result = manual_eval(ensemble_name, stream,
                         temperature_C=temperature_C, **stage_kwargs)
except Exception as exc:  # noqa: BLE001
    st.error(f"Ensemble evaluation failed.\n\n"
             f"`{type(exc).__name__}: {exc}`")
    st.stop()

out = result["predicted"]
truth = result["true"]

t_feas = as_float(getattr(truth, "feasibility", None))
t_recovery = as_float(getattr(truth, "recovery", None))
t_purity = as_float(getattr(truth, "purity", None))
t_cost = as_float(getattr(truth, "cost_per_kg", None))

infeasible_in_truth = t_feas is not None and t_feas < 0.5


def shown(value: float | None) -> float | None:
    """Truth markers vanish on masked heads rather than inventing an error.

    When the stream is infeasible in the data, the continuous targets were
    masked during training, so comparing predictions against them would
    manufacture errors that mean nothing.
    """
    return None if infeasible_in_truth else value


# (title, unit, attribute on ModelOutputDist, truth, clip to [0,1], format)
HEADS = [
    ("Recovery", "fraction", "recovery", t_recovery, True, "{:.3f}"),
    ("Purity", "fraction", "purity", t_purity, True, "{:.3f}"),
    ("Cost", "USD/kg", "cost_per_kg", t_cost, False, "{:.2f}"),
]

feasibility_panel(out.feasibility, t_feas)

st.markdown("<hr>", unsafe_allow_html=True)

p_feas = float(out.feasibility["dist"].probs)
if infeasible_in_truth:
    st.markdown(
        '<div class="subtle">This stream is infeasible in the data, so '
        'the continuous heads were masked during training. Their outputs '
        'below are undefined, not wrong — comparing them to a ground '
        'truth is not meaningful here.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)
elif p_feas < 0.5:
    st.markdown(
        '<div class="subtle">The model does not expect this separation '
        'to work. Recovery, purity, and cost are shown below but were '
        'trained on feasible streams — read them as extrapolation, not '
        'prediction.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

cols = st.columns(3)
for col, (title, unit, attr, t, clip, fmt) in zip(cols, HEADS):
    head_panel(col, title, unit, getattr(out, attr),
               truth=shown(t), clip01=clip, fmt=fmt)

st.markdown("<hr>", unsafe_allow_html=True)


# =====================================================================
# Footnotes: what the statistics mean, in plain language
# =====================================================================

st.markdown(
    '<div class="mono-note">The shaded band on each curve is the central '
    '90% interval of the plotted Normal distribution — the range '
    'μ ± 1.6449σ, inside which the model places 90% of its probability. '
    'That Normal is itself a summary: the ensemble\'s true prediction is a '
    'mixture of the members\' individual Gaussians, matched here by its '
    'mean and total variance (aleatoric + epistemic). The band is an '
    'honest 90% only to the extent the model is calibrated.</div>',
    unsafe_allow_html=True,
)

with st.expander("Reading the numbers"):
    st.markdown(
        """
**The curve** is the model's belief about the quantity, not a measurement.
Its centre (dotted copper line) is the best guess; its width is how unsure
the model is. The shaded region is the range μ&nbsp;±&nbsp;1.6449σ, which
contains 90% of the plotted distribution's probability.

**Aleatoric vs epistemic.** The total uncertainty splits into two parts.
*Aleatoric* uncertainty is scatter the process itself produces — run the
same separation twice and you will not get identical numbers. No amount of
extra data removes it. *Epistemic* uncertainty is disagreement among the
ensemble's independently trained networks: where they diverge, the model
knows it hasn't seen enough streams like this one. More
training data shrinks it. The "reducible" percentage tells you which kind
dominates — high epistemic share marks exactly the streams worth
labelling next (this is what drives the active-learning loop).

**PIT** (probability integral transform) asks: given the distribution the
model predicted, where did reality land? It is the predicted probability
of seeing a value below the true one — 0.5 means the truth hit dead
centre, values near 0 or 1 mean it fell in a tail the model considered
unlikely. One PIT value is a spot check, nothing more. Calibration — the
claim that the model's 90% bands really do catch the truth 90% of the
time — can only be judged from the *distribution* of PIT values over a
whole test set, which should be uniform for a calibrated model.

**Brier score** is the squared error on the feasibility probability:
predicting 95% feasible on an infeasible stream is punished much harder
than predicting 55%. Lower is better; 0 is perfect.
        """.strip()
    )