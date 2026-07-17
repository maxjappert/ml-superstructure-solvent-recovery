"""
Solvent recovery superstructure — inference interface.

Run with:  streamlit run app.py

Two tabs: the ensemble (distributions, uncertainty decomposition) and the
single model (point estimates). Both are evaluated against the same ground
truth so the two approaches can be compared on identical streams.
"""

from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------- project imports

MODEL_AVAILABLE = True
IMPORT_ERROR = ""

try:
    from solvent_recovery.solvents import (list_solvents, list_salts,
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


try:
    from models import StreamComposition
    from evaluate import manual_eval
except Exception as exc:  # noqa: BLE001
    MODEL_AVAILABLE = False
    IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------- page config

st.set_page_config(
    page_title="Solvent recovery — superstructure inference",
    page_icon="◇",
    layout="wide",
)

# Palette: cool laboratory greys, with a single signal colour per epistemic
# state. Deliberately not a dashboard-blue; these are ink/graphite tones with
# a copper accent borrowed from distillation column cladding.
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
SINGLE = "#6B5B7B"      # the point model — one voice, no chorus

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
      hr {{ border: none; border-top: 1px solid {RULE}; margin: 1.2rem 0; }}
      .stSlider label, .stSelectbox label {{
        font-size: 0.8rem !important; color: {GRAPHITE} !important;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- helpers

def bernoulli_entropy_np(p: float) -> float:
    """Entropy in nats. Clipped to avoid log(0)."""
    p = float(np.clip(p, 1e-12, 1 - 1e-12))
    return -(p * math.log(p) + (1 - p) * math.log(1 - p))


def gaussian_curve(mu: float, sigma: float, n: int = 400):
    lo, hi = mu - 4 * sigma, mu + 4 * sigma
    x = np.linspace(lo, hi, n)
    y = np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))
    return x, y


def as_float(v):
    """Ground-truth and single-model fields are bare floats or 0-d tensors.

    None means unlabelled — a masked head legitimately carries it, and every
    call site needs that distinction.
    """
    if v is None:
        return None
    if hasattr(v, "item"):
        try:
            v = v.item()
        except Exception:  # noqa: BLE001
            return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def normal_cdf(x: float, mu: float, sigma: float) -> float:
    """PIT value: where the truth sits in the predicted distribution."""
    return 0.5 * (1.0 + math.erf((x - mu) / (sigma * math.sqrt(2.0))))


def pit_reading(pit: float) -> tuple[str, str]:
    """Plain-language verdict for a single PIT value, plus a colour.

    One point cannot diagnose calibration — this only says whether this
    particular outcome was somewhere the model expected.
    """
    tail = min(pit, 1.0 - pit)
    if tail < 0.025:
        return "in the far tail — the model did not expect this", INFEASIBLE
    if tail < 0.10:
        return "toward the tail", COPPER
    return "well inside the predicted range", FEASIBLE


def distribution_figure(mu, sigma, label, unit, clip01=False, truth=None):
    """Density curve with the 90% band marked, and the truth if we have it.

    The band is the honest interval only if the model is calibrated — the
    caption says so rather than letting the picture imply more than it knows.
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


def number_line_figure(value, truth, label, unit, clip01=False):
    """A point estimate has no width. This draws exactly that.

    Same height as the density plot so the two tabs line up when flipped
    between — the empty space is the argument.
    """
    marks = [m for m in (value, truth) if m is not None]
    if not marks:
        return None

    span = max(marks) - min(marks)
    pad = max(span * 0.6, abs(float(np.mean(marks))) * 0.15, 0.05)
    lo, hi = min(marks) - pad, max(marks) + pad
    if clip01:
        lo, hi = max(lo, -0.02), min(hi, 1.02)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[0, 0], mode="lines",
        line=dict(color=RULE, width=1.2), hoverinfo="skip", showlegend=False,
    ))
    if value is not None and truth is not None:
        fig.add_trace(go.Scatter(
            x=[value, truth], y=[0, 0], mode="lines",
            line=dict(color=MIST, width=1.4, dash="dot"),
            hoverinfo="skip", showlegend=False,
        ))
    if value is not None:
        fig.add_trace(go.Scatter(
            x=[value], y=[0], mode="markers",
            marker=dict(color=SINGLE, size=16, symbol="line-ns",
                        line=dict(color=SINGLE, width=2.5)),
            hovertemplate=f"predicted: %{{x:.3f}} {unit}<extra></extra>",
            showlegend=False,
        ))
    if truth is not None:
        fig.add_trace(go.Scatter(
            x=[truth], y=[0], mode="markers",
            marker=dict(color=TRUTH, size=16, symbol="line-ns",
                        line=dict(color=TRUTH, width=2.5)),
            hovertemplate=f"true: %{{x:.3f}} {unit}<extra></extra>",
            showlegend=False,
        ))

    fig.update_layout(
        height=170,
        margin=dict(l=8, r=8, t=8, b=24),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            range=[lo, hi], showgrid=False, zeroline=False, showline=True,
            linecolor=RULE, tickfont=dict(size=10, color=MIST),
            title=dict(text=unit, font=dict(size=9, color=MIST)),
        ),
        yaxis=dict(visible=False, range=[-1, 1]),
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
        name="Aleatoric", hovertemplate="Aleatoric: %{x:.4g}<extra></extra>",
        width=0.5,
    ))
    fig.add_trace(go.Bar(
        x=[epistemic], y=[""], orientation="h", marker_color=EPISTEMIC,
        name="Epistemic", hovertemplate="Epistemic: %{x:.4g}<extra></extra>",
        width=0.5,
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


def head_panel(col, title, unit, res, truth=None, clip01=False, fmt="{:.3f}"):
    """One continuous head from the ensemble: distribution against truth.

    Reading order is deliberate — the error comes first, because that is
    what you actually want to know, and the variance split comes last,
    because it explains rather than reports.
    """
    mu = float(res["dist"].mean)
    sigma = float(res["dist"].stddev)
    ale = float(res["aleatoric"])
    epi = float(res["epistemic"])

    with col:
        st.markdown(f'<div class="eyebrow">{title}</div>', unsafe_allow_html=True)
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


def point_panel(col, title, unit, value, truth=None, clip01=False,
                fmt="{:.3f}"):
    """One continuous head from the single model: a number, and its error.

    There is no PIT here and no variance split, because there is no
    distribution. That absence is the finding, not an omission.
    """
    with col:
        st.markdown(f'<div class="eyebrow">{title}</div>',
                    unsafe_allow_html=True)

        if value is None:
            st.markdown(f'<div class="readout" style="color:{MIST}">—</div>',
                        unsafe_allow_html=True)
            st.markdown('<div class="mono-note">no output</div>',
                        unsafe_allow_html=True)
            return

        st.markdown(
            f'<div class="readout" style="color:{SINGLE}">'
            f'{fmt.format(value)}</div>',
            unsafe_allow_html=True,
        )

        if truth is not None:
            err = value - truth
            st.markdown(
                f'<div class="truth-line">true {fmt.format(truth)}'
                f'<span class="readout-unit">&nbsp;&nbsp;err {err:+.3f}'
                f'</span></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<div class="truth-line">&nbsp;</div>',
                        unsafe_allow_html=True)

        fig = number_line_figure(value, truth, title, unit, clip01)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True,
                            config={"displayModeBar": False})

        st.markdown(
            '<div class="mono-note">no interval — this model reports a '
            'number, not a belief</div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------- sidebar: inputs

with st.sidebar:
    st.markdown('<div class="eyebrow">Feed stream</div>', unsafe_allow_html=True)

    solvents = list_solvents()
    salts = list_salts()

    default_target = ("2-methyltetrahydrofuran" if "2-methyltetrahydrofuran"
                      in solvents else solvents[0])
    target_name = st.selectbox(
        "Target solvent", solvents, index=solvents.index(default_target),
    )
    target_kgph = st.slider("Target flow", 0.0, 200.0, 34.0, 0.5,
                            format="%.1f kg/h")

    st.markdown("<hr>", unsafe_allow_html=True)

    use_solvent2 = st.checkbox("Second solvent present", value=False)
    default_s2 = "acetone" if "acetone" in solvents else solvents[0]
    solvent2_name = st.selectbox(
        "Second solvent", solvents, index=solvents.index(default_s2),
        disabled=not use_solvent2,
    )
    solvent2_kgph = st.slider(
        "Second solvent flow", 0.0, 200.0, 0.0, 0.5,
        format="%.1f kg/h", disabled=not use_solvent2,
    )
    if not use_solvent2:
        solvent2_kgph = 0.0

    st.markdown("<hr>", unsafe_allow_html=True)

    use_salt = st.checkbox("Salt present", value=False)
    default_salt = ("sodium bicarbonate" if "sodium bicarbonate" in salts
                    else salts[0])
    salt_name = st.selectbox(
        "Salt", salts, index=salts.index(default_salt), disabled=not use_salt,
    )
    salt_kgph = st.slider("Salt flow", 0.0, 50.0, 0.0, 0.1,
                          format="%.1f kg/h", disabled=not use_salt)
    if not use_salt:
        salt_kgph = 0.0

    st.markdown("<hr>", unsafe_allow_html=True)

    water_kgph = st.slider("Water", 0.0, 200.0, 0.0, 0.5, format="%.1f kg/h")
    solids_kgph = st.slider("Solids", 0.0, 50.0, 0.0, 0.1, format="%.1f kg/h")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">Operating point</div>',
                unsafe_allow_html=True)
    temperature_C = st.slider("Temperature", -20.0, 200.0, 25.0, 1.0,
                              format="%.0f °C")

    st.markdown('<div class="eyebrow">Superstructure</div>',
                unsafe_allow_html=True)
    ss = [
        st.select_slider(f"Stage {i + 1}", options=[0, 1, 2, 3], value=0,
                         key=f"ss{i}")
        for i in range(4)
    ]

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">Checkpoints</div>',
                unsafe_allow_html=True)
    ensemble_name = st.text_input("Ensemble", value="ensemble_best_150726.pt")
    single_name = st.text_input("Single model", value="single_20260717_090153.pt")


# ---------------------------------------------------------------- header

st.markdown('<div class="eyebrow">Superstructure optimisation · '
            'solvent recovery</div>', unsafe_allow_html=True)
st.markdown(f"### {target_name}")

total_flow = target_kgph + solvent2_kgph + water_kgph + salt_kgph + solids_kgph

if total_flow <= 0:
    st.warning("No feed. Raise at least one flow above zero.")
    st.stop()

components = [
    (target_name, target_kgph, INK),
    (solvent2_name if use_solvent2 else "—", solvent2_kgph, GRAPHITE),
    ("water", water_kgph, "#6E8BA3"),
    (salt_name if use_salt else "—", salt_kgph, MIST),
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

st.markdown(
    f'<div class="mono-note">{total_flow:.1f} kg/h total &nbsp;·&nbsp; '
    f'{temperature_C:.0f} °C &nbsp;·&nbsp; superstructure {ss}</div>',
    unsafe_allow_html=True,
)

st.markdown("<hr>", unsafe_allow_html=True)


# ---------------------------------------------------------------- inference

if not MODEL_AVAILABLE:
    st.info(
        f"Stream builder only — the model modules did not import.\n\n"
        f"`{IMPORT_ERROR}`\n\n"
        f"Run this from the project root so `models` and `evaluate` are "
        f"importable, and the prediction panels will appear here."
    )
    st.stop()

run = st.button("Run inference", type="primary")

if not run:
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

idx = dict(
    solid_removal_idx=ss[0],
    recovery_idx=ss[1],
    purification_idx=ss[2],
    refinement_idx=ss[3],
    temperature_C=temperature_C,
)

ens_result = None
ens_error = ""
try:
    ens_result = manual_eval(ensemble_name, stream, model_type="ensemble", **idx)
except Exception as exc:  # noqa: BLE001
    ens_error = f"{type(exc).__name__}: {exc}"

single_result = None
single_error = ""
try:
    single_result = manual_eval(single_name, stream, model_type="single", **idx)
except Exception as exc:  # noqa: BLE001
    single_error = f"{type(exc).__name__}: {exc}"

if ens_result is None and single_result is None:
    st.error(
        f"Both models failed.\n\n"
        f"Ensemble: `{ens_error}`\n\n"
        f"Single: `{single_error}`"
    )
    st.stop()

# Ground truth is identical from either call — compute() does not depend on
# the model. Take whichever succeeded.
truth = (ens_result or single_result)["true"]

t_feas = as_float(getattr(truth, "feasibility", None))
t_recovery = as_float(getattr(truth, "recovery", None))
t_purity = as_float(getattr(truth, "purity", None))
t_cost = as_float(getattr(truth, "cost_per_kg", None))

infeasible_in_truth = t_feas is not None and t_feas < 0.5


def shown(v):
    """Truth markers vanish on masked heads rather than inventing an error."""
    return None if infeasible_in_truth else v


HEADS = [
    ("Recovery", "fraction", "recovery", t_recovery, True, "{:.3f}"),
    ("Purity", "fraction", "purity", t_purity, True, "{:.3f}"),
    ("Cost", "USD/kg", "cost_per_kg", t_cost, False, "{:.2f}"),
]


def gate_note(pred_infeasible: bool):
    """Same caveat in both tabs — the mask is a property of the data."""
    if infeasible_in_truth:
        st.markdown(
            '<div class="subtle">This stream is infeasible in the data, so '
            'the continuous heads were masked during training. Their outputs '
            'below are undefined, not wrong — comparing them to a ground '
            'truth is not meaningful here.</div>',
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)
    elif pred_infeasible:
        st.markdown(
            '<div class="subtle">The model does not expect this separation '
            'to work. Recovery, purity, and cost are shown below but were '
            'trained on feasible streams — read them as extrapolation, not '
            'prediction.</div>',
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)


def feasibility_verdict(p, label):
    """Shared readout for both tabs. p is a probability in [0, 1]."""
    colour = FEASIBLE if p >= 0.5 else INFEASIBLE
    verdict = "feasible" if p >= 0.5 else "infeasible"
    st.markdown(f'<div class="eyebrow">Feasibility · {label}</div>',
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
        mark_colour = FEASIBLE if correct else INFEASIBLE
        st.markdown(
            f'<div class="truth-line">true {true_label}'
            f'<span class="readout-unit">&nbsp;&nbsp;'
            f'<span style="color:{mark_colour}">'
            f'{"correct" if correct else "wrong"}</span>'
            f' &nbsp;·&nbsp; Brier {brier:.3f}</span></div>',
            unsafe_allow_html=True,
        )


tab_ens, tab_single = st.tabs(["Ensemble", "Point estimate"])


# ---------------------------------------------------------------- ensemble tab

with tab_ens:
    if ens_result is None:
        st.error(f"Ensemble evaluation failed.\n\n`{ens_error}`")
    else:
        out = ens_result["predicted"]

        p_feas = float(out.feasibility["dist"].probs)
        f_epi = float(out.feasibility["epistemic"])
        f_ale = float(out.feasibility["aleatoric"])
        f_total = f_epi + f_ale

        gate_col, unc_col = st.columns([1, 2])

        with gate_col:
            feasibility_verdict(p_feas, "ensemble")

        with unc_col:
            st.markdown('<div class="eyebrow">Where the doubt comes from</div>',
                        unsafe_allow_html=True)
            bar = variance_split_bar(f_ale, f_epi)
            if bar is not None:
                st.plotly_chart(bar, use_container_width=True,
                                config={"displayModeBar": False})
            st.markdown(
                f'<div class="mono-note">H = {f_total:.3f} nats '
                f'&nbsp;·&nbsp; '
                f'{(f_epi / f_total if f_total > 0 else 0):.0%} from '
                f'ensemble disagreement</div>',
                unsafe_allow_html=True,
            )

        st.markdown("<hr>", unsafe_allow_html=True)
        gate_note(p_feas < 0.5)

        cols = st.columns(3)
        for col, (title, unit, attr, t, clip, fmt) in zip(cols, HEADS):
            head_panel(col, title, unit, getattr(out, attr),
                       truth=shown(t), clip01=clip, fmt=fmt)

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(
            '<div class="mono-note">Bands are 90% intervals under the '
            'ensemble predictive distribution. They are only honest if the '
            'model is calibrated. A single PIT value says where one outcome '
            'landed — it cannot diagnose calibration, which needs the PIT '
            'distribution over a whole test set.</div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------- single tab

with tab_single:
    if single_result is None:
        st.error(f"Single-model evaluation failed.\n\n`{single_error}`")
    else:
        pt = single_result["predicted"]

        p_feas_single = as_float(getattr(pt, "feasibility", None))
        p_recovery = as_float(getattr(pt, "recovery", None))
        p_purity = as_float(getattr(pt, "purity", None))
        p_cost = as_float(getattr(pt, "cost_per_kg", None))

        if p_feas_single is not None:
            gate_col, note_col = st.columns([1, 2])
            with gate_col:
                feasibility_verdict(p_feas_single, "single")
            with note_col:
                st.markdown('<div class="eyebrow">Where the doubt comes '
                            'from</div>', unsafe_allow_html=True)
                st.markdown(
                    '<div class="subtle">Nowhere it can name. One model '
                    'cannot disagree with itself, so there is no epistemic '
                    'term to read — only the probability, taken on '
                    'faith.</div>',
                    unsafe_allow_html=True,
                )
            st.markdown("<hr>", unsafe_allow_html=True)
            gate_note(p_feas_single < 0.5)

        point_values = {"recovery": p_recovery, "purity": p_purity,
                        "cost_per_kg": p_cost}

        cols = st.columns(3)
        for col, (title, unit, attr, t, clip, fmt) in zip(cols, HEADS):
            point_panel(col, title, unit, point_values[attr],
                        truth=shown(t), clip01=clip, fmt=fmt)

        st.markdown("<hr>", unsafe_allow_html=True)

        # Head-to-head, but only where both models actually produced a number.
        if ens_result is not None and not infeasible_in_truth:
            ens_out = ens_result["predicted"]
            rows = []
            for title, unit, attr, t, clip, fmt in HEADS:
                pv = point_values[attr]
                if t is None or pv is None:
                    continue
                mu = float(getattr(ens_out, attr)["dist"].mean)
                e_ens, e_pt = abs(mu - t), abs(pv - t)
                if e_ens < e_pt:
                    who, who_colour = "ensemble", COPPER
                elif e_pt < e_ens:
                    who, who_colour = "single", SINGLE
                else:
                    who, who_colour = "tied", MIST
                rows.append(
                    f'{title}: ensemble {e_ens:.3f} vs single {e_pt:.3f} '
                    f'&nbsp;→&nbsp; <span style="color:{who_colour}">{who}'
                    f'</span>'
                )
            if rows:
                st.markdown('<div class="eyebrow">Absolute error, '
                            'head to head</div>', unsafe_allow_html=True)
                st.markdown(
                    '<div class="mono-note">' + '<br>'.join(rows) + '</div>',
                    unsafe_allow_html=True,
                )
                st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(
            '<div class="mono-note">A point estimate carries no interval, so '
            'there is no PIT and no calibration to check. Where the two '
            'models have similar errors, the difference is not accuracy — it '
            'is that only one of them told you how far to trust it.</div>',
            unsafe_allow_html=True,
        )