"""
Gradio demo: N x M grid of faders (sliders) feeding a pretrained model,
output rendered live as a matrix / heatmap.

Assumes `model` is already loaded and in eval mode. Replace the
`load_model()` and `predict()` internals with your actual model + I/O shape.
"""
import math

import gradio as gr
import numpy as np
import torch
import matplotlib.pyplot as plt

from config import PRED_METRICS
from datasets import Dataset
from evaluate import manual_eval
from models import load_model, Model
from solvent_recovery.properties import get_solvent_props, get_water_props, get_salt_props, get_solids_props, \
    get_extractant_props

# ---- 1. Load your pretrained model once, at startup ----

predicted_metric = 'cost_per_kg'

model_name = 'first_good.pt'
solvent_target_name = 'nmp'
solvent2_name = 'ethanol'
salt_name = 'sodium chloride'
solvent_target_flow = 1000
solvent2_flow = 300
water_flow = 100

model = Model()
model.load_state_dict(torch.load(model_name)['model_state_dict'])
model.eval()

dataset = Dataset('train')

def make_plots(matrix1, matrix2):
    xs, ys, zs = np.meshgrid(
        np.arange(matrix1.shape[0]),
        np.arange(matrix1.shape[1]),
        np.arange(matrix1.shape[2]),
        indexing="ij",
    )

    fig1, ax1 = plt.subplots(subplot_kw={"projection": "3d"})

    if not predicted_metric == 'cost_per_kg':
        sc = ax1.scatter(xs, ys, zs, c=matrix1.flatten(), cmap="viridis", s=100, vmin=0, vmax=1)
    else:
        sc = ax1.scatter(xs, ys, zs, c=matrix1.flatten(), cmap="viridis", s=100)
    fig1.colorbar(sc)
    ax1.set_xlabel("Recovery")
    ax1.set_xticks(np.arange(4))
    ax1.set_xticklabels(['Bypass', 'Distillation', 'Pervaporation', 'ATPE'])
    ax1.set_ylabel("Purification")
    ax1.set_yticks(np.arange(4))
    ax1.set_yticklabels(['Bypass', 'Distillation', 'Pervaporation', 'Ultrafiltration'])
    ax1.set_zlabel("Refinement")
    ax1.set_zticks(np.arange(5))
    ax1.set_zticklabels(['Bypass', 'Distillation', 'Pervaporation', 'Ultrafiltration', 'Nanofiltration'])

    fig2, ax2 = plt.subplots(subplot_kw={"projection": "3d"})
    xs, ys, zs = np.meshgrid(
        np.arange(matrix2.shape[0]),
        np.arange(matrix2.shape[1]),
        np.arange(matrix2.shape[2]),
        indexing="ij",
    )

#    if model.output == 'feasibility' or model.output == 'fractions':
    if not predicted_metric == 'cost_per_kg':
        sc = ax2.scatter(xs, ys, zs, c=matrix2.flatten(), cmap="viridis", s=100, vmin=0, vmax=1)
    else:
        sc = ax2.scatter(xs, ys, zs, c=matrix2.flatten(), cmap="viridis", s=100)
    fig2.colorbar(sc)
    ax2.set_xlabel("Recovery")
    ax2.set_xticks(np.arange(4))
    ax2.set_xticklabels(['Bypass', 'Distillation', 'Pervaporation', 'ATPE'])
    ax2.set_ylabel("Purification")
    ax2.set_yticks(np.arange(4))
    ax2.set_yticklabels(['Bypass', 'Distillation', 'Pervaporation', 'Ultrafiltration'])
    ax2.set_zlabel("Refinement")
    ax2.set_zticks(np.arange(5))
    ax2.set_zticklabels(['Bypass', 'Distillation', 'Pervaporation', 'Ultrafiltration', 'Nanofiltration'])

    return fig1, fig2

# ---- 2. Inference function ----
def predict(*fader_values):
    props = {
        "target": get_solvent_props(solvent_target_name),
        "solvent2": get_solvent_props(solvent2_name),
        "water": get_water_props(),
        "salt": get_salt_props(salt_name),
        "solids": get_solids_props(),
        "extractant": get_extractant_props(),
    }

    x = manual_eval(model=model,
                    props=props,
                    dataset=dataset,
                    solvent_target_flow=fader_values[1],
                    solvent2_flow=fader_values[2],
                    water_flow=fader_values[5],
                    salt_flow=fader_values[3],
                    solids_flow=fader_values[4],
                    solid_removal_idxs=[round(fader_values[0])],
                    recovery_idxs=[0, 1, 2, 3],
                    purification_idxs=[0, 1, 2, 3],
                    refinement_idxs=[0, 1, 2, 3, 4])

    predicted = x['predicted'][round(fader_values[0]), :, :, :, PRED_METRICS[predicted_metric]]
    true = x['true'][round(fader_values[0]), :, :, :, PRED_METRICS[predicted_metric]]
    return make_plots(predicted, true)


# ---- 3. Build the fader grid UI ----
with gr.Blocks() as demo:
    gr.Markdown(f"## {predicted_metric} prediction for extracting {solvent_target_name} from a mixture with {solvent2_name} and {salt_name}")

    sliders = []

    with gr.Column():
        with gr.Row():
            s = gr.Slider(minimum=0,
                          maximum=3,
                          step=1,
                          label='solid_removal_idx')
            sliders.append(s)
            s = gr.Slider(minimum=1,
                          maximum=2000,
                          step=1,
                          label='solvent target flow kg/h')
            sliders.append(s)
            s = gr.Slider(minimum=0,
                          maximum=1000,
                          step=1,
                          label='solvent 2 flow kg/h')
            sliders.append(s)
            s = gr.Slider(minimum=0,
                          maximum=200,
                          step=1,
                          label='salt flow kg/h')
            sliders.append(s)
            s = gr.Slider(minimum=0,
                          maximum=100,
                          step=1,
                          label='solids flow kg/h')
            sliders.append(s)
            s = gr.Slider(minimum=0,
                          maximum=1500,
                          step=1,
                          label='water flow kg/h')
            sliders.append(s)

    # output_plot = gr.Plot(label="Output")

    with gr.Row():
        plot1 = gr.Plot(label="Predicted target purity")
        plot2 = gr.Plot(label="True target purity")

    # Fire prediction on every slider move (live), not just on release
    for s in sliders:
        s.change(fn=predict, inputs=sliders, outputs=[plot1, plot2], show_progress='hidden')

    # Initial render
    demo.load(fn=predict, inputs=sliders, outputs=[plot1, plot2])


def compare_dicts(a: dict[str, float], b: dict[str, float],
                  name_a: str = "A", name_b: str = "B") -> str:
    keys = list(a)  # assumes same keys; use a.keys() | b.keys() if not guaranteed
    kw = max(len(k) for k in keys)
    cw = max(len(name_a), len(name_b), 9)  # 9 fits a formatted float

    lines = [f"{'':<{kw}}  {name_a:>{cw}}  {name_b:>{cw}}  {'Δ':>{cw}}"]
    for k in keys:
        d = b[k] - a[k]
        lines.append(
            f"{k:<{kw}}  {a[k]:>{cw}.3f}  {b[k]:>{cw}.3f}  {d:>{cw}.3f}"
        )

    lines.append('\n')
    return "\n".join(lines)

if __name__ == "__main__":
    demo.launch()