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

from datasets import Dataset
from evaluate import manual_eval
from models import load_model, Model
from solvent_recovery.properties import get_solvent_props, get_water_props, get_salt_props, get_solids_props, \
    get_extractant_props

# ---- 1. Load your pretrained model once, at startup ----

model_name = 'fractions_20260707_155708.pt'
solvent_target_name = 'nmp'
solvent2_name = 'ethanol'
salt_name = 'sodium chloride'
output = 'fractions'
solvent_target_flow = 1000
solvent2_flow = 300
water_flow = 100

output = 'fractions'

model = Model(output)
model.load_state_dict(torch.load(model_name)['model_state_dict'])
model.eval()

dataset = Dataset('train', output)

def make_plots(matrix1, matrix2):
    xs, ys, zs = np.meshgrid(
        np.arange(matrix1.shape[0]),
        np.arange(matrix1.shape[1]),
        np.arange(matrix1.shape[2]),
        indexing="ij",
    )

    fig1, ax1 = plt.subplots(subplot_kw={"projection": "3d"})
    if model.output == 'feasibility' or model.output == 'fractions':
        sc = ax1.scatter(xs, ys, zs, c=matrix1.flatten(), cmap="viridis", s=100, vmin=0, vmax=1)
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

    if model.output == 'feasibility' or model.output == 'fractions':
        sc = ax2.scatter(xs, ys, zs, c=matrix2.flatten(), cmap="viridis", s=100, vmin=0, vmax=1)
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
                    refinement_idxs=[0, 1, 2, 3, 4],
                    ground_truth=True)

    if model.output == 'feasibility':
        return make_plots(x['predicted'][round(fader_values[0]), :, :, :], x['true'][round(fader_values[0]), :, :, :])
    elif model.output == 'fractions':
        predicted = x['predicted'][round(fader_values[0]), :, :, :, 1]
        true = x['true'][round(fader_values[0]), :, :, :, 1]
        return make_plots(predicted, true)
    else:
        print('faulty output type specified')
        exit(-1)


# ---- 3. Build the fader grid UI ----
with gr.Blocks() as demo:
    gr.Markdown(f"## {output} prediction for extracting {solvent_target_name} from a mixture with {solvent2_name} and {salt_name}")

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

if __name__ == "__main__":
    demo.launch()