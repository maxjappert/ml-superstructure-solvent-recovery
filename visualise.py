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

model_name = 'best_06-07-26_feasibility2.pt'
solvent_target_name = 'dmso'
solvent2_name = 'ethyl acetate'
salt_name = 'magnesium sulfate'
solvent_target_flow = 1000
solvent2_flow = 300
water_flow = 100

model = Model()
model.load_state_dict(torch.load(model_name)['model_state_dict'])
model.eval()

dataset = Dataset('train')

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

    output_matrix = x['true'][round(fader_values[0]), :, :, :]

    xs, ys, zs = np.meshgrid(
        np.arange(output_matrix.shape[0]),
        np.arange(output_matrix.shape[1]),
        np.arange(output_matrix.shape[2]),
        indexing="ij",
    )

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    sc = ax.scatter(xs, ys, zs, c=output_matrix.flatten(), cmap="viridis", s=100)
    fig.colorbar(sc)
    ax.set_xlabel("Recovery")
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(['Bypass', 'Distillation', 'Pervaporation', 'ATPE'])
    ax.set_ylabel("Purification")
    ax.set_yticks(np.arange(4))
    ax.set_yticklabels(['Bypass', 'Distillation', 'Pervaporation', 'Ultrafiltration'])
    ax.set_zlabel("Refinement")
    ax.set_zticks(np.arange(5))
    ax.set_zticklabels(['Bypass', 'Distillation', 'Pervaporation', 'Ultrafiltration', 'Nanofiltration'])
    plt.show()

    # fig, ax = plt.subplots(figsize=(4, 4))
    # im = ax.imshow(output_matrix, cmap="viridis")
    # fig.colorbar(im, ax=ax)
    # ax.set_title("Model output")
    # plt.tight_layout()
    return fig


# ---- 3. Build the fader grid UI ----
with gr.Blocks() as demo:
    gr.Markdown(f"## Flow chart for extracting {solvent_target_name} from a mixture with {solvent2_name} and {salt_name}")

    sliders = []

    with gr.Column():
        with gr.Row():
            s = gr.Slider(minimum=0,
                          maximum=3,
                          step=1,
                          label='solid_removal_idx')
            sliders.append(s)
            s = gr.Slider(minimum=1,
                          maximum=10000,
                          step=1,
                          label='solvent target flow kg/h')
            sliders.append(s)
            s = gr.Slider(minimum=0,
                          maximum=10000,
                          step=1,
                          label='solvent 2 flow kg/h')
            sliders.append(s)
            s = gr.Slider(minimum=0,
                          maximum=10000,
                          step=1,
                          label='salt flow kg/h')
            sliders.append(s)
            s = gr.Slider(minimum=0,
                          maximum=10000,
                          step=1,
                          label='solids flow kg/h')
            sliders.append(s)
            s = gr.Slider(minimum=0,
                          maximum=10000,
                          step=1,
                          label='water flow kg/h')
            sliders.append(s)

    output_plot = gr.Plot(label="Output")

    # Fire prediction on every slider move (live), not just on release
    for s in sliders:
        s.change(fn=predict, inputs=sliders, outputs=output_plot)

    # Initial render
    demo.load(fn=predict, inputs=sliders, outputs=output_plot)

if __name__ == "__main__":
    demo.launch()