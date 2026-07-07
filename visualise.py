"""
Gradio demo: N x M grid of faders (sliders) feeding a pretrained model,
output rendered live as a matrix / heatmap.

Assumes `model` is already loaded and in eval mode. Replace the
`load_model()` and `predict()` internals with your actual model + I/O shape.
"""

import gradio as gr
import numpy as np
import torch
import matplotlib.pyplot as plt

from evaluate import manual_eval
from models import load_model

# ---- 1. Load your pretrained model once, at startup ----

model_name = 'best_06-07-26_feasibility2.pt'
solvent_target = '2-methyltetrahydrofuran'
solvent2 = 'acetone'
salt = 'sodium bicarbonate'

N_ROWS, N_COLS = 4, 4  # shape of the fader grid

# ---- 2. Inference function ----
def predict(*fader_values):
    output_matrix = np.zeros((4, 4))

    x = manual_eval(model_name,
                    solvent_target_name=solvent_target,
                    solvent2_name=solvent2,
                    salt_name=salt,
                    solvent_target_flow=2,
                    solvent2_flow=2,
                    water_flow=2,
                    salt_flow=2,
                    solids_flow=2,
                    solid_removal_idxs=[0, 1, 2, 3],
                    recovery_idxs=[0, 1, 2, 3],
                    purification_idxs=[fader_values[0]],
                    refinement_idxs=[fader_values[1]],
                    ground_truth=False)

    output_matrix = x['predicted'][:, :, fader_values[0], fader_values[1]]

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(output_matrix, cmap="viridis")
    fig.colorbar(im, ax=ax)
    ax.set_title("Model output")
    plt.tight_layout()
    return fig


# ---- 3. Build the fader grid UI ----
with gr.Blocks() as demo:
    gr.Markdown("## Parameter matrix -> model -> live output")

    sliders = []

    with gr.Column():
        with gr.Row():
            s = gr.Slider(minimum=0,
                          maximum=3,
                          step=1,
                          label='purification_idx')
            sliders.append(s)
            s = gr.Slider(minimum=0,
                          maximum=3,
                          step=1,
                          label='refinement_idx')
            sliders.append(s)

    output_plot = gr.Plot(label="Output")

    # Fire prediction on every slider move (live), not just on release
    for s in sliders:
        s.change(fn=predict, inputs=sliders, outputs=output_plot)

    # Initial render
    demo.load(fn=predict, inputs=sliders, outputs=output_plot)

if __name__ == "__main__":
    demo.launch()