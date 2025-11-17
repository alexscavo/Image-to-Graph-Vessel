import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import Compose, Normalize
# from fitsne import FItSNE


def draw_graph(nodes, edges, ax):
    ax.set_xlim(0,128)
    ax.set_ylim(0, 128)

    xs = nodes[:, 0] * 128.
    ys = nodes[:, 1] * 128.
    xs = 128. - xs
    ax.scatter(ys, xs)

    # Add all edges
    for edge in edges:
        ax.plot([ys[edge[0]], ys[edge[1]]], [xs[edge[0]], xs[edge[1]]], color="black")

def create_sample_visual(samples, number_samples=10):
    import numpy as np
    import matplotlib.pyplot as plt

    px = 1 / plt.rcParams['figure.dpi']  # pixel in inches
    number_samples = min(number_samples, len(samples["images"]))

    # always force axs to be 2D (even for one sample)
    fig, axs = plt.subplots(number_samples, 3, figsize=(1000 * px, number_samples * 300 * px), squeeze=False)

    for i in range(number_samples):
        # Input image
        axs[i, 0].imshow(
            inv_norm(samples["images"][i].clone().cpu().detach()).permute(1, 2, 0)
        )
        axs[i, 0].axis("off")

        # Ground truth graph
        plt.sca(axs[i, 1])
        draw_graph(
            samples["nodes"][i].clone().cpu().detach(),
            samples["edges"][i].clone().cpu().detach(),
            axs[i, 1],
        )
        axs[i, 1].axis("off")

        # Predicted graph
        plt.sca(axs[i, 2])
        draw_graph(
            samples["pred_nodes"][i].clone().cpu().detach(),
            samples["pred_edges"][i],
            axs[i, 2],
        )
        axs[i, 2].axis("off")

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()

    # TkAgg provides ARGB; convert to RGB
    argb = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(h, w, 4)
    rgb = argb[..., 1:4].copy()  # drop alpha, keep R,G,B
    plt.close(fig)

    # channels-first with batch dim: (1, 3, H, W)
    res = np.transpose(rgb, (2, 0, 1))[None, ...]
    return res



inv_norm = Compose([
    Normalize(
        mean=[0., 0., 0.],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    ),
    Normalize(
        mean=[-0.485, -0.456, -0.406],
        std=[1., 1., 1.]
    ),
])


