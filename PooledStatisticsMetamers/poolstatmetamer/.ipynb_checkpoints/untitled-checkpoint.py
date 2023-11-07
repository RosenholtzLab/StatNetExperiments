import numpy as np
import shap
import torch
from torch import nn
import torch

      class EmbeddingModel(nn.Module):
        def __init__(self):
            super(EmbeddingModel, self).__init__()
            self.layer1 = nn.Linear(150, 50)

        def forward(self, x):
            x = self.layer1(x)
            return x


    model = EmbeddingModel()
    model.load_state_dict(torch.load('contrastive_model_HB_KTH.pth'))

    loaded = np.load("example_inputs.npz")
    stats = torch.tensor(loaded["stats"])
    labels = torch.tensor(loaded["labels"])

    b, n_images, n_stats = stats.shape

    assert (labels[:, 0] == labels[:, 1]).all(), "Labels need to be not permuted"

    def sim_func(mask):
        with torch.no_grad():

            mask = torch.tensor(mask).to(torch.float32)

            print(mask.shape)
            b, n, c = stats.shape
            d, c2 = mask.shape
            assert c == c2
            masked_stats = (torch.einsum("bnc,dc->bdnc", stats, mask)
                            .reshape(b * d * n, c))

            embs = (model
                    .forward(masked_stats)
                    .reshape(b, d, n_images, -1))

            return (torch.einsum("bdnc,bdmc->bdnm", embs, embs)
                    .mean(dim=[0, 2, 3])
                    .unsqueeze(-1).numpy())

    reference = np.zeros((1, n_stats))
    input = np.ones((1, n_stats))
    explainer = shap.KernelExplainer(
        sim_func,
        reference
    )

    shap_values = np.abs(explainer.shap_values(input)[0].squeeze())

    weight_values = model.layer1.weight.square().sum(1).sqrt()

    import matplotlib.pyplot as plt
    plt.scatter(shap_values, weight_values, alpha=.5)
    plt.show()