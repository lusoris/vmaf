"""C1 — feature-vector → MOS regressor (replaces / augments SVM)."""

from __future__ import annotations

import lightning as L
import torch
from torch import nn

from ..confidence import gaussian_nll


class FRRegressor(L.LightningModule):
    """Tiny MLP over precomputed libvmaf feature vectors (adm, vif, motion, ...).

    When @c emit_variance is True, the model outputs ``(N, 2)`` — column
    0 is the score (μ), column 1 is log-variance. Training switches
    from MSE to Gaussian NLL so inference-time callers can surface a
    confidence interval per sample. Single-output legacy mode stays the
    default so existing checkpoints continue to load unchanged.
    """

    def __init__(
        self,
        in_features: int = 7,
        hidden: int = 64,
        depth: int = 2,
        dropout: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        emit_variance: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        out_features = 2 if emit_variance else 1
        layers: list[nn.Module] = []
        prev = in_features
        for _ in range(depth):
            layers += [nn.Linear(prev, hidden), nn.GELU(), nn.Dropout(dropout)]
            prev = hidden
        layers.append(nn.Linear(prev, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        if self.hparams.emit_variance:
            return out  # (N, 2): [:, 0] = score, [:, 1] = logvar
        return out.squeeze(-1)

    def _step(self, batch: tuple[torch.Tensor, torch.Tensor], tag: str) -> torch.Tensor:
        x, y = batch
        out = self(x)
        if self.hparams.emit_variance:
            pred = out[..., 0]
            logvar = out[..., 1]
            loss = gaussian_nll(pred, y, logvar).mean()
            self.log(f"{tag}/nll", loss, prog_bar=True, on_epoch=True)
            # Log the plain MSE too so runs with/without variance stay comparable.
            with torch.no_grad():
                self.log(
                    f"{tag}/mse",
                    nn.functional.mse_loss(pred, y),
                    on_epoch=True,
                )
            return loss
        loss = nn.functional.mse_loss(out, y)
        self.log(f"{tag}/mse", loss, prog_bar=True, on_epoch=True)
        return loss

    def training_step(self, batch, _idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch, _idx: int) -> None:
        self._step(batch, "val")

    def test_step(self, batch, _idx: int) -> None:
        self._step(batch, "test")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
