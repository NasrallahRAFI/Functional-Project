"""Plain PyTorch CPU loop shared by the MLP and CNN experiments."""
from __future__ import annotations

import copy
import time
import warnings
from typing import Callable

import numpy as np
import torch
from torch import nn


def train_torch_model(model: nn.Module, train_batch_fn: Callable,
                      validation_X: np.ndarray, y_train: np.ndarray, y_validation: np.ndarray,
                      *, seed: int, batch_size: int, learning_rate: float = 1e-3,
                      weight_decay: float = 1e-2, max_epochs: int = 100,
                      patience: int = 8, device: str = "cpu", expected_threads: int | None = None,
                      lineage_hash: str = "") -> dict:
    """Train with seeded per-epoch sampling, weighted BCE, plateau scheduling and early stop.

    ``train_batch_fn(indices, rng)`` returns the input batch for those clip rows.
    It is invoked anew each epoch, allowing cached-power windows to be sampled again.
    """
    if device != "cpu":
        raise ValueError("the approved training device is CPU")
    if expected_threads is not None and torch.get_num_threads() != expected_threads:
        raise RuntimeError(f"torch thread count changed: {torch.get_num_threads()} != {expected_threads}")
    y_train = np.asarray(y_train, dtype=np.float32).reshape(-1)
    y_validation = np.asarray(y_validation, dtype=np.float32).reshape(-1)
    validation_X = np.asarray(validation_X, dtype=np.float32)
    if len(y_train) < 1 or len(y_validation) < 1 or not np.isfinite(validation_X).all():
        raise ValueError("training and validation arrays must be nonempty and finite")
    if not np.isin(y_train, (0, 1)).all() or not np.isin(y_validation, (0, 1)).all():
        raise ValueError("labels must be binary")
    n_pos = int(np.sum(y_train == 1)); n_neg = int(np.sum(y_train == 0))
    if not n_pos or not n_neg:
        raise ValueError("training needs both classes")
    if batch_size < 1 or max_epochs < 1 or patience < 1:
        raise ValueError("invalid training-loop limits")
    model = model.to(torch.device("cpu"))
    torch.manual_seed(seed)
    np.random.seed(seed)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(n_neg / n_pos, dtype=torch.float32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3)
    history: list[dict] = []
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    best_logits = None
    stale_epochs = 0
    started = time.perf_counter()
    warning_messages: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for epoch in range(max_epochs):
            model.train()
            epoch_seed = int(seed) * 1_000_003 + epoch
            rng = np.random.default_rng(epoch_seed)
            order_rng = np.random.default_rng(epoch_seed ^ 0x5EED)
            order = order_rng.permutation(len(y_train))
            loss_sum = 0.0
            seen = 0
            for start in range(0, len(order), batch_size):
                batch_indices = order[start:start + batch_size]
                batch_X = np.asarray(train_batch_fn(batch_indices, rng), dtype=np.float32)
                batch_y = torch.as_tensor(y_train[batch_indices], dtype=torch.float32)
                if not np.isfinite(batch_X).all():
                    raise ValueError("training batch contains NaN or infinity")
                inputs = torch.from_numpy(batch_X)
                optimizer.zero_grad(set_to_none=True)
                logits = model(inputs)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.detach()) * len(batch_indices)
                seen += len(batch_indices)
            train_loss = loss_sum / seen
            model.eval()
            with torch.no_grad():
                logits = model(torch.from_numpy(validation_X)).detach().cpu().numpy().astype(np.float64)
                val_loss = float(criterion(torch.from_numpy(logits.astype(np.float32)),
                                           torch.from_numpy(y_validation)).item())
            lr = float(optimizer.param_groups[0]["lr"])
            scheduler.step(val_loss)
            history.append({"epoch": epoch + 1, "train_loss": train_loss,
                            "validation_loss": val_loss, "learning_rate": lr})
            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch + 1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_logits = logits.copy()
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    break
        warning_messages.extend(str(w.message) for w in caught)
    if best_state is None or best_logits is None:
        raise RuntimeError("training produced no validation checkpoint")
    model.load_state_dict(best_state)
    return {"model": model, "state_dict": best_state, "history": history,
            "best_epoch": best_epoch, "best_validation_loss": float(best_loss),
            "validation_logits": best_logits, "training_seconds": time.perf_counter() - started,
            "seed": int(seed), "device": device, "torch_threads": torch.get_num_threads(),
            "pos_weight": float(n_neg / n_pos), "lineage_hash": lineage_hash,
            "warnings": warning_messages, "best_state_hash": _state_hash(best_state)}


def _state_hash(state: dict[str, torch.Tensor]) -> str:
    import hashlib
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()
