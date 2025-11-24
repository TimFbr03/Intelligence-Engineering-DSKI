from transformers import EarlyStoppingCallback


def build_callbacks(cfg: dict):
    """
    Baut alle Callbacks, die im Training genutzt werden.
    """
    callbacks = []

    # Early Stopping aktivieren?
    if cfg.get("early_stopping", True):
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=cfg.get("early_stopping_patience", 2),
                early_stopping_threshold=cfg.get("early_stopping_threshold", 0.0),
            )
        )

    return callbacks

