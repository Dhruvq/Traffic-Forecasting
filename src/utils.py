class EarlyStopping:
    def __init__(self, patience: int = 10):
        self.patience = patience
        self.counter = 0
        self.best_loss = float('inf')
        self.best_state = None

    def step(self, val_loss: float, model) -> bool:
        """
        Call once per epoch. Returns True when training should stop.
        Saves the model's state_dict whenever val_loss improves.
        """
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience

    def restore(self, model) -> None:
        """Load the weights from the best epoch back into the model."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
