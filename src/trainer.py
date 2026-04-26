import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from config import DEVICE, BATCH_SIZE, LR, WEIGHT_DECAY

class Trainer:
    def __init__(self, model, X_train, X_val=None, lr=LR, batch_size=BATCH_SIZE, device=DEVICE, patience=8):
        """
        X_train and X_val should be numpy arrays of windows. For AE, X_train is expected to contain mostly/only normal windows.
        """
        self.device = device
        self.model = model.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
        self.batch_size = batch_size
        self.patience = patience

        # datasets
        self.train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32))
        self.val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32)) if X_val is not None else None

        # scheduler
        try:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=4, verbose=False)
        except TypeError:
    # Fallback for older PyTorch versions (no 'verbose' argument)
         self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=4)


    def train(self, epochs=50):
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False) if self.val_dataset is not None else None

        train_losses, val_losses = [], []
        best_val = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            # training pass
            self.model.train()
            total_train = 0.0
            for batch in train_loader:
                batch_x = batch[0].to(self.device)
                recon, _ = self.model(batch_x)
                loss = self.criterion(recon, batch_x)
                self.optimizer.zero_grad()
                loss.backward()
                # gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                total_train += loss.item()
            avg_train = total_train / max(1, len(train_loader))
            train_losses.append(avg_train)

            # validation
            if val_loader is not None:
                self.model.eval()
                total_val = 0.0
                with torch.no_grad():
                    for batch in val_loader:
                        batch_x = batch[0].to(self.device)
                        recon, _ = self.model(batch_x)
                        total_val += self.criterion(recon, batch_x).item()
                avg_val = total_val / max(1, len(val_loader))
                val_losses.append(avg_val)

                # scheduler step
                self.scheduler.step(avg_val)

                # early stopping
                if avg_val < best_val - 1e-6:
                    best_val = avg_val
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1

                print(f"Epoch [{epoch}/{epochs}] | Train Loss: {avg_train:.6f} | Val Loss: {avg_val:.6f} | Patience: {patience_counter}/{self.patience}")

                if patience_counter >= self.patience:
                    print("⚠️ Early stopping triggered.")
                    break
            else:
                print(f"Epoch [{epoch}/{epochs}] | Train Loss: {avg_train:.6f}")

        # restore best state if available
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return {"train_loss": train_losses, "val_loss": val_losses}

    def evaluate(self, X_test, y_test=None):
        """
        Returns:
        recon_errors: (n_windows,)
        recons: (n_windows, seq_len, features)
        latents: (n_windows, latent_dim)
        y_test: (n_windows,) or None
        """
        self.model.eval()
        test_tensor = torch.tensor(X_test, dtype=torch.float32)
        test_loader = DataLoader(TensorDataset(test_tensor), batch_size=self.batch_size, shuffle=False, drop_last=False)

        recons_list = []
        latents_list = []
        errors = []
        with torch.no_grad():
            for batch in test_loader:
                batch_x = batch[0].to(self.device)
                recon, lat = self.model(batch_x)
                batch_errors = torch.mean((recon - batch_x) ** 2, dim=(1, 2)).cpu().numpy()
                errors.extend(batch_errors.tolist())
                recons_list.append(recon.cpu().numpy())
                latents_list.append(lat.cpu().numpy())

        recons = np.concatenate(recons_list, axis=0) if len(recons_list) else np.empty((0, X_test.shape[1], X_test.shape[2]))
        latents = np.concatenate(latents_list, axis=0) if len(latents_list) else np.empty((0, 0))
        return np.array(errors), recons, latents, (np.array(y_test) if y_test is not None else None)
