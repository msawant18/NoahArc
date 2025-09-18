"""
Training loop.
"""
# src/models/train.py
import argparse
import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from src.models.unet import UNet
from src.evaluation.metrics import dice_coef, iou_score, precision_score, recall_score, f1_score
from tqdm import trange

class TileDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.folder = Path(folder)
        self.items = sorted([p for p in self.folder.glob("*.npy") if p.name.endswith(".npy") and "mask" not in p.name])
        self.transform = transform
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        tpath = self.items[idx]
        mpath = Path(str(tpath).replace(".npy", ".mask.npy"))
        tile = np.load(str(tpath)).astype(np.float32)
        mask = np.load(str(mpath)).astype(np.uint8)
        # normalize per tile (zscore)
        nanmask = np.isnan(tile)
        tile = np.where(nanmask, 0.0, tile)
        tile = (tile - tile.mean()) / (tile.std() + 1e-6)
        x = np.expand_dims(tile, 0)
        y = np.expand_dims(mask, 0).astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)

def save_checkpoint(state, path):
    torch.save(state, path)

def train(config, out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = TileDataset(config.get("data_path", "data/processed"))
    n = len(ds)
    split = int(0.8 * n)
    train_ds = torch.utils.data.Subset(ds, list(range(0, split)))
    val_ds = torch.utils.data.Subset(ds, list(range(split, n)))
    tr = DataLoader(train_ds, batch_size=config.get("batch_size", 8), shuffle=True, num_workers=0)
    va = DataLoader(val_ds, batch_size=config.get("batch_size", 8), shuffle=False, num_workers=0)

    model = UNet(in_channels=1, out_channels=1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config.get("learning_rate", 1e-3), weight_decay=config.get("weight_decay",1e-5))
    bce = nn.BCEWithLogitsLoss()
    best_iou = 0.0

    epochs = config.get("epochs", 10)
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for x,y in tr:
            x = x.to(device); y = y.to(device)
            logits = model(x)
            loss_bce = bce(logits, y)
            # dice
            probs = torch.sigmoid(logits)
            inter = (probs * y).sum()
            union = probs.sum() + y.sum()
            dice = 1 - (2. * inter + 1e-6) / (union + 1e-6)
            loss = loss_bce + dice
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        # validation
        model.eval()
        ious=[]; f1s=[]
        with torch.no_grad():
            for x,y in va:
                x = x.to(device); y = y.to(device)
                logits = model(x)
                probs = torch.sigmoid(logits).cpu().numpy()
                preds = (probs >= 0.5).astype(np.uint8)
                gt = y.cpu().numpy().astype(np.uint8)
                for p,g in zip(preds, gt):
                    p = p[0]; g = g[0]
                    ious.append(iou_score(g, p))
                    f1s.append(f1_score(g, p))
        mean_iou = float(np.mean(ious)) if len(ious)>0 else 0.0
        mean_f1 = float(np.mean(f1s)) if len(f1s)>0 else 0.0
        print(f"Epoch {epoch+1}/{epochs} loss={epoch_loss:.4f} val_iou={mean_iou:.4f} val_f1={mean_f1:.4f}")
        ckpt = os.path.join(out_dir, "check_epoch_%03d.pt"% (epoch+1))
        save_checkpoint({"epoch":epoch+1, "model_state":model.state_dict(), "opt_state":opt.state_dict()}, ckpt)
        # save best
        if mean_iou > best_iou:
            best_iou = mean_iou
            save_checkpoint({"epoch":epoch+1, "model_state":model.state_dict()}, os.path.join(out_dir, "best.pt"))
    # final save
    save_checkpoint({"epoch":epochs, "model_state":model.state_dict()}, os.path.join(out_dir, "final.pt"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/model_config.yaml")
    parser.add_argument("--out_dir", default="artifacts/checkpoints")
    args = parser.parse_args()
    # basic config loader
    cfg = {"data_path":"data/processed"}
    if Path(args.config).exists():
        with open(args.config) as f:
            import yaml
            cfg.update(yaml.safe_load(f))
    train(cfg, args.out_dir)
