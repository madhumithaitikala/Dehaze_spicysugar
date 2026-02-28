"""
AOD-Net Training Script
Trains AOD-Net on paired (hazy, clear) images from the I-HAZE dataset.

Usage:
    python aodnet_train.py --hazy_dir data/I-HAZE/train/hazy --clear_dir data/I-HAZE/train/clear --epochs 100
"""

import argparse
import os
import glob
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from aodnet_model import AODNet
import re


class HazeDataset(Dataset):
    """Dataset for paired hazy/clear images."""
    def __init__(self, hazy_dir, clear_dir, image_size=256, augment=True):
        self.hazy_dir = hazy_dir
        self.clear_dir = clear_dir
        self.image_size = image_size
        self.augment = augment

        # Find all hazy images and match to clear
        self.pairs = []
        for hazy_file in sorted(os.listdir(hazy_dir)):
            if not hazy_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            # Match hazy to clear filename
            basename = os.path.splitext(hazy_file)[0]
            cleaned = re.sub(r'(_hazy|_fog|_input)', '', basename)
            ext = os.path.splitext(hazy_file)[1]

            clear_file = None
            for candidate_ext in [ext, '.png', '.jpg']:
                candidate = cleaned + candidate_ext
                if os.path.exists(os.path.join(clear_dir, candidate)):
                    clear_file = candidate
                    break

            if clear_file:
                self.pairs.append((hazy_file, clear_file))

        print(f"  Found {len(self.pairs)} paired images")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        hazy_path = os.path.join(self.hazy_dir, self.pairs[idx][0])
        clear_path = os.path.join(self.clear_dir, self.pairs[idx][1])

        hazy = cv2.imread(hazy_path)
        clear = cv2.imread(clear_path)

        # Resize
        hazy = cv2.resize(hazy, (self.image_size, self.image_size))
        clear = cv2.resize(clear, (self.image_size, self.image_size))

        # Random augmentation
        if self.augment:
            if random.random() > 0.5:
                hazy = cv2.flip(hazy, 1)
                clear = cv2.flip(clear, 1)
            if random.random() > 0.5:
                hazy = cv2.flip(hazy, 0)
                clear = cv2.flip(clear, 0)

        # BGR -> RGB, HWC -> CHW, normalize to [0, 1]
        hazy = cv2.cvtColor(hazy, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        clear = cv2.cvtColor(clear, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        hazy = torch.from_numpy(hazy).permute(2, 0, 1)
        clear = torch.from_numpy(clear).permute(2, 0, 1)

        return hazy, clear


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  AOD-Net Training")
    print(f"  Device: {device}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Image Size: {args.image_size}\n")

    # Dataset and DataLoader
    train_dataset = HazeDataset(args.hazy_dir, args.clear_dir,
                                 image_size=args.image_size, augment=True)

    if len(train_dataset) == 0:
        print("  ERROR: No paired images found!")
        return

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                               shuffle=True, num_workers=0, drop_last=True)

    # Validation dataset (if provided)
    val_loader = None
    if args.val_hazy_dir and args.val_clear_dir:
        val_dataset = HazeDataset(args.val_hazy_dir, args.val_clear_dir,
                                   image_size=args.image_size, augment=False)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    # Model, loss, optimizer
    model = AODNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=max(1, args.epochs // 3), gamma=0.5)

    # Training loop
    os.makedirs(args.save_dir, exist_ok=True)
    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0

        for batch_idx, (hazy, clear) in enumerate(train_loader):
            hazy, clear = hazy.to(device), clear.to(device)

            optimizer.zero_grad()
            output = model(hazy)
            loss = criterion(output, clear)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)

        # Validation
        val_msg = ""
        if val_loader:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for hazy, clear in val_loader:
                    hazy, clear = hazy.to(device), clear.to(device)
                    output = model(hazy)
                    val_loss += criterion(output, clear).item()
            avg_val = val_loss / len(val_loader)
            val_msg = f" | Val Loss: {avg_val:.6f}"

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                torch.save(model.state_dict(), os.path.join(args.save_dir, 'aodnet_best.pth'))
                val_msg += " *"

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch [{epoch}/{args.epochs}] Train Loss: {avg_loss:.6f}{val_msg}")

    # Save final model
    final_path = os.path.join(args.save_dir, 'aodnet_final.pth')
    torch.save(model.state_dict(), final_path)
    print(f"\n  Training complete! Model saved to {final_path}")
    if os.path.exists(os.path.join(args.save_dir, 'aodnet_best.pth')):
        print(f"  Best validation model saved to {os.path.join(args.save_dir, 'aodnet_best.pth')}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train AOD-Net on paired hazy/clear images")
    parser.add_argument('--hazy_dir', required=True, help="Training hazy images directory")
    parser.add_argument('--clear_dir', required=True, help="Training clear images directory")
    parser.add_argument('--val_hazy_dir', default=None, help="Validation hazy images directory")
    parser.add_argument('--val_clear_dir', default=None, help="Validation clear images directory")
    parser.add_argument('--epochs', type=int, default=100, help="Number of epochs (default: 100)")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size (default: 4)")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate (default: 0.001)")
    parser.add_argument('--image_size', type=int, default=256, help="Image size (default: 256)")
    parser.add_argument('--save_dir', default='models/aodnet', help="Directory to save model weights")
    args = parser.parse_args()

    train(args)
