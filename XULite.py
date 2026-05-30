import os
import sys
import time
import warnings
from datetime import datetime
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device.type == 'cuda':
    torch.cuda.set_device(0)
print("Using:", device, "-", torch.cuda.get_device_name(0) if device.type == 'cuda' else "CPU")

torch.manual_seed(666)

class XConv(nn.Module):
    def __init__(self, channels, kernel_size=7, bias=False):
        super().__init__()
        assert kernel_size % 2 == 1
        padding = kernel_size // 2

        self.diag_conv = nn.Conv2d(
            channels, channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=channels,
            bias=bias
        )
        self.anti_conv = nn.Conv2d(
            channels, channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=channels,
            bias=bias
        )

        diag_mask = torch.zeros(kernel_size, kernel_size)
        for i in range(kernel_size):
            diag_mask[i, i] = 1.0

        anti_mask = torch.zeros(kernel_size, kernel_size)
        for i in range(kernel_size):
            anti_mask[i, kernel_size - 1 - i] = 1.0

        self.register_buffer(
            "diag_mask",
            diag_mask.view(1, 1, kernel_size, kernel_size).expand(channels, 1, kernel_size, kernel_size)
        )
        self.register_buffer(
            "anti_mask",
            anti_mask.view(1, 1, kernel_size, kernel_size).expand(channels, 1, kernel_size, kernel_size)
        )

    def forward(self, x):
        w_diag = self.diag_conv.weight * self.diag_mask
        w_anti = self.anti_conv.weight * self.anti_mask

        y_diag = F.conv2d(
            x, weight=w_diag, bias=self.diag_conv.bias,
            stride=self.diag_conv.stride, padding=self.diag_conv.padding,
            dilation=self.diag_conv.dilation, groups=self.diag_conv.groups,
        )
        y_anti = F.conv2d(
            x, weight=w_anti, bias=self.anti_conv.bias,
            stride=self.anti_conv.stride, padding=self.anti_conv.padding,
            dilation=self.anti_conv.dilation, groups=self.anti_conv.groups,
        )
        return y_diag + y_anti


class AxialDW(nn.Module):
    def __init__(self, channels, bias=False):
        super().__init__()
        self.dw_1x7 = nn.Conv2d(
            channels, channels,
            kernel_size=(1, 7), padding=(0, 3),
            groups=channels, bias=bias
        )
        self.dw_7x1 = nn.Conv2d(
            channels, channels,
            kernel_size=(7, 1), padding=(3, 0),
            groups=channels, bias=bias
        )

    def forward(self, x):
        return x + self.dw_1x7(x) + self.dw_7x1(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.xconv = XConv(in_c, kernel_size=7, bias=False)
        self.bn = nn.BatchNorm2d(in_c)
        self.pw = nn.Conv2d(in_c, out_c, kernel_size=1, bias=False)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.act = nn.GELU()

    def forward(self, x):
        skip = self.bn(self.xconv(x) + x)
        out = self.act(self.pool(self.pw(skip)))
        return out, skip


class DecoderBlock(nn.Module):
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_c)
        self.pw1 = nn.Conv2d(in_c + skip_c, out_c, kernel_size=1, bias=False)
        self.xconv = XConv(out_c, kernel_size=7, bias=False)
        self.pw2 = nn.Conv2d(out_c, out_c, kernel_size=1, bias=False)
        self.act = nn.GELU()

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=True)
        x = self.bn(x)
        x = torch.cat([x, skip], dim=1)
        x = self.pw1(x)
        x = self.xconv(x) + x
        x = self.act(self.pw2(x))
        return x


class BottleNeckBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.axial = AxialDW(channels, bias=False)
        self.xconv = XConv(channels, kernel_size=7, bias=False)
        self.dw3 = nn.Conv2d(
            channels, channels,
            kernel_size=3, padding=1,
            groups=channels, bias=False
        )
        self.bn = nn.BatchNorm2d(channels * 3)
        self.pw = nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False)
        self.act = nn.GELU()

    def forward(self, x):
        b1 = self.axial(x)
        b2 = self.xconv(x)
        b3 = self.dw3(x)
        out = torch.cat([b1, b2, b3], dim=1)
        out = self.bn(out)
        out = self.pw(out)
        out = self.act(out)
        return out


class XULite(nn.Module):
    def __init__(self, img_channels=1, num_classes=1, channels=(12, 24, 48, 96, 192, 424)):
        super().__init__()
        c1, c2, c3, c4, c5, c6 = channels

        self.stem = nn.Sequential(
            nn.Conv2d(img_channels, c1, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(c1),
            nn.GELU()
        )

        self.enc1 = EncoderBlock(c1, c2)
        self.enc2 = EncoderBlock(c2, c3)
        self.enc3 = EncoderBlock(c3, c4)
        self.enc4 = EncoderBlock(c4, c5)
        self.enc5 = EncoderBlock(c5, c6)

        self.bottleneck = BottleNeckBlock(c6)

        self.dec5 = DecoderBlock(c6, c5, c5)
        self.dec4 = DecoderBlock(c5, c4, c4)
        self.dec3 = DecoderBlock(c4, c3, c3)
        self.dec2 = DecoderBlock(c3, c2, c2)
        self.dec1 = DecoderBlock(c2, c1, c1)

        self.out_conv = nn.Conv2d(c1, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.stem(x)
        x, skip1 = self.enc1(x)
        x, skip2 = self.enc2(x)
        x, skip3 = self.enc3(x)
        x, skip4 = self.enc4(x)
        x, skip5 = self.enc5(x)

        x = self.bottleneck(x)

        x = self.dec5(x, skip5)
        x = self.dec4(x, skip4)
        x = self.dec3(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)

        logits = self.out_conv(x)
        return logits


def build_npz(data_dir, csv_path, output_path, set_name):
    csv = pd.read_csv(csv_path)
    images, masks = [], []
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    for row in tqdm(csv.itertuples(), total=len(csv), desc=f"Building {set_name}.npz"):
        img_path = os.path.join(data_dir, row.ImageId)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        msk = cv2.imread(os.path.splitext(img_path)[0] + '.png', cv2.IMREAD_GRAYSCALE)
        if img is None or msk is None:
            warnings.warn(f"Missing file for {row.ImageId}, skipping")
            continue
        images.append(clahe.apply(img))
        masks.append(msk)
    if set_name == 'train':
        np.savez_compressed(output_path, x_train=np.array(images), y_train=np.array(masks))
    elif set_name == 'val':
        np.savez_compressed(output_path, x_val=np.array(images), y_val=np.array(masks))
    elif set_name == 'test':
        np.savez_compressed(output_path, x_test=np.array(images), y_test=np.array(masks))


def augment_batch_gpu(images, masks):
    B = images.shape[0]
    hflip = torch.rand(B, 1, 1, 1, device=images.device) < 0.5
    images = torch.where(hflip, torch.flip(images, dims=[3]), images)
    masks = torch.where(hflip, torch.flip(masks, dims=[3]), masks)
    vflip = torch.rand(B, 1, 1, 1, device=images.device) < 0.5
    images = torch.where(vflip, torch.flip(images, dims=[2]), images)
    masks = torch.where(vflip, torch.flip(masks, dims=[2]), masks)
    angles = (torch.rand(B, device=images.device) * 30) - 15
    cos_a = torch.cos(torch.deg2rad(angles))
    sin_a = torch.sin(torch.deg2rad(angles))
    cos_a = cos_a.unsqueeze(1)
    sin_a = sin_a.unsqueeze(1)
    theta = torch.cat([cos_a, -sin_a, torch.zeros_like(cos_a),
                       sin_a,  cos_a, torch.zeros_like(cos_a)], dim=1).view(-1, 2, 3)
    grid = F.affine_grid(theta, images.shape, align_corners=True)
    images = F.grid_sample(images, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    masks = F.grid_sample(masks, grid, mode='nearest', padding_mode='zeros', align_corners=True)
    images = images * 2 - 1
    return images, masks


class SurfaceDefectDatasetV2(Dataset):
    def __init__(self, data_path, set_name='train'):
        with np.load(data_path) as data:
            if set_name == 'train':
                img_arr, msk_arr = data['x_train'], data['y_train']
            elif set_name == 'val':
                img_arr, msk_arr = data['x_val'], data['y_val']
            else:
                img_arr, msk_arr = data['x_test'], data['y_test']
        self.images = torch.from_numpy(img_arr).float().div_(255.0)
        self.masks = (torch.from_numpy(msk_arr) > 0).float()

    def __getitem__(self, idx):
        return self.images[idx].unsqueeze(0), self.masks[idx].unsqueeze(0)

    def __len__(self):
        return len(self.images)


def pixel_accuracy(output, mask):
    pred = (torch.sigmoid(output) > 0.5).float()
    mask = (mask > 0.5).float()
    return (pred == mask).float().mean().item()

def iou_score(output, mask):
    pred = (torch.sigmoid(output) > 0.5).float()
    mask = (mask > 0.5).float()
    inter = (pred * mask).sum()
    union = pred.sum() + mask.sum() - inter
    if union == 0:
        return 1.0
    return (inter / union).item()


if __name__ == '__main__':
    task_name = 'NEU-seg'
    model_name = 'XU-Lite'
    session_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    train_dr = os.path.join(task_name, "TrainingData")
    val_dr   = os.path.join(task_name, "ValData")
    test_dr  = os.path.join(task_name, "TestData")

    train_csv_path = os.path.join(train_dr, "NEU_train.csv")
    val_csv_path   = os.path.join(val_dr, "NEU_val.csv")

    save_path          = os.path.join(task_name, model_name, session_name) + os.sep
    model_save_path    = os.path.join(save_path, 'models') + os.sep
    log_save_path      = os.path.join(save_path, 'logs') + os.sep
    result_save_path   = os.path.join(save_path, 'results') + os.sep

    build_npz(train_dr, train_csv_path, "train.npz", "train")
    build_npz(val_dr, val_csv_path, "val.npz", "val")

    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(log_save_path, exist_ok=True)
    os.makedirs(result_save_path, exist_ok=True)

    train_loader = DataLoader(SurfaceDefectDatasetV2("train.npz", 'train'), batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(SurfaceDefectDatasetV2("val.npz", 'val'), batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

    model = XULite(1, 1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    best_val_loss = float('inf')
    print(f"Training on: {device}\n" + "-"*60)

    NUM_EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        train_acc  = 0.0
        train_iou  = 0.0
        start_time = time.time()

        loop = tqdm(train_loader, desc=f'Epoch {epoch+1:02d}/{NUM_EPOCHS} [Train]')
        for images, masks in loop:
            images, masks = images.to(device), masks.to(device)
            images, masks = augment_batch_gpu(images, masks)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc  += pixel_accuracy(outputs, masks)
            train_iou  += iou_score(outputs, masks)
            loop.set_postfix(loss=loss.item(), acc=train_acc/(loop.n+1), iou=train_iou/(loop.n+1))

        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc  = train_acc  / len(train_loader)
        avg_train_iou  = train_iou  / len(train_loader)

        model.eval()
        val_loss = 0.0
        val_acc  = 0.0
        val_iou  = 0.0
        with torch.no_grad():
            loop = tqdm(val_loader, desc=f'Epoch {epoch+1:02d}/{NUM_EPOCHS} [Val]')
            for images, masks in loop:
                images, masks = images.to(device), masks.to(device)
                images = images * 2 - 1
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                val_acc  += pixel_accuracy(outputs, masks)
                val_iou  += iou_score(outputs, masks)
                loop.set_postfix(loss=loss.item(), acc=val_acc/(loop.n+1), iou=val_iou/(loop.n+1))

        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc  = val_acc  / len(val_loader)
        avg_val_iou  = val_iou  / len(val_loader)
        scheduler.step(avg_val_loss)

        elapsed = time.time() - start_time
        print(f'Epoch {epoch+1:02d}/{NUM_EPOCHS} | '
              f'Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.4f} | Train IoU: {avg_train_iou:.4f} | '
              f'Val Loss: {avg_val_loss:.4f} | Val Acc: {avg_val_acc:.4f} | Val IoU: {avg_val_iou:.4f} | '
              f'Time: {elapsed:.1f}s')

        log_path = os.path.join(log_save_path, 'training_log.csv')
        header = not os.path.exists(log_path)
        with open(log_path, 'a') as f:
            if header:
                f.write('epoch,train_loss,train_acc,train_iou,val_loss,val_acc,val_iou,time_s\n')
            f.write(f'{epoch+1},{avg_train_loss:.6f},{avg_train_acc:.6f},{avg_train_iou:.6f},'
                    f'{avg_val_loss:.6f},{avg_val_acc:.6f},{avg_val_iou:.6f},{elapsed:.2f}\n')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(model_save_path, 'best_xulite_model.pth'))
            print(f'--> Saved best model (val_loss: {best_val_loss:.4f})')

    print("Training complete!")

    build_npz(test_dr, os.path.join(test_dr, "NEU_test.csv"), "test.npz", "test")
    test_dataset = SurfaceDefectDatasetV2("test.npz", 'test')
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

    model.load_state_dict(torch.load(os.path.join(model_save_path, 'best_xulite_model.pth'), map_location=device))
    model.eval()
    test_loss = 0.0
    test_acc = 0.0
    test_iou = 0.0
    test_csv = pd.read_csv(os.path.join(test_dr, "NEU_test.csv"))
    pred_dir = os.path.join(result_save_path, 'predictions')
    vis_dir = os.path.join(result_save_path, 'comparisons')
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    with torch.no_grad():
        loop = tqdm(test_loader, desc='Test')
        idx = 0
        for images, masks in loop:
            images, masks = images.to(device), masks.to(device)
            images = images * 2 - 1
            outputs = model(images)
            loss = criterion(outputs, masks)
            test_loss += loss.item()
            test_acc += pixel_accuracy(outputs, masks)
            test_iou += iou_score(outputs, masks)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            for b in range(images.size(0)):
                raw_np = (images[b, 0].cpu().numpy() * 0.5 + 0.5) * 255
                raw_np = raw_np.astype(np.uint8)
                gt_np = (masks[b, 0].cpu().numpy() * 255).astype(np.uint8)
                pr_np = (preds[b, 0].cpu().numpy() * 255).astype(np.uint8)
                fname = os.path.splitext(test_csv.iloc[idx]['ImageId'])[0] + '.png'
                cv2.imwrite(os.path.join(pred_dir, fname), pr_np)
                trio = cv2.hconcat([raw_np, gt_np, pr_np])
                cv2.imwrite(os.path.join(vis_dir, fname), trio)
                idx += 1

    avg_test_loss = test_loss / len(test_loader)
    avg_test_acc = test_acc / len(test_loader)
    avg_test_iou = test_iou / len(test_loader)
    print(f'Test Loss: {avg_test_loss:.4f} | Test Acc: {avg_test_acc:.4f} | Test IoU: {avg_test_iou:.4f}')
    res_path = os.path.join(result_save_path, 'test_results.csv')
    with open(res_path, 'w') as f:
        f.write('test_loss,test_acc,test_iou\n')
        f.write(f'{avg_test_loss:.6f},{avg_test_acc:.6f},{avg_test_iou:.6f}\n')
    print(f'Results saved to {result_save_path}')
