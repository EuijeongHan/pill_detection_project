"""
detr/dataset.py
===============
DETR 학습/평가용 Dataset & DataLoader

DETRDataset은 COCO JSON을 읽어 DETR 포맷(cx, cy, w, h 정규화)으로 변환합니다.
detr_train.ipynb / detr_eval.ipynb / detr_tunning.ipynb 모두 이 모듈을 import합니다.

사용법:
    from src.models.detr.dataset import DETRDataset, get_detr_loaders

    train_loader, val_loader, idx2cat = get_detr_loaders(
        base_dir=BASE_DIR,
        target_size=800,   # 해상도 실험 시 1024 등으로 변경
        batch_size=4,
    )
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T


class DETRDataset(Dataset):
    """
    DETR 학습용 PyTorch Dataset.

    COCO JSON을 읽어 이미지당 (image_tensor, target) 쌍을 반환합니다.

    BBox 포맷 변환:
        COCO  : [x_min, y_min, w, h]  (픽셀 절댓값)
        DETR  : [cx, cy, w, h]        (0~1 정규화)

    레이블:
        원본 category_id → 0-based 연속 인덱스 (cat2idx)
        역매핑: idx2cat = {v: k for k, v in cat2idx.items()}

    Args:
        json_path   : letterbox 처리된 COCO JSON 경로
        img_dir     : letterbox 이미지 폴더 경로
        target_size : 이미지 해상도 (Letterbox 규격과 일치해야 함)
        transforms  : torchvision transforms (None이면 기본 ImageNet 정규화 적용)
    """

    def __init__(self, json_path, img_dir, target_size=800, transforms=None):
        with open(json_path, 'r') as f:
            coco = json.load(f)

        self.img_dir     = img_dir
        self.target_size = target_size

        self.images      = {img['id']: img for img in coco['images']}
        cats             = sorted([c['id'] for c in coco['categories']])
        self.cat2idx     = {c: i for i, c in enumerate(cats)}
        self.num_classes = len(cats)
        self.img_ids     = list(self.images.keys())

        self.annots = {img_id: [] for img_id in self.img_ids}
        for ann in coco['annotations']:
            if ann['image_id'] in self.annots:
                self.annots[ann['image_id']].append(ann)

        self.transforms = transforms or T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id   = self.img_ids[idx]
        img_info = self.images[img_id]

        image = Image.open(
            os.path.join(self.img_dir, img_info['file_name'])
        ).convert('RGB')
        W, H = image.size  # Letterbox 후엔 target_size x target_size

        boxes, labels = [], []
        for ann in self.annots[img_id]:
            x, y, w, h = ann['bbox']
            cx = (x + w / 2) / W
            cy = (y + h / 2) / H
            boxes.append([cx, cy, w / W, h / H])
            labels.append(self.cat2idx[ann['category_id']])

        target = {
            'boxes':    torch.tensor(boxes,  dtype=torch.float32),
            'labels':   torch.tensor(labels, dtype=torch.long),
            'image_id': torch.tensor([img_id]),
        }

        if self.transforms:
            image = self.transforms(image)

        return image, target


def collate_fn(batch):
    images, targets = zip(*batch)
    return torch.stack(images), list(targets)


def get_detr_loaders(base_dir, target_size=800, batch_size=4, num_workers=2):
    """
    train / val DataLoader와 idx2cat 역매핑을 반환합니다.

    Args:
        base_dir    : letterbox 산출물이 있는 데이터 루트
        target_size : Letterbox 해상도 (800 or 1024 등)
        batch_size  : 배치 크기 (고해상도일수록 줄여야 함)
        num_workers : DataLoader 워커 수

    Returns:
        train_loader, val_loader, idx2cat
    """
    suffix = f'_{target_size}' if target_size != 800 else ''

    train_json = os.path.join(base_dir, f'train_letterbox{suffix}.json')
    val_json   = os.path.join(base_dir, f'val_letterbox{suffix}.json')
    train_img  = os.path.join(base_dir, f'letterbox_images{suffix}', 'train')
    val_img    = os.path.join(base_dir, f'letterbox_images{suffix}', 'val')

    # 800px이면 기존 산출물 그대로 사용
    if target_size == 800:
        train_json = os.path.join(base_dir, 'train_letterbox.json')
        val_json   = os.path.join(base_dir, 'val_letterbox.json')
        train_img  = os.path.join(base_dir, 'letterbox_images', 'train')
        val_img    = os.path.join(base_dir, 'letterbox_images', 'val')

    train_ds = DETRDataset(train_json, train_img, target_size=target_size)
    val_ds   = DETRDataset(val_json,   val_img,   target_size=target_size)

    idx2cat = {v: k for k, v in train_ds.cat2idx.items()}

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, collate_fn=collate_fn)

    print(f'✅ target_size : {target_size}px')
    print(f'✅ train       : {len(train_ds)}장')
    print(f'✅ val         : {len(val_ds)}장')
    print(f'✅ num_classes : {train_ds.num_classes}')

    return train_loader, val_loader, idx2cat
