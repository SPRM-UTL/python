from pathlib import Path

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset


class DETRData(Dataset):
    def __init__(self, path: str | Path, train: bool = True):
        self.path = Path(path)
        self.images_path = self.path / "images"
        self.labels_path = self.path / "labels"
        self.train = train

        if not self.images_path.exists():
            raise FileNotFoundError(f"No existe la carpeta de imagenes: {self.images_path}")
        if not self.labels_path.exists():
            raise FileNotFoundError(f"No existe la carpeta de etiquetas: {self.labels_path}")

        self.labels = sorted(self.labels_path.glob("*.txt"))
        if not self.labels:
            raise FileNotFoundError(f"No hay etiquetas .txt en: {self.labels_path}")

        self.transform = A.Compose(
            [
                A.Resize(500, 500),
                *([A.RandomCrop(width=224, height=224, p=0.33)] if train else []),
                A.Resize(224, 224),
                *([A.HorizontalFlip(p=0.5)] if train else []),
                *(
                    [
                        A.ColorJitter(
                            brightness=0.5,
                            contrast=0.5,
                            saturation=0.5,
                            hue=0.5,
                            p=0.5,
                        )
                    ]
                    if train
                    else []
                ),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label_path = self.labels[idx]
        image_path = self.images_path / f"{label_path.stem}.jpg"

        if not image_path.exists():
            raise FileNotFoundError(f"No existe la imagen para {label_path.name}: {image_path}")

        image = np.array(Image.open(image_path).convert("RGB"))
        class_labels, bounding_boxes = self.read_label_file(label_path)

        augmented = self.safe_transform(image, bounding_boxes, class_labels)

        labels = torch.tensor(augmented["class_labels"], dtype=torch.long)
        boxes = torch.tensor(augmented["bboxes"], dtype=torch.float32)

        return augmented["image"], {"labels": labels, "boxes": boxes}

    def read_label_file(self, label_path: Path):
        class_labels = []
        bounding_boxes = []

        with label_path.open("r", encoding="utf-8") as file:
            for line in file:
                values = line.strip().split()
                if not values:
                    continue

                class_labels.append(int(values[0]))
                bounding_boxes.append([float(value) for value in values[1:5]])

        return class_labels, bounding_boxes

    def safe_transform(self, image, bboxes, labels, max_attempts: int = 50):
        for _ in range(max_attempts):
            transformed = self.transform(image=image, bboxes=bboxes, class_labels=labels)
            if transformed["bboxes"]:
                return transformed

        fallback = A.Compose(
            [
                A.Resize(224, 224),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
        )
        return fallback(image=image, bboxes=bboxes, class_labels=labels)
