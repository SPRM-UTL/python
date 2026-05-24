import argparse
import json
from pathlib import Path

import torch
from torch import optim, save
from torch.utils.data import DataLoader

from aplicacion.boxes import stacker
from aplicacion.model import DETR
from entrenamiento.data import DETRData
from entrenamiento.loss import DETRLoss, HungarianMatcher


ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT_DIR / "aplicacion" / "config.json"
TRAIN_DIR = ROOT_DIR / "entrenamiento" / "data" / "train"
TEST_DIR = ROOT_DIR / "entrenamiento" / "data" / "test"
CHECKPOINTS_DIR = ROOT_DIR / "entrenamiento" / "checkpoints"
PRETRAINED_PATH = ROOT_DIR / "aplicacion" / "pretrained" / "4426_model.pt"


def load_classes() -> list[str]:
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)["classes"]


def parse_args():
    parser = argparse.ArgumentParser(description="Entrena el modelo DETR con tus senas.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--train-dir", type=Path, default=TRAIN_DIR)
    parser.add_argument("--test-dir", type=Path, default=TEST_DIR)
    parser.add_argument("--pretrained", type=Path, default=PRETRAINED_PATH)
    parser.add_argument("--checkpoints-dir", type=Path, default=CHECKPOINTS_DIR)
    return parser.parse_args()


def total_loss(loss_dict, weight_dict):
    return (
        loss_dict["labels"]["loss_ce"] * weight_dict["class_weighting"]
        + loss_dict["boxes"]["loss_bbox"] * weight_dict["bbox_weighting"]
        + loss_dict["boxes"]["loss_giou"] * weight_dict["giou_weighting"]
    )


def main():
    args = parse_args()
    classes = load_classes()
    num_classes = len(classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = DETRData(args.train_dir, train=True)
    test_dataset = DETRData(args.test_dir, train=False)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=stacker,
        drop_last=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        collate_fn=stacker,
        drop_last=True,
    )

    if len(train_dataloader) == 0:
        raise RuntimeError("No hay suficientes datos de entrenamiento para formar un batch.")
    if len(test_dataloader) == 0:
        raise RuntimeError("No hay suficientes datos de prueba para formar un batch.")

    model = DETR(num_classes=num_classes)
    model.load_pretrained(args.pretrained)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        len(train_dataloader) * 30,
        T_mult=2,
    )

    weights = {"class_weighting": 1, "bbox_weighting": 5, "giou_weighting": 2}
    matcher = HungarianMatcher(weights)
    criterion = DETRLoss(
        num_classes=num_classes,
        matcher=matcher,
        weight_dict=weights,
        eos_coef=0.1,
    )

    print(f"Clases: {', '.join(classes)}")
    print(f"Dispositivo: {device}")
    print(f"Train batches: {len(train_dataloader)} | Test batches: {len(test_dataloader)}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_epoch_loss = 0.0

        for images, targets in train_dataloader:
            images = images.to(device)
            targets = [
                {
                    "labels": target["labels"].to(device),
                    "boxes": target["boxes"].to(device),
                }
                for target in targets
            ]

            predictions = model(images)
            loss_dict = criterion(predictions, targets)
            loss = total_loss(loss_dict, weights)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_epoch_loss += loss.item()

        scheduler.step()

        model.eval()
        test_epoch_loss = 0.0
        with torch.no_grad():
            for images, targets in test_dataloader:
                images = images.to(device)
                targets = [
                    {
                        "labels": target["labels"].to(device),
                        "boxes": target["boxes"].to(device),
                    }
                    for target in targets
                ]

                predictions = model(images)
                loss_dict = criterion(predictions, targets)
                test_epoch_loss += total_loss(loss_dict, weights).item()

        train_loss = train_epoch_loss / len(train_dataloader)
        test_loss = test_epoch_loss / len(test_dataloader)
        print(f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.5f} | test_loss={test_loss:.5f}")

        if epoch % 10 == 0:
            checkpoint_path = args.checkpoints_dir / f"{epoch}_model.pt"
            save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint guardado: {checkpoint_path}")

    final_checkpoint = args.checkpoints_dir / f"{args.epochs}_model.pt"
    save(model.state_dict(), final_checkpoint)
    print(f"Entrenamiento terminado. Modelo guardado en: {final_checkpoint}")


if __name__ == "__main__":
    main()
