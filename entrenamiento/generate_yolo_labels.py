import argparse
import json
import random
import shutil
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT_DIR / "aplicacion" / "config.json"
DATA_DIR = ROOT_DIR / "entrenamiento" / "data"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_classes() -> list[str]:
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)["classes"]


def class_from_filename(image_path: Path, classes: list[str]) -> str | None:
    image_name = image_path.stem.lower()
    matches = [class_name for class_name in classes if image_name.startswith(f"{class_name.lower()}-")]

    if len(matches) == 1:
        return matches[0]

    return None


def yolo_box(mode: str, box_size: float) -> tuple[float, float, float, float]:
    if mode == "full":
        return 0.5, 0.5, 1.0, 1.0

    return 0.5, 0.5, box_size, box_size


def image_files(images_dir: Path) -> list[Path]:
    if not images_dir.exists():
        return []

    return sorted(
        image_path
        for image_path in images_dir.iterdir()
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS
    )


def write_labels(split_dir: Path, classes: list[str], mode: str, box_size: float, overwrite: bool) -> int:
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    created = 0
    for image_path in image_files(images_dir):
        class_name = class_from_filename(image_path, classes)
        if class_name is None:
            print(f"Saltando imagen sin clase reconocible: {image_path.name}")
            continue

        label_path = labels_dir / f"{image_path.stem}.txt"
        if label_path.exists() and not overwrite:
            continue

        class_index = classes.index(class_name)
        x_center, y_center, width, height = yolo_box(mode, box_size)

        label_path.write_text(
            f"{class_index} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n",
            encoding="utf-8",
        )
        created += 1

    return created


def split_train_to_test(data_dir: Path, test_ratio: float, seed: int) -> int:
    if test_ratio <= 0:
        return 0

    train_images_dir = data_dir / "train" / "images"
    test_images_dir = data_dir / "test" / "images"
    test_images_dir.mkdir(parents=True, exist_ok=True)

    images = image_files(train_images_dir)
    if not images:
        return 0

    random.seed(seed)
    random.shuffle(images)
    test_count = max(1, int(len(images) * test_ratio))
    selected_images = images[:test_count]

    moved = 0
    for image_path in selected_images:
        destination = test_images_dir / image_path.name
        if destination.exists():
            continue

        shutil.move(str(image_path), str(destination))
        moved += 1

    return moved


def parse_args():
    parser = argparse.ArgumentParser(
        description="Genera etiquetas YOLO automaticamente desde nombres de imagenes."
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--mode",
        choices=["full", "center"],
        default="full",
        help="full usa toda la imagen; center usa una caja centrada.",
    )
    parser.add_argument(
        "--box-size",
        type=float,
        default=0.75,
        help="Tamano de la caja centrada en modo center, normalizado de 0 a 1.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0,
        help="Mueve una parte de train/images a test/images antes de generar etiquetas.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    classes = load_classes()

    moved = split_train_to_test(args.data_dir, args.test_ratio, args.seed)
    train_created = write_labels(args.data_dir / "train", classes, args.mode, args.box_size, args.overwrite)
    test_created = write_labels(args.data_dir / "test", classes, args.mode, args.box_size, args.overwrite)

    print(f"Imagenes movidas a test: {moved}")
    print(f"Etiquetas train generadas: {train_created}")
    print(f"Etiquetas test generadas: {test_created}")


if __name__ == "__main__":
    main()
