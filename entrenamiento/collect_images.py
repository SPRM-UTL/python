import argparse
import json
import time
import uuid
from pathlib import Path

import cv2


ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT_DIR / "aplicacion" / "config.json"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "train" / "images"


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_camera_source(config: dict):
    camera_config = config.get("camera", {"type": "webcam", "source": 0})
    source = camera_config.get("source", 0)

    if camera_config.get("type") == "wifi":
        return source

    return int(source)


class CaptureImages:
    def __init__(self, output_dir: Path, classes: list[str], camera_source) -> None:
        self.output_dir = output_dir
        self.classes = classes
        self.cap = cv2.VideoCapture(camera_source)

        if not self.cap.isOpened():
            raise RuntimeError(f"No se pudo abrir la camara: {camera_source}")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Guardando imagenes en: {self.output_dir}")

    def capture(self, class_name: str, image_number: int) -> bool:
        ret, frame = self.cap.read()
        if not ret:
            print("No se pudo leer un frame de la camara.")
            return False

        raw_frame = frame.copy()
        preview = cv2.putText(
            frame,
            f"Capturando {class_name} #{image_number}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("Captura de imagenes", preview)

        filename = f"{class_name}-{uuid.uuid1()}.jpg"
        filepath = self.output_dir / filename
        cv2.imwrite(str(filepath), raw_frame)
        print(f"Imagen guardada: {filepath.name}")

        return not (cv2.waitKey(1) & 0xFF == ord("q"))

    def run(self, sleep_time: float, num_images: int) -> None:
        try:
            for class_name in self.classes:
                print(f"\nPreparando clase: {class_name}")
                time.sleep(2)

                for image_number in range(1, num_images + 1):
                    should_continue = self.capture(class_name, image_number)
                    if not should_continue:
                        print("Captura detenida por el usuario.")
                        return
                    time.sleep(sleep_time)
        finally:
            self.cap.release()
            cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Captura imagenes por clase para entrenar el detector."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Carpeta donde se guardaran las imagenes.",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=30,
        help="Cantidad de imagenes por clase.",
    )
    parser.add_argument(
        "--sleep-time",
        type=float,
        default=1,
        help="Segundos entre capturas.",
    )
    parser.add_argument(
        "--camera",
        default=None,
        help="Camara a usar. Si se omite, se toma de aplicacion/config.json.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = load_config()
    classes = config["classes"]
    camera_source = args.camera if args.camera is not None else get_camera_source(config)

    capture = CaptureImages(args.output, classes, camera_source)
    capture.run(sleep_time=args.sleep_time, num_images=args.num_images)
