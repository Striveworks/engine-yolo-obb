import base64
import io
import json
import logging
import tempfile
import zipfile
from itertools import groupby
from operator import itemgetter
from pathlib import Path
from typing import TypeAlias

import torch
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
from ultralytics.engine.results import Results
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import ops

JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None
RawInput = list[dict[str, "JSON"]]
PreprocessedInput: TypeAlias = list[tuple[torch.FloatTensor | Image.Image, float]]

DEFAULT_SCORE_THRESHOLD = 0.5

logger = logging.getLogger(__name__)


def load_model_mar(model_mar_path: Path):
    temp_model_pt_file = tempfile.NamedTemporaryFile("wb", suffix=".pt", delete=False)
    with zipfile.ZipFile(model_mar_path, "r") as zfp:
        model_config = {}
        if "model_config.json" in zfp.namelist():
            model_config = json.loads(zfp.read("model_config.json"))

        manifest = json.loads(zfp.read("MAR-INF/MANIFEST.json"))
        with zfp.open(manifest["model"]["serializedFile"]) as zip_model_pt_file:
            temp_model_pt_file.write(zip_model_pt_file.read())
    return model_config, Path(temp_model_pt_file.name)


class ModelHandler:
    model: YOLO
    threshold = 0.5
    image_processing: transforms.Compose = transforms.Compose([transforms.ToTensor()])

    def __init__(self, model_path: str | Path):
        logger.info("Loading model...")
        self._load_model(Path(model_path))
        logger.info("Done loading model.")

    def _load_model(self, model_path: Path):
        assert model_path.exists(), f"model_path {model_path!r} does not exist"
        if model_path.is_dir():
            model_mar_path = model_path / "model-store" / "model.mar"
            model_config, model_pt_path = load_model_mar(model_mar_path)
        else:
            # TODO: load `idx_to_class` and `resize_to` from some kind of config file
            model_config = {}
            model_pt_path = model_path

        if torch.cuda.is_available():
            logger.info(f"CUDA is available. Using device {torch.cuda.current_device()}.")
        else:
            logger.info("CUDA is not available. Using CPU.")
        self.device = torch.device(
            "cuda:" + str(torch.cuda.current_device()) if torch.cuda.is_available() else "cpu"
        )
        self.model = YOLO(model_pt_path, task="obb").to(self.device)

        if model_path.is_dir():
            model_pt_path.unlink()  # cleanup the temporary file

        self.predictor = self._get_predictor(max_nms=2048)

        idx_to_class = model_config.get("idx_to_class")
        if idx_to_class:
            assert isinstance(self.model.model, DetectionModel)
            self.model.model.names = {int(id): label for id, label in idx_to_class.items()}

        resize_to = model_config.get("resize_to")
        if resize_to:
            self.resize_to = resize_to
        else:
            self.resize_to = 640

    def _get_predictor(self, max_nms: int = 30000):
        predictor_cls = self.model._smart_load("predictor")

        def postprocess(self, preds, img, orig_imgs, **kwargs):
            """Post-processes predictions and returns a list of Results objects."""
            preds = ops.non_max_suppression(
                preds,
                self.args.conf,
                self.args.iou,
                self.args.classes,
                self.args.agnostic_nms,
                max_det=self.args.max_det,
                max_nms=max_nms,
                nc=len(self.model.names),
                end2end=getattr(self.model, "end2end", False),
                rotated=self.args.task == "obb",
            )

            if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
                orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

            return self.construct_results(preds, img, orig_imgs, **kwargs)

        predictor_cls.postprocess = postprocess
        return predictor_cls

    def preprocess(self, data: RawInput) -> PreprocessedInput:
        image_threshold_pairs = []
        for row in data:
            image = row.get("data") or row.get("body")
            if isinstance(image, list) and len(image) == 1:
                image = image[0]
            if isinstance(image, str):
                image = base64.b64decode(image)

            if isinstance(image, (bytearray, bytes)):
                image = Image.open(io.BytesIO(image))
            else:
                image = torch.FloatTensor(image)

            parameters = row.get("parameters", {})
            if type(parameters) is not dict:
                raise TypeError("expected 'parameters' to be a JSON object")

            raw_score_threshold = parameters.get("score_threshold", DEFAULT_SCORE_THRESHOLD)
            if type(raw_score_threshold) is not float and type(raw_score_threshold) is not int:
                raise TypeError("expected 'parameters.score_threshold' to be a number")

            score_threshold = float(raw_score_threshold)

            image_threshold_pairs.append((image, score_threshold))

        return image_threshold_pairs

    def inference(self, data: PreprocessedInput, *args, **kwargs):
        # Group consecutive images that have the same `score_threshold` together.  It may
        # often be the case that all score thresholds are the same, in which case this is
        # equivalent to just returning the value of `self.model.predict(...)`.
        return [
            model_output
            for score_threshold, image_threshold_pairs in groupby(data, key=itemgetter(1))
            for model_output in self.model.predict(
                source=[model_inputs for model_inputs, _ in image_threshold_pairs],
                *args,
                **{**kwargs, **{"conf": score_threshold}},
                predictor=self.predictor,
                imgsz=self.resize_to,
                save_conf=True,
                device=self.device,
            )
        ]

    def postprocess(self, data: list[Results]) -> list[dict]:
        return convert_yolo_results(self.device, data)

    def handle(self, data: RawInput):
        preprocess_result = self.preprocess(data)
        inference_result = self.inference(preprocess_result)
        return self.postprocess(inference_result)


def prepare_data(local_data_file: str, batch_size: int, score_thresholds: list[float]):
    """
    Function to prepare data based on the desired batch size
    """
    f = open(local_data_file, "rb", buffering=0)
    read_data = f.read()
    data = []
    for i in range(batch_size):
        tmp = {}
        tmp["data"] = read_data
        tmp["parameters"] = {"score_threshold": score_thresholds[i]}
        data.append(tmp)
    return data


def convert_yolo_results(device, data: list[Results]) -> list[dict]:
    results_json = []
    for result in data:
        result_json = {
            "detection_classes": [],
            "detection_scores": [],
            "oriented_detection_boxes": [],
        }
        if not result.obb:
            results_json.append(result_json)
            continue
        for box in result.obb:
            try:
                score = box.data[0][5].item()
            except IndexError:
                logger.warning(f"cannot find detection score in data: {box.data}")
                score = 0.0

            xywhr = box.xywhr[0].to(device)
            orig_height = box.orig_shape[0]
            orig_width = box.orig_shape[1]
            xywhr_scaled = xywhr / torch.tensor(
                [orig_width, orig_height, orig_width, orig_height, 1],
                device=device,
            )
            result_json["oriented_detection_boxes"].append(
                {
                    "cx": float(xywhr_scaled[0].item()),
                    "cy": float(xywhr_scaled[1].item()),
                    "w": float(xywhr_scaled[2].item()),
                    "h": float(xywhr_scaled[3].item()),
                    "r": float(xywhr_scaled[4].item()),
                }
            )
            result_json["detection_classes"].append(result.names[int(box.cls.item())])
            result_json["detection_scores"].append(score)

        results_json.append(result_json)

    return results_json
