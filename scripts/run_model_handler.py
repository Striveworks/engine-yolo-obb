import argparse

import torch
from PIL import Image, ImageDraw
from ultralytics.utils import ops

from engine_yolo_obb.model_handler import ModelHandler, prepare_data

parser = argparse.ArgumentParser()
parser.add_argument(
    "--image",
    required=True,
    type=str,
    help="path to an image file (e.g., png) stored locally to run inference on",
)
parser.add_argument(
    "--model-file",
    required=False,
    type=str,
    help="path to a model file",
    default="test-data/yolo11n-obb.pt",
)
parser.add_argument(
    "--score-threshold",
    type=float,
    help="score threshold to use",
    default=0.5,
)
args = parser.parse_args()

torch.manual_seed(42 * 42)
handler = ModelHandler(args.model_file)

data = prepare_data(args.image, 1, [args.score_threshold])

results = handler.handle(data)
result = results[0]  # Batch size of 1
print(results)
img = Image.open(args.image)

draw = ImageDraw.ImageDraw(img)
assert type(result) is dict
with torch.inference_mode():
    for box, cls in zip(result["oriented_detection_boxes"], result["detection_classes"]):
        box_corners = (
            ops.xywhr2xyxyxyxy(
                torch.tensor(
                    [
                        box["cx"] * img.width,
                        box["cy"] * img.height,
                        box["w"] * img.width,
                        box["h"] * img.height,
                        box["r"],
                    ]
                )
            )
            .reshape(8)
            .tolist()
        )
        draw.polygon(box_corners, outline="red", width=5)
        top_left = box_corners[4:6]
        draw.text((top_left[0] + 10, top_left[1] + 10), cls, fill="red")
img.show()
