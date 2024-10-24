import cv2
import numpy as np
import supervision as sv
import os
import json

import torch
import torchvision

from groundingdino.util.inference import Model
from segment_anything import SamPredictor
from LightHQSAM.setup_light_hqsam import setup_model

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

par_path = os.path.abspath(os.path.dirname(os.getcwd()))
root_path = os.path.abspath(os.path.dirname(par_path))

# GroundingDINO config and checkpoint
GROUNDING_DINO_CONFIG_PATH = os.path.join(par_path, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(par_path, "groundingdino_swint_ogc.pth")

# Building GroundingDINO inference model
grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)

# Building MobileSAM predictor
HQSAM_CHECKPOINT_PATH = "./sam_hq_vit_tiny.pth"
checkpoint = torch.load(HQSAM_CHECKPOINT_PATH)
light_hqsam = setup_model()
light_hqsam.load_state_dict(checkpoint, strict=True)
light_hqsam.to(device=DEVICE)

sam_predictor = SamPredictor(light_hqsam)

record_timestamps_path = os.path.join(root_path, "grasp_detection/out", "record_timestamps.txt")
# read timestamps
with open(record_timestamps_path, "r") as file:
    timestamps = file.read().splitlines()

# get latest timestamp
if timestamps:
    latest_timestamp = timestamps[-1]
else:
    print("No timestamps found in the file.")

OUT_PATH = os.path.join(root_path, "grasp_detection/out", latest_timestamp)
SOURCE_IMAGE_PATH = os.path.join(OUT_PATH, "color.png")
RESULT_PATH = os.path.join(OUT_PATH, "grounded_light_hqsam")
os.mkdir(RESULT_PATH)

# Predict classes and hyper-param for GroundingDINO
CLASSES = [""]
user_input = input("Please enter the object you want: ")
CLASSES[0] = user_input

BOX_THRESHOLD = 0.4
TEXT_THRESHOLD = 0.25
NMS_THRESHOLD = 0.8


# load image
image = cv2.imread(SOURCE_IMAGE_PATH)
print(SOURCE_IMAGE_PATH)

# detect objects
detections = grounding_dino_model.predict_with_classes(
    image=image,
    classes=CLASSES,
    box_threshold=BOX_THRESHOLD,
    text_threshold=TEXT_THRESHOLD
)

# annotate image with detections
box_annotator = sv.BoxAnnotator()
labels = [
    f"{CLASSES[class_id]} {confidence:0.2f}" 
    for _, _, confidence, class_id, _, _
    in detections]
annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)

# save the xyxy of bounding box
print(f"bounding box num: {detections.xyxy}")
bbox_data = {
    "label": CLASSES[0],
    "bounding_boxes": detections.xyxy.tolist()  # convert to python list
}

json_file_path = os.path.join(OUT_PATH, "bounding_box.json")

# dic to json
with open(json_file_path, "w") as json_file:
    json.dump(bbox_data, json_file, indent=4)

# save the annotated grounding dino image
cv2.imwrite(os.path.join((RESULT_PATH), "groundingdino_annotated_image.jpg"), annotated_frame)


# NMS post process
print(f"Before NMS: {len(detections.xyxy)} boxes")
nms_idx = torchvision.ops.nms(
    torch.from_numpy(detections.xyxy), 
    torch.from_numpy(detections.confidence), 
    NMS_THRESHOLD
).numpy().tolist()

detections.xyxy = detections.xyxy[nms_idx]
detections.confidence = detections.confidence[nms_idx]
detections.class_id = detections.class_id[nms_idx]

print(f"After NMS: {len(detections.xyxy)} boxes")

# Prompting SAM with detected boxes
def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=False,
            hq_token_only=True,
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)


# convert detections to masks
detections.mask = segment(
    sam_predictor=sam_predictor,
    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
    xyxy=detections.xyxy
)

# annotate image with detections
box_annotator = sv.BoxAnnotator()
mask_annotator = sv.MaskAnnotator()
labels = [
    f"{CLASSES[class_id]} {confidence:0.2f}" 
    for _, _, confidence, class_id, _, _
    in detections]
annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

# save the annotated grounded-sam image
grounded_light_hqsam_mask = detections.mask.astype(np.uint8)*255
# print(grounded_light_hqsam_mask.shape)
mask_data = np.array(grounded_light_hqsam_mask, dtype=np.uint8)
mask_data = mask_data.squeeze()
cv2_mask = cv2.cvtColor(mask_data, cv2.COLOR_GRAY2BGR)
cv2.imwrite(os.path.join((RESULT_PATH), "mask.jpg"), cv2_mask)
cv2.imwrite(os.path.join((RESULT_PATH), "grounded_light_hqsam_annotated_image.jpg"), annotated_image)
