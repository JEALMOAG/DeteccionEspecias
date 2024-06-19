from ultralytics import YOLO
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch


# Use the model
model.train(data="config.yaml", epochs=100)  # train the model
