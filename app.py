from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI()

# Load the YOLO model
model = YOLO("yolov5su.pt")  # Update with your model path
classes = model.names

@app.post("/detect_objects/")
async def detect_objects(file: UploadFile = File(...)):
    try:
        # Read image file as bytes
        image = Image.open(io.BytesIO(await file.read()))

        # Perform object detection
        results = model(image)
        predictions = results[0].boxes.data  # Get bounding boxes and scores

        # Prepare the response with bounding boxes and class info
        response_data = []
        for box in predictions:
            x1, y1, x2, y2, conf, class_id = box
            response_data.append({
                "bbox": [x1.item(), y1.item(), x2.item(), y2.item()],
                "confidence": conf.item(),
                "class_id": class_id.item(),
                "class_name": classes[int(class_id)]  # Add the class name
            })

        return JSONResponse({"results": response_data})

    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})














# from ultralytics import YOLO
# from PIL import Image
# from fastapi import FastAPI, UploadFile, File
# from fastapi.responses import JSONResponse
# import io

# app = FastAPI()

# model = YOLO("Task7/yolov5su.pt")
# classes = model.names

# # Endpoint for object detection
# @app.post("/detect_objects/")
# async def detect_objects(file: UploadFile = File(...)):
#     if file is not None:
#         image = Image.open(io.BytesIO(await file.read()))
#         results = model(image)
#     # Extract results (assuming the first result contains detections)
#         predictions = results[0].boxes.data  # This will give us the bounding boxes and confidence scores

#         # Format the result to return useful data (bounding boxes, class IDs, and scores)
#         response_data = []
#         for box in predictions:
#             x1, y1, x2, y2, conf, class_id = box
#             response_data.append({
#                 "bbox": [x1.item(), y1.item(), x2.item(), y2.item()],
#                 "confidence": conf.item(),
#                 "class_id": class_id.item(),
#             })

#         # Return the results in a JSON response
#         return JSONResponse({"results": response_data})