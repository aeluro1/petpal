from ultralytics import YOLO

model = YOLO("yolo8vn.pt")
results = model.train(data = "oxford-iiit-pet.yaml", epochs = 3)
validation = model.val()
save = model.export(format = "onnx")