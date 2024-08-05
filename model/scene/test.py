from ultralytics import YOLO

model = YOLO(r'C:\Users\leishen\Desktop\Workspaces\15S\system\model\scene\model\sence.pt')

result = model.predict(r'C:\Users\leishen\Desktop\Workspaces\15S\system\model\scene\source\fire.png', show=True , save=False)

pass