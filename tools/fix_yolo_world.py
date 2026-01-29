from ultralytics import YOLO
import os

def fix_yolo_world():
    base_model_path = "models/yolov8s-worldv2.pt"
    target_model_path = "models/yolov8s-world-forklift.pt"
    classes = ["person", "forklift"]

    print(f"🔄 Loading base model: {base_model_path}")
    model = YOLO(base_model_path)

    print(f"🔧 Setting classes: {classes}")
    model.set_classes(classes)

    print(f"💾 Saving to: {target_model_path}")
    model.save(target_model_path)
    
    print("✅ Model saved with embedded vocabulary.")
    
    # Verification
    print("🔎 Verifying new model...")
    model_new = YOLO(target_model_path)
    print(f"   Classes: {model_new.names}")
    
    if len(model_new.names) == 2 and model_new.names[1] == 'forklift':
        print("✅ Verification PASSED")
    else:
        print("❌ Verification FAILED")

if __name__ == "__main__":
    if not os.path.exists("models/yolov8s-worldv2.pt"):
        print("❌ Base model models/yolov8s-worldv2.pt not found!")
    else:
        fix_yolo_world()
