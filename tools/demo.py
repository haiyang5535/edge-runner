import time, json
import cv2

# Camera stabilization parameters
W, H, FPS = 1280, 800, 100

gst = (
    f"v4l2src device=/dev/video0 io-mode=2 ! "
    f"image/jpeg,width={W},height={H},framerate={FPS}/1 ! "
    f"jpegdec ! videoconvert ! video/x-raw,format=BGR ! "
    f"appsink drop=1 max-buffers=1 sync=false"
)

cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
assert cap.isOpened(), "Failed to open camera"

# Placeholder for VLM inference
# Goal: feed VLM one frame every 0.5s, output structured JSON
VLM_INTERVAL = 0.5
last_vlm = 0.0

def vlm_infer_stub(frame_bgr):
    # Stub: return fixed JSON to validate pipeline rhythm
    return {"target": {"bbox": [0,0,0,0], "description": "stub"},
            "confidence": 0.0, "action": "follow"}

# Frame counter
n = 0
t0 = time.time()

while True:
    ok, frame = cap.read()
    if not ok:
        print("read failed")
        break

    n += 1
    now = time.time()

    # Report capture FPS every 1s
    if now - t0 >= 1.0:
        print(f"CAM FPS: {n/(now-t0):.2f}")
        n = 0
        t0 = now

    # Low-frequency VLM inference (2fps)
    if now - last_vlm >= VLM_INTERVAL:
        last_vlm = now
        out = vlm_infer_stub(frame)
        print("VLM:", json.dumps(out, ensure_ascii=False))

cap.release()
