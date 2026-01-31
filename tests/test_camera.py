import time
import cv2

W, H, FPS = 1280, 800, 100

gst = (
    f"v4l2src device=/dev/video0 io-mode=2 ! "
    f"image/jpeg,width={W},height={H},framerate={FPS}/1 ! "
    f"jpegdec ! videoconvert ! video/x-raw,format=BGR ! "
    f"appsink drop=1 max-buffers=1 sync=false"
)

cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
assert cap.isOpened(), "Failed to open camera via GStreamer"

t0 = time.time()
n = 0
last = time.time()

while True:
    ok, frame = cap.read()
    if not ok:
        print("frame read failed")
        break
    n += 1
    now = time.time()
    if now - last >= 1.0:
        print("FPS:", n/(now-last))
        n = 0
        last = now

cap.release()
