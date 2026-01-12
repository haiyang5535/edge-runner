#!/usr/bin/env python3
"""
Standalone Camera Check Tool for EdgeRunner
Usage: python3 camera_check.py
View at: http://<JETSON_IP>:8090
"""
import cv2
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

HOST_IP = '0.0.0.0'
PORT = 8090

class GlobalFrame:
    frame = None
    lock = threading.Lock()

cam_frame = GlobalFrame()

class MJPEGHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=--jpgboundary')
        self.end_headers()
        while True:
            with cam_frame.lock:
                if cam_frame.frame is None:
                    time.sleep(0.01)
                    continue
                # Reduce quality for smooth preview, focus check only
                ret, jpeg = cv2.imencode('.jpg', cam_frame.frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
            
            if not ret: continue
            try:
                self.wfile.write(b'--jpgboundary\r\n')
                self.send_header('Content-type', 'image/jpeg')
                self.send_header('Content-length', str(len(jpeg)))
                self.end_headers()
                self.wfile.write(jpeg.tobytes())
                self.wfile.write(b'\r\n')
                time.sleep(0.03) # Limit stream rate to prevent browser lag
            except BrokenPipeError:
                break

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""

def start_camera():
    # V4L2 direct mode (more stable than GStreamer)
    cap = cv2.VideoCapture('/dev/video0', cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # 720p for focus check
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    if not cap.isOpened():
        print("❌ Error: Cannot open camera")
        return

    print(f"✅ Camera Started. View at http://<JETSON_IP>:{PORT}")
    print("👉 Press Ctrl+C to stop and free the camera.")

    try:
        server = ThreadedHTTPServer((HOST_IP, PORT), MJPEGHandler)
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()

        while True:
            ret, frame = cap.read()
            if not ret: continue
            with cam_frame.lock:
                cam_frame.frame = frame
    except KeyboardInterrupt:
        print("\n🛑 Stopping...")
    finally:
        cap.release()
        server.shutdown()

if __name__ == '__main__':
    start_camera()