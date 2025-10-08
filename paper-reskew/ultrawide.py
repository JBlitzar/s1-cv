import os
import signal
import subprocess

# Start ffmpeg as a subprocess so we can get its PID and terminate it later
ffmpeg_proc = subprocess.Popen([
    "ffmpeg", "-f", "avfoundation", "-framerate", "30", "-i", "1",
    "-f", "mpegts", "udp://127.0.0.1:23000"
])

try:
    import cv2
    cap = cv2.VideoCapture("udp://127.0.0.1:23000")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
except KeyboardInterrupt:
    pass
finally:
    ffmpeg_proc.terminate()
    ffmpeg_proc.wait()
