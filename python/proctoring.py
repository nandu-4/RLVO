"""
RLVO - Proctoring (Python reference implementation)
---------------------------------------------------
Mirrors the browser useProctoring hook using MediaPipe Face Mesh + Ultralytics
YOLOv8 (equivalent to COCO-SSD in the browser version).

Detections:
    - Head yaw (left/right turn)               from face landmarks
    - Gaze direction (iris offset)             from iris landmarks (468, 473)
    - Looking down / phone use (faceAR drop)   from face geometry
    - Multiple faces in frame                  from face mesh count
    - Phone in frame                           from YOLOv8 "cell phone" class
    - No face detected                         from absence of landmarks

Trust score decay:
    high   -> -6
    medium -> -3
    low    -> -1
    info   ->  0

Usage:
    python proctoring.py                 # webcam mode
    python proctoring.py video.mp4       # offline mode on a recorded video

Press 'q' to quit. Exports session report to proctoring_report.json on exit.

Deps:
    pip install mediapipe opencv-python ultralytics numpy
"""

import sys
import json
import time
from collections import defaultdict
from datetime import datetime

import cv2
import numpy as np
import mediapipe as mp

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[WARN] ultralytics not installed - phone detection disabled")


# ---------------------------------------------------------------------------
# Tunables (must mirror useProctoring.ts)
# ---------------------------------------------------------------------------
CALIB_FRAMES = 90              # ~3 sec at 30 fps
YAW_THRESHOLD = 0.33           # |yaw| > 0.33 triggers head turn
LOOK_DOWN_AR_THRESHOLD = 0.10  # 10% drop in faceAR
LOOK_DOWN_SUSTAINED = 20       # frames (~0.7 s)
GAZE_DELTA = 0.07              # 7% of eye width
GAZE_SUSTAINED = 10            # frames
PHONE_CONF = 0.6               # YOLO confidence
PHONE_EVERY_N = 10             # run YOLO every 10th frame

COOLDOWN = {
    "head_turn_left":  2500,
    "head_turn_right": 2500,
    "gaze_away":       2000,
    "no_face":         3000,
    "multiple_faces":  3000,
    "looking_down":    3000,
    "phone_detected":  5000,
    "tab_switch":      1000,
}

SEVERITY_DECAY = {"high": 6, "medium": 3, "low": 1, "info": 0}


# ---------------------------------------------------------------------------
# Landmark indices (MediaPipe Face Mesh - 468 landmarks + iris)
# ---------------------------------------------------------------------------
NOSE_TIP = 1
LEFT_EAR = 234
RIGHT_EAR = 454
CHIN = 152
FOREHEAD = 10
LEFT_EYE_INNER = 133
LEFT_EYE_OUTER = 33
RIGHT_EYE_INNER = 362
RIGHT_EYE_OUTER = 263
LEFT_IRIS = 468   # requires refine_landmarks=True
RIGHT_IRIS = 473


# ---------------------------------------------------------------------------
# Proctoring session
# ---------------------------------------------------------------------------
class ProctoringSession:
    def __init__(self):
        self.mp_face = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=2,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.yolo = YOLO("yolov8n.pt") if YOLO_AVAILABLE else None

        self.start_time = time.time()
        self.trust_score = 100
        self.alerts = []
        self.stats = defaultdict(int)
        self.last_alert_ms = {}

        # baselines
        self.calib_frames_seen = 0
        self.calib_face_ar_sum = 0.0
        self.calib_gaze_sum = 0.0
        self.face_ar_baseline = 0.0
        self.gaze_baseline = 0.0

        # sustained counters
        self.look_down_frames = 0
        self.gaze_frames = 0
        self.phone_frame_count = 0

    # ----- alert plumbing -------------------------------------------------
    def push_alert(self, kind: str, message: str, severity: str):
        now_ms = int(time.time() * 1000)
        last = self.last_alert_ms.get(kind, 0)
        if now_ms - last < COOLDOWN.get(kind, 2000):
            return
        self.last_alert_ms[kind] = now_ms

        alert = {
            "id": f"{now_ms}_{kind}",
            "time": datetime.now().strftime("%H:%M:%S"),
            "timestamp": now_ms,
            "type": kind,
            "message": message,
            "severity": severity,
        }
        self.alerts.append(alert)
        self.trust_score = max(0, self.trust_score - SEVERITY_DECAY[severity])
        self.stats[kind] += 1
        print(f"[{alert['time']}] [{severity.upper():6}] {kind}: {message}  (trust={self.trust_score})")

    # ----- per-frame analysis --------------------------------------------
    def analyze(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.mp_face.process(rgb)

        face_detected = bool(res.multi_face_landmarks)
        multiple_faces = face_detected and len(res.multi_face_landmarks) > 1

        if not face_detected:
            self.push_alert("no_face", "No face detected in frame", "medium")
            return

        if multiple_faces:
            self.push_alert("multiple_faces",
                            f"{len(res.multi_face_landmarks)} faces detected",
                            "high")

        landmarks = res.multi_face_landmarks[0].landmark

        def pt(idx):
            return np.array([landmarks[idx].x, landmarks[idx].y])

        nose = pt(NOSE_TIP)
        l_ear, r_ear = pt(LEFT_EAR), pt(RIGHT_EAR)
        chin = pt(CHIN)
        forehead = pt(FOREHEAD)

        # ----- Head yaw ----------------------------------------------------
        ear_mid_x = (l_ear[0] + r_ear[0]) / 2
        ear_span = abs(r_ear[0] - l_ear[0])
        yaw = (nose[0] - ear_mid_x) / (ear_span / 2) if ear_span > 0.01 else 0.0
        if yaw > YAW_THRESHOLD:
            self.push_alert("head_turn_right",
                            f"Head turned right ({int(abs(yaw)*100)}% deviation)",
                            "medium")
        elif yaw < -YAW_THRESHOLD:
            self.push_alert("head_turn_left",
                            f"Head turned left ({int(abs(yaw)*100)}% deviation)",
                            "medium")

        # ----- Face Aspect Ratio (looking down / phone tilt) ---------------
        face_h = abs(chin[1] - forehead[1])
        face_width = ear_span
        face_ar = face_h / face_width if face_width > 0.01 else 0.0

        # ----- Iris gaze ---------------------------------------------------
        l_iris = pt(LEFT_IRIS)
        r_iris = pt(RIGHT_IRIS)
        l_in, l_out = pt(LEFT_EYE_INNER), pt(LEFT_EYE_OUTER)
        r_in, r_out = pt(RIGHT_EYE_INNER), pt(RIGHT_EYE_OUTER)

        l_eye_w = abs(l_out[0] - l_in[0]) or 0.01
        r_eye_w = abs(r_out[0] - r_in[0]) or 0.01
        l_eye_mid = (l_in[0] + l_out[0]) / 2
        r_eye_mid = (r_in[0] + r_out[0]) / 2

        l_gaze = (l_iris[0] - l_eye_mid) / l_eye_w
        r_gaze = (r_iris[0] - r_eye_mid) / r_eye_w
        gaze_offset = (l_gaze + r_gaze) / 2

        # ----- Calibration window -----------------------------------------
        if self.calib_frames_seen < CALIB_FRAMES:
            self.calib_face_ar_sum += face_ar
            self.calib_gaze_sum += gaze_offset
            self.calib_frames_seen += 1
            if self.calib_frames_seen == CALIB_FRAMES:
                self.face_ar_baseline = self.calib_face_ar_sum / CALIB_FRAMES
                self.gaze_baseline = self.calib_gaze_sum / CALIB_FRAMES
                print(f"[calibration done] faceAR_baseline={self.face_ar_baseline:.3f} "
                      f"gaze_baseline={self.gaze_baseline:.3f}")
            return

        # ----- Looking down via faceAR compression ------------------------
        face_ar_delta = ((face_ar - self.face_ar_baseline) / self.face_ar_baseline
                        if self.face_ar_baseline > 0.01 else 0.0)

        if face_ar_delta < -LOOK_DOWN_AR_THRESHOLD and face_width > 0.01:
            self.look_down_frames += 1
            if self.look_down_frames == LOOK_DOWN_SUSTAINED:
                self.push_alert("looking_down",
                                f"Sustained downward gaze (face {int(abs(face_ar_delta)*100)}% compressed)",
                                "high")
        else:
            self.look_down_frames = 0

        # ----- Sustained off-screen gaze ----------------------------------
        gaze_delta = gaze_offset - self.gaze_baseline
        if abs(gaze_delta) > GAZE_DELTA:
            self.gaze_frames += 1
            direction = "left" if gaze_delta > 0 else "right"
            if self.gaze_frames == GAZE_SUSTAINED:
                self.push_alert("gaze_away",
                                f"Gaze directed off-screen (looking {direction})",
                                "medium")
        else:
            self.gaze_frames = 0

        # ----- Phone in frame (YOLO, every 10th frame) --------------------
        self.phone_frame_count += 1
        if self.yolo and self.phone_frame_count % PHONE_EVERY_N == 0:
            results = self.yolo.predict(frame_bgr, verbose=False, conf=PHONE_CONF)
            for r in results:
                for box in r.boxes:
                    cls_name = self.yolo.names[int(box.cls[0])]
                    conf = float(box.conf[0])
                    if cls_name == "cell phone" and conf > PHONE_CONF:
                        self.push_alert("phone_detected",
                                        f"Mobile phone detected ({int(conf*100)}% confidence)",
                                        "high")
                        break

    # ----- Export ---------------------------------------------------------
    def export(self, path: str = "proctoring_report.json"):
        duration = int(time.time() - self.start_time)
        report = {
            "exportedAt": datetime.utcnow().isoformat() + "Z",
            "sessionStart": datetime.fromtimestamp(self.start_time).isoformat() + "Z",
            "durationSeconds": duration,
            "trustScore": self.trust_score,
            "stats": dict(self.stats),
            "alerts": self.alerts,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nExported {len(self.alerts)} alerts to {path}")
        print(f"Final trust score: {self.trust_score}/100")


# ---------------------------------------------------------------------------
# CLI loop
# ---------------------------------------------------------------------------
def run(source=0):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Cannot open source: {source}")
        return

    session = ProctoringSession()
    print("Calibrating - please look straight at the camera for ~3 seconds...")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        session.analyze(frame)

        # overlay trust score + calib progress
        status = (f"Trust: {session.trust_score}/100  "
                  f"Calib: {min(session.calib_frames_seen, CALIB_FRAMES)}/{CALIB_FRAMES}")
        cv2.putText(frame, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("RLVO Proctoring (Python)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    session.export()


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else 0
    try:
        src = int(src)  # webcam index
    except (TypeError, ValueError):
        pass
    run(src)
