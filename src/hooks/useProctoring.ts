import { useState, useRef, useCallback, useEffect } from "react";
import { toast } from "sonner";
import { invokeAi } from "@/integrations/aiClient";

// ─── Types ───────────────────────────────────────────────────────────────────

export type ViolationType =
  | "head_turn_left"
  | "head_turn_right"
  | "looking_down"
  | "gaze_away"
  | "no_face"
  | "multiple_faces"
  | "phone_detected"
  | "tab_switch"
  | "session_start"
  | "session_end";

export type Severity = "info" | "low" | "medium" | "high";

// ── Agentic flag verification ──
// High-severity flags are fact-checked by a VLM verifier before they
// penalize the trust score: detectors propose, the verifier disposes.
export type VerificationStatus =
  | "pending"     // frame sent, awaiting verdict
  | "confirmed"   // verifier agrees — full trust penalty applied
  | "dismissed"   // verifier refuted the flag — no penalty
  | "uncertain"   // frame ambiguous — reduced penalty
  | "unverified"; // verifier unavailable — standard penalty (fail-safe)

export interface FlagVerification {
  status: VerificationStatus;
  evidence?: string;   // verifier's visual reasoning
  confidence?: number; // 0-1
  frame?: string;      // the flagged frame (JPEG data URL) — the evidence exhibit
}

export interface ProctoringAlert {
  id: string;
  time: string;
  timestamp: number;
  type: ViolationType;
  message: string;
  severity: Severity;
  verification?: FlagVerification;
}

export interface SessionStats {
  headTurns: number;
  gazeAways: number;
  noFaceEvents: number;
  multipleFaceEvents: number;
  lookingDownEvents: number;
  phoneDetectedEvents: number;
  tabSwitches: number;
  dismissedFlags: number;
}

export interface LiveStatus {
  faceDetected: boolean;
  multipleFaces: boolean;
  phoneInFrame: boolean;
  headDirection: "center" | "left" | "right";
  gazeDirection: "center" | "left" | "right";
  lookingDown: boolean;
  yawPct: number;
  faceARDelta: number;  // face compression % from baseline (negative = looking down)
  gazeDelta: number;    // iris offset deviation from calibrated baseline (%)
  isCalibrating: boolean;
  calibProgress: number; // 0-100
}

// ─── Detection constants ──────────────────────────────────────────────────────

const ALERT_COOLDOWN: Record<ViolationType, number> = {
  head_turn_left: 2500,
  head_turn_right: 2500,
  looking_down: 4000,
  gaze_away: 2000,
  no_face: 2000,
  multiple_faces: 3000,
  phone_detected: 4000,
  tab_switch: 5000,
  session_start: 0,
  session_end: 0,
};

// Head yaw: fraction of half-ear-span — raised to 0.33 to cut borderline turns
const HEAD_TURN_THRESHOLD = 0.33;

// Calibration: sample 90 frames (~3 s) while the person looks at the screen.
// All pitch / gaze thresholds are DELTAS from this calibrated baseline.
const CALIB_FRAMES = 90;

// Face Aspect Ratio compression: faceAR = faceH/faceWidth.
// When head tilts down (phone use), the face foreshortens vertically → faceAR
// drops from its calibrated baseline. A 10 % drop triggers an alert.
const LOOK_DOWN_AR_THRESHOLD = 0.10;

// Consecutive frames faceAR must stay compressed before alerting (~0.7 s at 30 fps)
const LOOK_DOWN_SUSTAINED = 20;

// Gaze (iris offset from eye-centre, normalised by eye-width) must deviate
// this much from the calibrated baseline to trigger an alert
const GAZE_DELTA = 0.05;

// Leaky-counter score gaze must accumulate before alerting (~0.4 s of
// deviation; centered frames drain the score instead of hard-resetting it,
// so single-frame iris jitter no longer wipes out a sustained glance)
const GAZE_SUSTAINED = 10;
const GAZE_LEAK = 2;

// ── Phone detection ──
// COCO-SSD inference runs on its own timer (decoupled from the FaceMesh rAF
// loop) against a downscaled canvas — full-resolution 1280x720 inference on
// the main thread was the bottleneck that made detection feel slow.
const PHONE_CHECK_MS = 700;
const PHONE_DETECT_WIDTH = 320;
// lite_mobilenet_v2 rarely scores phones above 0.6; 0.45 with a 2-hit
// confirmation catches phones far sooner without adding false positives.
const PHONE_SCORE_THRESHOLD = 0.45;
const PHONE_CONFIRM_HITS = 2;

// ── Agentic verification ──
// Flag types whose claim can be fact-checked from a single frame.
// tab_switch is not visual; head turns / gaze are medium-severity and
// too frequent to verify economically.
const VERIFIABLE_TYPES = new Set<ViolationType>([
  "phone_detected",
  "multiple_faces",
  "no_face",
  "looking_down",
]);
const VERIFY_FRAME_WIDTH = 640;   // flagged-frame resolution sent to the verifier
const MAX_CONCURRENT_VERIFY = 2;  // beyond this, fall back to unverified penalty

const NO_FACE_GRACE_MS = 1500;
const VIOLATION_BANNER_MS = 3500;
const LIVE_THROTTLE_MS = 200;

// ─── Helpers ─────────────────────────────────────────────────────────────────

function uid() {
  return Date.now().toString(36) + Math.random().toString(36).slice(2);
}
function timeLabel(d = new Date()) {
  return d.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}
function formatDuration(s: number) {
  return `${Math.floor(s / 60).toString().padStart(2, "0")}:${(s % 60).toString().padStart(2, "0")}`;
}
function isoTimestamp() {
  return new Date().toISOString().slice(0, 19).replace(/[:.]/g, "-");
}
function loadCdnScript(src: string): Promise<void> {
  return new Promise((resolve, reject) => {
    if (document.querySelector(`script[src="${src}"]`)) { resolve(); return; }
    const s = document.createElement("script");
    s.src = src;
    s.onload = () => resolve();
    s.onerror = () => reject(new Error(`Failed to load: ${src}`));
    document.head.appendChild(s);
  });
}
function downloadBlob(content: string, mimeType: string, filename: string) {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = filename;
  document.body.appendChild(a); a.click();
  document.body.removeChild(a); URL.revokeObjectURL(url);
}

const DEFAULT_STATS: SessionStats = { headTurns: 0, gazeAways: 0, noFaceEvents: 0, multipleFaceEvents: 0, lookingDownEvents: 0, phoneDetectedEvents: 0, tabSwitches: 0, dismissedFlags: 0 };
const DEFAULT_LIVE: LiveStatus = { faceDetected: false, multipleFaces: false, phoneInFrame: false, headDirection: "center", gazeDirection: "center", lookingDown: false, yawPct: 0, faceARDelta: 0, gazeDelta: 0, isCalibrating: false, calibProgress: 0 };

// ─── Hook ─────────────────────────────────────────────────────────────────────

export function useProctoring() {
  const [isMonitoring, setIsMonitoring] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [alerts, setAlerts] = useState<ProctoringAlert[]>([]);
  const [sessionTime, setSessionTime] = useState(0);
  const [trustScore, setTrustScore] = useState(100);
  const [stats, setStats] = useState<SessionStats>({ ...DEFAULT_STATS });
  const [currentViolation, setCurrentViolation] = useState<string | null>(null);
  const [liveStatus, setLiveStatus] = useState<LiveStatus>({ ...DEFAULT_LIVE });
  const [verifyEnabled, setVerifyEnabled] = useState(true);

  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const faceMeshRef = useRef<any>(null);
  const rafRef = useRef<number | null>(null);
  const sessionTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const bannerTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const isMonitoringRef = useRef(false);
  const trustRef = useRef(100);
  const lastAlertRef = useRef<Partial<Record<ViolationType, number>>>({});
  const noFaceStartRef = useRef<number | null>(null);
  const hasViolationRef = useRef(false);
  const sessionStartRef = useRef(0);
  const sessionSecondsRef = useRef(0);
  const alertsRef = useRef<ProctoringAlert[]>([]);
  const statsRef = useRef<SessionStats>({ ...DEFAULT_STATS });

  // Sustained looking-down counter
  const lookDownFramesRef = useRef(0);
  // Sustained gaze-away counter
  const gazeFramesRef = useRef(0);

  // COCO-SSD phone detection
  const cocoModelRef = useRef<any>(null);
  const phoneTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const phoneCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const isCocoRunningRef = useRef(false);
  const phoneInFrameRef = useRef(false);
  const phoneHitsRef = useRef(0);

  // Calibration
  const calibCountRef = useRef(0);
  const calibFaceARSumRef = useRef(0);
  const calibGazeSumRef = useRef(0);
  const faceARBaselineRef = useRef(0);
  const gazeBaselineRef = useRef(0);
  const isCalibratedRef = useRef(false);

  // Live status throttle
  const lastLiveRef = useRef(0);

  // Agentic verification
  const verifyEnabledRef = useRef(true);
  const activeVerifyRef = useRef(0);

  useEffect(() => { alertsRef.current = alerts; }, [alerts]);
  useEffect(() => { statsRef.current = stats; }, [stats]);
  useEffect(() => { verifyEnabledRef.current = verifyEnabled; }, [verifyEnabled]);

  // ─── Alert emitter + agentic verification ─────────────────────────────────

  const TRUST_DECAY: Record<Severity, number> = { info: 0, low: 1, medium: 3, high: 6 };

  const applyTrustDecay = useCallback((amount: number) => {
    trustRef.current = Math.max(0, trustRef.current - amount);
    setTrustScore(trustRef.current);
  }, []);

  // Capture the current webcam frame as a downscaled JPEG — the "exhibit"
  // that gets fact-checked and attached to the alert as evidence.
  const captureFrame = useCallback((): string | null => {
    const video = videoRef.current;
    if (!video || video.readyState < 2) return null;
    try {
      const canvas = document.createElement("canvas");
      const scale = VERIFY_FRAME_WIDTH / (video.videoWidth || 1280);
      canvas.width = VERIFY_FRAME_WIDTH;
      canvas.height = Math.round((video.videoHeight || 720) * scale);
      const ctx = canvas.getContext("2d");
      if (!ctx) return null;
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      return canvas.toDataURL("image/jpeg", 0.7);
    } catch {
      return null;
    }
  }, []);

  const updateAlertVerification = useCallback((alertId: string, patch: Partial<FlagVerification>) => {
    setAlerts((prev) =>
      prev.map((a) =>
        a.id === alertId && a.verification
          ? { ...a, verification: { ...a.verification, ...patch } }
          : a,
      ),
    );
  }, []);

  // Second stage: the VLM verifier fact-checks the detector's claim against
  // the flagged frame. Only its verdict decides the trust penalty:
  //   CONFIRMED → full penalty · UNCERTAIN → half · REFUTED → none (dismissed)
  //   verifier unreachable → full penalty, marked "unverified" (fail-safe)
  const verifyFlag = useCallback(async (alertId: string, type: ViolationType, claim: string, frame: string, severity: Severity) => {
    activeVerifyRef.current++;
    try {
      const { data, error } = await invokeAi<{ verdict: string; evidence: string; confidence: number }>(
        "verify-flag",
        { frame, flagType: type, claim },
      );
      if (error || !data?.verdict) throw error ?? new Error("No verdict");

      if (data.verdict === "CONFIRMED") {
        applyTrustDecay(TRUST_DECAY[severity]);
        updateAlertVerification(alertId, { status: "confirmed", evidence: data.evidence, confidence: data.confidence });
      } else if (data.verdict === "REFUTED") {
        setStats((prev) => ({ ...prev, dismissedFlags: prev.dismissedFlags + 1 }));
        updateAlertVerification(alertId, { status: "dismissed", evidence: data.evidence, confidence: data.confidence });
        toast.success("Flag dismissed by AI verifier — no penalty applied");
      } else {
        applyTrustDecay(Math.ceil(TRUST_DECAY[severity] / 2));
        updateAlertVerification(alertId, { status: "uncertain", evidence: data.evidence, confidence: data.confidence });
      }
    } catch {
      // Verifier unreachable — apply the standard penalty so verification
      // can never be gamed by blocking the network
      applyTrustDecay(TRUST_DECAY[severity]);
      updateAlertVerification(alertId, { status: "unverified", evidence: "Verifier unavailable — standard penalty applied" });
    } finally {
      activeVerifyRef.current--;
    }
  }, [applyTrustDecay, updateAlertVerification]);

  const verifyFlagRef = useRef(verifyFlag);
  useEffect(() => { verifyFlagRef.current = verifyFlag; }, [verifyFlag]);

  const pushAlert = useCallback((type: ViolationType, message: string, severity: Severity) => {
    const now = Date.now();
    const cooldown = ALERT_COOLDOWN[type];
    const last = lastAlertRef.current[type] ?? 0;
    if (cooldown > 0 && now - last < cooldown) return;
    lastAlertRef.current[type] = now;

    // First stage (detector) proposes; decide whether the second stage
    // (VLM verifier) gets to dispose before any trust penalty applies
    const wantsVerify =
      verifyEnabledRef.current &&
      VERIFIABLE_TYPES.has(type) &&
      activeVerifyRef.current < MAX_CONCURRENT_VERIFY;
    const frame = wantsVerify ? captureFrame() : null;
    const willVerify = wantsVerify && !!frame;

    const alert: ProctoringAlert = {
      id: uid(), time: timeLabel(), timestamp: now, type, message, severity,
      ...(willVerify ? { verification: { status: "pending" as VerificationStatus, frame: frame! } } : {}),
    };
    setAlerts((prev) => [alert, ...prev].slice(0, 60));

    setStats((prev) => {
      const next = { ...prev };
      if (type === "head_turn_left" || type === "head_turn_right") next.headTurns++;
      else if (type === "gaze_away") next.gazeAways++;
      else if (type === "no_face") next.noFaceEvents++;
      else if (type === "multiple_faces") next.multipleFaceEvents++;
      else if (type === "looking_down") next.lookingDownEvents++;
      else if (type === "phone_detected") next.phoneDetectedEvents++;
      else if (type === "tab_switch") next.tabSwitches++;
      return next;
    });

    if (willVerify) {
      // Trust decay is DEFERRED until the verifier's verdict
      verifyFlagRef.current(alert.id, type, message, frame!, severity);
    } else {
      applyTrustDecay(TRUST_DECAY[severity]);
    }

    if (severity !== "info") {
      hasViolationRef.current = true;
      setCurrentViolation(willVerify ? `${message} — verifying…` : message);
      if (bannerTimerRef.current) clearTimeout(bannerTimerRef.current);
      bannerTimerRef.current = setTimeout(() => {
        hasViolationRef.current = false;
        setCurrentViolation(null);
      }, VIOLATION_BANNER_MS);
    }
  }, [applyTrustDecay, captureFrame]);

  const pushAlertRef = useRef(pushAlert);
  useEffect(() => { pushAlertRef.current = pushAlert; }, [pushAlert]);

  // ─── Frame analysis ───────────────────────────────────────────────────────

  const analyzeFrame = useCallback((results: any) => {
    if (!isMonitoringRef.current) return;

    const allFaces: any[][] = results.multiFaceLandmarks ?? [];
    const now = Date.now();

    // ── No face ────────────────────────────────────────────────────────────
    if (allFaces.length === 0) {
      lookDownFramesRef.current = 0;
      if (noFaceStartRef.current === null) {
        noFaceStartRef.current = now;
      } else if (now - noFaceStartRef.current > NO_FACE_GRACE_MS) {
        pushAlertRef.current("no_face", "Face not detected — candidate may have left the frame", "high");
      }
      if (now - lastLiveRef.current > LIVE_THROTTLE_MS) {
        lastLiveRef.current = now;
        setLiveStatus({ ...DEFAULT_LIVE, faceDetected: false });
      }
      return;
    }

    noFaceStartRef.current = null;

    if (allFaces.length > 1) {
      pushAlertRef.current("multiple_faces", `${allFaces.length} faces detected in frame`, "high");
    }

    const lm = allFaces[0];

    // ── Head pose ──────────────────────────────────────────────────────────
    const noseTip  = lm[1];
    const leftEar  = lm[234];
    const rightEar = lm[454];
    const forehead = lm[10];
    const chin     = lm[152];

    const earSpan  = rightEar.x - leftEar.x;
    const faceMidX = (leftEar.x + rightEar.x) / 2;
    const yaw = earSpan > 0.01 ? (noseTip.x - faceMidX) / (earSpan / 2) : 0;

    const faceH    = chin.y - forehead.y;

    // ── Iris gaze ──────────────────────────────────────────────────────────
    // Iris landmarks 468 / 473 are only present with refineLandmarks: true
    let gazeOff = 0;
    let gazeValid = false;
    if (lm.length > 473) {
      const lIris = lm[468]; // left iris centre (person's left = image right)
      const rIris = lm[473]; // right iris centre

      // Eye-corner landmarks (verified against MediaPipe canonical face model)
      const lOuter = lm[33];   // person's left eye, outer (temporal) corner — high x
      const lInner = lm[133];  // person's left eye, inner (nasal) corner  — lower x
      const rInner = lm[362];  // person's right eye, inner corner          — higher x
      const rOuter = lm[263];  // person's right eye, outer corner          — low x

      const lEyeW = lOuter.x - lInner.x; // should be > 0
      const rEyeW = rInner.x - rOuter.x; // should be > 0

      if (lEyeW > 0.005 && rEyeW > 0.005) {
        // Signed offset of iris from eye-centre, normalised by eye-width
        // Positive = iris is toward the temporal side (outer corner)
        // When looking right (person's right, image left): both irises move toward nasal — offset goes negative
        // When looking left  (person's left,  image right): both irises move toward temporal — offset goes positive
        const lOff = (lIris.x - (lOuter.x + lInner.x) / 2) / lEyeW;
        const rOff = (rIris.x - (rOuter.x + rInner.x) / 2) / rEyeW;
        gazeOff = (lOff + rOff) / 2;
        gazeValid = true;
      }
    }

    // ── Calibration (first CALIB_FRAMES frames) ────────────────────────────
    const faceWidth = earSpan; // earSpan = rightEar.x - leftEar.x (positive when earSpan > 0)
    const faceAR = faceWidth > 0.01 ? faceH / faceWidth : 0;

    if (!isCalibratedRef.current) {
      calibCountRef.current++;
      calibFaceARSumRef.current += faceAR;
      if (gazeValid) calibGazeSumRef.current += gazeOff;

      const progress = Math.round((calibCountRef.current / CALIB_FRAMES) * 100);
      if (now - lastLiveRef.current > LIVE_THROTTLE_MS) {
        lastLiveRef.current = now;
        setLiveStatus({ ...DEFAULT_LIVE, faceDetected: true, isCalibrating: true, calibProgress: progress });
      }

      if (calibCountRef.current >= CALIB_FRAMES) {
        faceARBaselineRef.current = calibFaceARSumRef.current / CALIB_FRAMES;
        gazeBaselineRef.current   = calibGazeSumRef.current  / CALIB_FRAMES;
        isCalibratedRef.current   = true;
        toast.info("Calibration complete — monitoring active");
      }
      return; // no violation checks during calibration
    }

    // ── Head turn alerts ───────────────────────────────────────────────────
    if (yaw > HEAD_TURN_THRESHOLD) {
      pushAlertRef.current("head_turn_right", `Head turned right (${Math.round(Math.abs(yaw) * 100)}% deviation)`, "medium");
    } else if (yaw < -HEAD_TURN_THRESHOLD) {
      pushAlertRef.current("head_turn_left", `Head turned left (${Math.round(Math.abs(yaw) * 100)}% deviation)`, "medium");
    }

    // ── Looking down via face aspect-ratio compression ─────────────────────
    // When head tilts forward toward a phone, the face foreshortens vertically
    // so faceAR = faceH/faceWidth drops below the calibrated baseline.
    const faceARDelta = faceARBaselineRef.current > 0.01
      ? (faceAR - faceARBaselineRef.current) / faceARBaselineRef.current
      : 0;

    if (faceARDelta < -LOOK_DOWN_AR_THRESHOLD && faceWidth > 0.01) {
      lookDownFramesRef.current++;
      if (lookDownFramesRef.current === LOOK_DOWN_SUSTAINED) {
        pushAlertRef.current(
          "looking_down",
          `Sustained downward gaze — possible mobile phone usage (face ${Math.round(Math.abs(faceARDelta) * 100)}% compressed)`,
          "high",
        );
      }
    } else {
      lookDownFramesRef.current = 0;
    }

    // ── Gaze — delta from baseline, leaky sustained counter ────────────────
    // Skipped while the head itself is turned: the iris/eye-corner geometry
    // is skewed at high yaw and the head-turn alert already covers it. Gaze
    // specifically catches eyes-only glancing with the head still centered.
    let gazeDir: LiveStatus["gazeDirection"] = "center";
    const headCentered = Math.abs(yaw) <= HEAD_TURN_THRESHOLD;
    const gazeDelta = gazeValid ? gazeOff - gazeBaselineRef.current : 0;
    if (gazeValid && headCentered) {
      if (Math.abs(gazeDelta) > GAZE_DELTA) {
        gazeFramesRef.current++;
        gazeDir = gazeDelta > 0 ? "left" : "right";
        if (gazeFramesRef.current >= GAZE_SUSTAINED) {
          pushAlertRef.current(
            "gaze_away",
            `Gaze directed off-screen (looking ${gazeDir}, ${Math.round(Math.abs(gazeDelta) * 100)}% deviation)`,
            "medium",
          );
          gazeFramesRef.current = 0;
        }
      } else {
        // Drain instead of hard reset — one jittery frame no longer erases
        // an otherwise-sustained off-screen glance
        gazeFramesRef.current = Math.max(0, gazeFramesRef.current - GAZE_LEAK);
      }
    }

    // ── Live status update (throttled) ─────────────────────────────────────
    if (now - lastLiveRef.current > LIVE_THROTTLE_MS) {
      lastLiveRef.current = now;
      setLiveStatus({
        faceDetected: true,
        multipleFaces: allFaces.length > 1,
        phoneInFrame: phoneInFrameRef.current,
        headDirection:
          yaw > HEAD_TURN_THRESHOLD  ? "right" :
          yaw < -HEAD_TURN_THRESHOLD ? "left"  : "center",
        gazeDirection: gazeDir,
        lookingDown: lookDownFramesRef.current >= LOOK_DOWN_SUSTAINED,
        yawPct:      Math.round(Math.abs(yaw) * 100),
        faceARDelta: Math.round(faceARDelta * 100),
        gazeDelta:   Math.round(Math.abs(gazeDelta) * 100),
        isCalibrating: false,
        calibProgress: 100,
      });
    }
  }, []);

  const analyzeFrameRef = useRef(analyzeFrame);
  useEffect(() => { analyzeFrameRef.current = analyzeFrame; }, [analyzeFrame]);

  // ─── Detection loop ───────────────────────────────────────────────────────

  const loop = useCallback(async () => {
    if (!isMonitoringRef.current || !faceMeshRef.current || !videoRef.current) return;
    const video = videoRef.current;
    if (video.readyState >= 2 && !video.paused) {
      try { await faceMeshRef.current.send({ image: video }); } catch { /* skip frame */ }
    }
    if (isMonitoringRef.current) rafRef.current = requestAnimationFrame(loopRef.current);
  }, []);

  const loopRef = useRef(loop);
  useEffect(() => { loopRef.current = loop; }, [loop]);

  // ── COCO-SSD phone check: own timer, downscaled input ─────────────────────
  // Runs independently of the FaceMesh loop so FaceMesh latency never delays
  // it, and infers on a small canvas instead of the full 1280x720 frame.
  const checkPhone = useCallback(() => {
    const video = videoRef.current;
    if (!isMonitoringRef.current || !cocoModelRef.current || !video) return;
    if (video.readyState < 2 || video.paused || isCocoRunningRef.current) return;

    if (!phoneCanvasRef.current) phoneCanvasRef.current = document.createElement("canvas");
    const canvas = phoneCanvasRef.current;
    const scale = PHONE_DETECT_WIDTH / (video.videoWidth || 1280);
    canvas.width = PHONE_DETECT_WIDTH;
    canvas.height = Math.round((video.videoHeight || 720) * scale);
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    isCocoRunningRef.current = true;
    cocoModelRef.current
      .detect(canvas, 5, PHONE_SCORE_THRESHOLD)
      .then((preds: any[]) => {
        const phone = preds.find(
          (p) => p.class === "cell phone" && p.score >= PHONE_SCORE_THRESHOLD,
        );
        if (phone) {
          phoneHitsRef.current++;
          // Confirm across consecutive checks before alerting to keep the
          // lower score threshold from producing false positives
          if (phoneHitsRef.current >= PHONE_CONFIRM_HITS) {
            phoneInFrameRef.current = true;
            if (isMonitoringRef.current) {
              pushAlertRef.current(
                "phone_detected",
                `Mobile phone detected in frame (${Math.round(phone.score * 100)}% confidence)`,
                "high",
              );
            }
          }
        } else {
          phoneHitsRef.current = 0;
          phoneInFrameRef.current = false;
        }
      })
      .catch(() => {})
      .finally(() => { isCocoRunningRef.current = false; });
  }, []);

  const checkPhoneRef = useRef(checkPhone);
  useEffect(() => { checkPhoneRef.current = checkPhone; }, [checkPhone]);

  // ─── Start ────────────────────────────────────────────────────────────────

  const startMonitoring = useCallback(async () => {
    setIsLoading(true);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 1280, height: 720, facingMode: "user" },
        audio: false,
      });
      streamRef.current = stream;

      const video = videoRef.current!;
      video.srcObject = stream;
      await new Promise<void>((res, rej) => {
        video.onloadedmetadata = () => res();
        setTimeout(() => rej(new Error("Video load timeout")), 10000);
      });
      await video.play();

      const CDN = "https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh@0.4.1633559619";
      await loadCdnScript(`${CDN}/face_mesh.js`);

      const FaceMeshCtor = (window as any).FaceMesh;
      if (typeof FaceMeshCtor !== "function") throw new Error("FaceMesh failed to load from CDN");

      const fm = new FaceMeshCtor({ locateFile: (f: string) => `${CDN}/${f}` });
      fm.setOptions({ maxNumFaces: 3, refineLandmarks: true, minDetectionConfidence: 0.5, minTrackingConfidence: 0.5 });
      fm.onResults((r: any) => analyzeFrameRef.current(r));
      await fm.initialize();
      faceMeshRef.current = fm;

      // Load COCO-SSD for phone object detection (TF.js must load first)
      const TF_CDN = "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.15.0/dist/tf.min.js";
      const COCO_CDN = "https://cdn.jsdelivr.net/npm/@tensorflow-models/coco-ssd@2.2.2/dist/coco-ssd.min.js";
      try {
        await loadCdnScript(TF_CDN);
        await loadCdnScript(COCO_CDN);
        const tf = (window as any).tf;
        // GPU-accelerated inference; without this TF.js can fall back to the
        // much slower CPU backend and every detect blocks the main thread
        if (tf?.setBackend) { try { await tf.setBackend("webgl"); await tf.ready(); } catch {} }
        const cocoSsd = (window as any).cocoSsd;
        if (cocoSsd?.load) {
          cocoModelRef.current = await cocoSsd.load({ base: "lite_mobilenet_v2" });
          // Warm-up inference — first detect compiles WebGL shaders (~1-2 s);
          // doing it now means the first real phone check is already fast
          try {
            const warm = document.createElement("canvas");
            warm.width = PHONE_DETECT_WIDTH; warm.height = 240;
            await cocoModelRef.current.detect(warm);
          } catch {}
        }
      } catch {
        // Phone object detection unavailable — faceAR tilt detection still works
      }

      // Reset everything
      trustRef.current = 100;
      lastAlertRef.current = {};
      noFaceStartRef.current = null;
      hasViolationRef.current = false;
      lookDownFramesRef.current = 0;
      gazeFramesRef.current = 0;
      isCocoRunningRef.current = false;
      phoneInFrameRef.current = false;
      phoneHitsRef.current = 0;
      activeVerifyRef.current = 0;
      calibCountRef.current = 0;
      calibFaceARSumRef.current = 0;
      calibGazeSumRef.current = 0;
      faceARBaselineRef.current = 0;
      gazeBaselineRef.current = 0;
      isCalibratedRef.current = false;
      lastLiveRef.current = 0;
      sessionStartRef.current = Date.now();
      sessionSecondsRef.current = 0;

      setAlerts([]);
      setStats({ ...DEFAULT_STATS });
      setTrustScore(100);
      setSessionTime(0);
      setCurrentViolation(null);
      setLiveStatus({ ...DEFAULT_LIVE });
      setIsMonitoring(true);
      isMonitoringRef.current = true;

      pushAlertRef.current("session_start", "Proctoring started — calibrating baseline (look at screen)…", "info");
      toast.info("Look straight at the screen — calibrating for 3 seconds…");

      sessionTimerRef.current = setInterval(() => {
        sessionSecondsRef.current++;
        setSessionTime(sessionSecondsRef.current);
      }, 1000);

      phoneTimerRef.current = setInterval(() => checkPhoneRef.current(), PHONE_CHECK_MS);

      rafRef.current = requestAnimationFrame(() => loopRef.current());
    } catch (err: any) {
      toast.error("Failed to start: " + (err?.message ?? "Camera access denied"));
    } finally {
      setIsLoading(false);
    }
  }, []);

  // ─── Stop ─────────────────────────────────────────────────────────────────

  const stopMonitoring = useCallback(() => {
    isMonitoringRef.current = false;
    if (rafRef.current) { cancelAnimationFrame(rafRef.current); rafRef.current = null; }
    if (sessionTimerRef.current) { clearInterval(sessionTimerRef.current); sessionTimerRef.current = null; }
    if (bannerTimerRef.current) { clearTimeout(bannerTimerRef.current); bannerTimerRef.current = null; }
    if (phoneTimerRef.current) { clearInterval(phoneTimerRef.current); phoneTimerRef.current = null; }
    if (faceMeshRef.current) { try { faceMeshRef.current.close(); } catch {} faceMeshRef.current = null; }
    if (cocoModelRef.current) { try { cocoModelRef.current.dispose(); } catch {} cocoModelRef.current = null; }
    isCocoRunningRef.current = false;
    phoneInFrameRef.current = false;
    phoneHitsRef.current = 0;
    if (streamRef.current) { streamRef.current.getTracks().forEach(t => t.stop()); streamRef.current = null; }
    if (videoRef.current) videoRef.current.srcObject = null;

    setIsMonitoring(false);
    setCurrentViolation(null);
    setLiveStatus({ ...DEFAULT_LIVE });
    hasViolationRef.current = false;
    lookDownFramesRef.current = 0;
    pushAlertRef.current("session_end", "Proctoring session ended", "info");
    toast.info("Proctoring session ended");
  }, []);

  // ─── Tab / window focus ───────────────────────────────────────────────────

  useEffect(() => {
    const onVisibility = () => {
      if (document.hidden && isMonitoringRef.current) {
        pushAlertRef.current("tab_switch", "Tab switch or window minimize detected", "high");
        toast.warning("⚠ Tab switch detected");
      }
    };
    const onBlur = () => {
      if (isMonitoringRef.current) pushAlertRef.current("tab_switch", "Browser window lost focus", "medium");
    };
    document.addEventListener("visibilitychange", onVisibility);
    window.addEventListener("blur", onBlur);
    return () => {
      document.removeEventListener("visibilitychange", onVisibility);
      window.removeEventListener("blur", onBlur);
    };
  }, []);

  useEffect(() => {
    return () => { if (isMonitoringRef.current) stopMonitoring(); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ─── Export ───────────────────────────────────────────────────────────────

  const exportCSV = useCallback(() => {
    const s = statsRef.current;
    const rows = [
      ["Timestamp", "Time", "Type", "Message", "Severity", "Verification", "Evidence"],
      ...alertsRef.current.map(a => [
        a.timestamp, a.time, a.type, `"${a.message}"`, a.severity,
        a.verification?.status ?? "",
        a.verification?.evidence ? `"${a.verification.evidence.replace(/"/g, "'")}"` : "",
      ]),
      [],
      ["=== SESSION SUMMARY ==="],
      ["Trust Score", `${trustRef.current}%`],
      ["Duration", formatDuration(sessionSecondsRef.current)],
      ["Head Turns", s.headTurns],
      ["Gaze Aways", s.gazeAways],
      ["No Face Events", s.noFaceEvents],
      ["Multiple Faces", s.multipleFaceEvents],
      ["Looking Down Events", s.lookingDownEvents],
      ["Phone Detections", s.phoneDetectedEvents],
      ["Tab Switches", s.tabSwitches],
      ["Flags Dismissed by AI Verifier", s.dismissedFlags],
    ].map(r => r.join(",")).join("\n");
    downloadBlob(rows, "text/csv;charset=utf-8;", `proctor-${isoTimestamp()}.csv`);
    toast.success("CSV report downloaded");
  }, []);

  const exportJSON = useCallback(() => {
    const payload = {
      exportedAt: new Date().toISOString(),
      sessionStart: sessionStartRef.current ? new Date(sessionStartRef.current).toISOString() : null,
      durationSeconds: sessionSecondsRef.current,
      trustScore: trustRef.current,
      verificationEnabled: verifyEnabledRef.current,
      stats: statsRef.current,
      // Full audit trail: each verified alert carries the verifier's verdict,
      // reasoning, and the flagged frame itself as the evidence exhibit
      alerts: alertsRef.current,
    };
    downloadBlob(JSON.stringify(payload, null, 2), "application/json", `proctor-${isoTimestamp()}.json`);
    toast.success("JSON report downloaded");
  }, []);

  return {
    isMonitoring, isLoading, alerts, sessionTime, trustScore,
    stats, currentViolation, liveStatus, videoRef,
    verifyEnabled, setVerifyEnabled,
    startMonitoring, stopMonitoring, exportCSV, exportJSON,
  };
}
