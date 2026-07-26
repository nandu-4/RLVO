import {
  Video,
  AlertTriangle,
  CheckCircle,
  Download,
  Play,
  Square,
  FileJson,
  ArrowLeft,
  ArrowRight,
  ArrowDown,
  EyeOff,
  UserX,
  Users,
  ExternalLink,
  ShieldCheck,
  ShieldAlert,
  ShieldX,
  Eye,
  User,
  Activity,
  Smartphone,
  Loader2,
  BadgeCheck,
  HelpCircle,
  Sparkles,
  Package,
} from "lucide-react";
import { Switch } from "@/components/ui/switch";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import Navigation from "@/components/Navigation";
import {
  useProctoring,
  type ProctoringAlert,
  type ViolationType,
  type Severity,
  type LiveStatus,
} from "@/hooks/useProctoring";

// ─── Alert log item ───────────────────────────────────────────────────────────

const VIOLATION_ICONS: Record<ViolationType, React.ReactNode> = {
  head_turn_left: <ArrowLeft className="h-4 w-4" />,
  head_turn_right: <ArrowRight className="h-4 w-4" />,
  looking_down: <ArrowDown className="h-4 w-4" />,
  gaze_away: <EyeOff className="h-4 w-4" />,
  no_face: <UserX className="h-4 w-4" />,
  multiple_faces: <Users className="h-4 w-4" />,
  phone_detected: <Smartphone className="h-4 w-4" />,
  new_object: <Package className="h-4 w-4" />,
  tab_switch: <ExternalLink className="h-4 w-4" />,
  session_start: <Play className="h-4 w-4" />,
  session_end: <Square className="h-4 w-4" />,
};

const VIOLATION_LABELS: Record<ViolationType, string> = {
  head_turn_left: "Head Left",
  head_turn_right: "Head Right",
  looking_down: "Looking Down",
  gaze_away: "Gaze Away",
  no_face: "No Face",
  multiple_faces: "Multi-Face",
  phone_detected: "Phone Detected",
  new_object: "New Object",
  tab_switch: "Tab Switch",
  session_start: "Started",
  session_end: "Ended",
};

const SEVERITY_CARD: Record<Severity, string> = {
  info: "border-blue-200 dark:border-blue-800 bg-blue-50 dark:bg-blue-950/40",
  low: "border-yellow-200 dark:border-yellow-800 bg-yellow-50 dark:bg-yellow-950/40",
  medium: "border-orange-200 dark:border-orange-800 bg-orange-50 dark:bg-orange-950/40",
  high: "border-red-200 dark:border-red-800 bg-red-50 dark:bg-red-950/40",
};

const SEVERITY_ICON: Record<Severity, string> = {
  info: "text-blue-500",
  low: "text-yellow-500",
  medium: "text-orange-500",
  high: "text-red-500",
};

// ─── Verification badge — the verdict of the agentic second stage ─────────────

function VerificationBadge({ alert }: { alert: ProctoringAlert }) {
  const v = alert.verification;
  if (!v) return null;

  switch (v.status) {
    case "pending":
      return (
        <span className="inline-flex items-center gap-1 text-xs font-medium text-blue-600 dark:text-blue-400">
          <Loader2 className="h-3 w-3 animate-spin" /> AI verifying…
        </span>
      );
    case "confirmed":
      return (
        <span className="inline-flex items-center gap-1 text-xs font-medium text-red-600 dark:text-red-400">
          <BadgeCheck className="h-3 w-3" /> Confirmed by AI verifier
        </span>
      );
    case "dismissed":
      return (
        <span className="inline-flex items-center gap-1 text-xs font-medium text-green-600 dark:text-green-400">
          <ShieldX className="h-3 w-3" /> Dismissed — no penalty
        </span>
      );
    case "uncertain":
      return (
        <span className="inline-flex items-center gap-1 text-xs font-medium text-yellow-600 dark:text-yellow-400">
          <HelpCircle className="h-3 w-3" /> Uncertain — reduced penalty
        </span>
      );
    default:
      return (
        <span className="inline-flex items-center gap-1 text-xs font-medium text-muted-foreground">
          <HelpCircle className="h-3 w-3" /> Unverified
        </span>
      );
  }
}

function AlertItem({ alert }: { alert: ProctoringAlert }) {
  const v = alert.verification;
  const dismissed = v?.status === "dismissed";
  return (
    <div className={`p-2.5 rounded-lg border text-sm animate-fade-in ${dismissed ? "border-green-200 dark:border-green-800 bg-green-50/60 dark:bg-green-950/30 opacity-80" : SEVERITY_CARD[alert.severity]}`}>
      <div className="flex items-start gap-2">
        <span className={`mt-0.5 flex-shrink-0 ${dismissed ? "text-green-500" : SEVERITY_ICON[alert.severity]}`}>
          {VIOLATION_ICONS[alert.type]}
        </span>
        <div className="flex-1 min-w-0">
          <span className="font-semibold text-xs uppercase tracking-wide opacity-60">
            {VIOLATION_LABELS[alert.type]}
          </span>
          <p className={`text-xs leading-snug mt-0.5 ${dismissed ? "line-through opacity-70" : ""}`}>{alert.message}</p>
          <p className="text-xs text-muted-foreground mt-0.5">{alert.time}</p>
          {v && (
            <div className="mt-1.5 pt-1.5 border-t border-black/5 dark:border-white/10">
              <VerificationBadge alert={alert} />
              {v.evidence && v.status !== "pending" && (
                <p className="text-xs text-muted-foreground italic mt-0.5 leading-snug">
                  "{v.evidence}"
                </p>
              )}
            </div>
          )}
        </div>
        {v?.frame && (
          <img
            src={v.frame}
            alt="Flagged frame evidence"
            className="w-16 h-12 object-cover rounded border flex-shrink-0 mt-0.5"
          />
        )}
      </div>
    </div>
  );
}

// ─── Calibration overlay — center dot, then 4 corner dots for gaze bounds ─────

const CORNER_POS = [
  "top-6 left-6",      // 0 TL
  "top-6 right-6",     // 1 TR
  "bottom-6 right-6",  // 2 BR
  "bottom-6 left-6",   // 3 BL
];

function CalibrationOverlay({ status }: { status: LiveStatus }) {
  if (!status.isCalibrating) return null;
  const corners = status.calibStage === "corners";
  return (
    <div className="fixed inset-0 z-50 bg-black/80 select-none">
      {/* The dot */}
      <div
        className={
          corners
            ? `absolute ${CORNER_POS[status.calibCorner]} h-6 w-6 rounded-full bg-primary ring-8 ring-primary/30 animate-pulse`
            : "absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 h-6 w-6 rounded-full bg-primary ring-8 ring-primary/30 animate-pulse"
        }
      />
      {/* Instructions */}
      <div className="absolute inset-x-0 top-1/2 translate-y-8 text-center text-white px-6 pointer-events-none">
        <p className="text-xl font-semibold">
          {corners
            ? `Follow the dot with your EYES only (${status.calibCorner + 1}/4)`
            : "Look at the dot — measuring your neutral position"}
        </p>
        <p className="text-sm text-white/70 mt-2">
          {corners
            ? "Keep your head still — this teaches the system your personal gaze range"
            : "Sit naturally, as you will during the exam"}
        </p>
        <div className="max-w-xs mx-auto mt-4">
          <Progress value={status.calibProgress} className="h-2" />
          <p className="text-xs text-white/60 mt-1">{status.calibProgress}%</p>
        </div>
      </div>
    </div>
  );
}

// ─── Live status panel ────────────────────────────────────────────────────────

function StatusRow({
  icon,
  label,
  value,
  ok,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  ok: boolean;
}) {
  return (
    <div className="flex items-center justify-between py-2 border-b last:border-0">
      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        {icon}
        <span>{label}</span>
      </div>
      <div className="flex items-center gap-1.5">
        <span
          className={`h-2 w-2 rounded-full flex-shrink-0 ${ok ? "bg-green-500" : "bg-red-500 animate-pulse"}`}
        />
        <span className={`text-sm font-semibold ${ok ? "text-green-600 dark:text-green-400" : "text-red-600 dark:text-red-400"}`}>
          {value}
        </span>
      </div>
    </div>
  );
}

function LiveStatusPanel({
  status,
  isMonitoring,
  phoneDetectorStatus,
}: {
  status: LiveStatus;
  isMonitoring: boolean;
  phoneDetectorStatus: "loading" | "ready" | "unavailable";
}) {
  const headLabel =
    status.headDirection === "left"
      ? `Turned Left (${status.yawPct}%)`
      : status.headDirection === "right"
        ? `Turned Right (${status.yawPct}%)`
        : "Centered";

  const gazeLabel =
    status.gazeDirection === "center"
      ? "On Screen"
      : `Off-screen ${status.gazeDirection[0].toUpperCase()}${status.gazeDirection.slice(1)} (Δ${status.gazeDelta}%)`;

  const pitchLabel = status.lookingDown
    ? `Down — phone suspected (${Math.abs(status.faceARDelta)}% face compression)`
    : status.faceARDelta < -8
      ? `Slightly down (${Math.abs(status.faceARDelta)}% compression)`
      : "Normal";

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center gap-2 text-base">
          <Activity className="h-4 w-4 text-primary" />
          Live Detection Status
        </CardTitle>
        <CardDescription>
          {!isMonitoring
            ? "Start monitoring to see live status"
            : status.isCalibrating
              ? "Calibrating baseline — keep looking at the screen…"
              : "Real-time AI analysis · updated every 200 ms"}
        </CardDescription>
      </CardHeader>
      <CardContent className="pt-0">
        {/* Calibration progress bar */}
        {isMonitoring && status.isCalibrating && (
          <div className="mb-3">
            <div className="flex justify-between text-xs text-muted-foreground mb-1">
              <span>Calibrating neutral position</span>
              <span>{status.calibProgress}%</span>
            </div>
            <Progress value={status.calibProgress} className="h-2" />
            <p className="text-xs text-muted-foreground mt-1">
              Look straight at the screen and stay still
            </p>
          </div>
        )}

        <div className="divide-y">
          <StatusRow
            icon={<User className="h-4 w-4" />}
            label="Face Detected"
            value={
              !isMonitoring
                ? "—"
                : status.isCalibrating
                  ? "Calibrating…"
                  : status.multipleFaces
                    ? "Multiple — suspicious"
                    : status.faceDetected
                      ? "Detected"
                      : "Not Detected"
            }
            ok={isMonitoring ? status.faceDetected && !status.multipleFaces : true}
          />
          <StatusRow
            icon={<ArrowLeft className="h-4 w-4" />}
            label="Head Direction"
            value={isMonitoring && !status.isCalibrating ? headLabel : "—"}
            ok={!isMonitoring || status.isCalibrating || status.headDirection === "center"}
          />
          <StatusRow
            icon={<Eye className="h-4 w-4" />}
            label="Gaze Direction"
            value={isMonitoring && !status.isCalibrating ? gazeLabel : "—"}
            ok={!isMonitoring || status.isCalibrating || status.gazeDirection === "center"}
          />
          <StatusRow
            icon={<ArrowDown className="h-4 w-4" />}
            label="Looking Down"
            value={isMonitoring && !status.isCalibrating ? pitchLabel : "—"}
            ok={!isMonitoring || status.isCalibrating || !status.lookingDown}
          />
          <StatusRow
            icon={<Smartphone className="h-4 w-4" />}
            label="Phone in Frame"
            value={
              !isMonitoring
                ? "—"
                : phoneDetectorStatus === "unavailable"
                  ? "Detector blocked!"
                  : phoneDetectorStatus === "loading"
                    ? "Loading model…"
                    : status.isCalibrating
                      ? "—"
                      : status.phoneInFrame
                        ? "DETECTED!"
                        : "Clear"
            }
            ok={
              !isMonitoring ||
              (phoneDetectorStatus !== "unavailable" &&
                (phoneDetectorStatus === "loading" || status.isCalibrating || !status.phoneInFrame))
            }
          />
          <StatusRow
            icon={<Users className="h-4 w-4" />}
            label="Multiple Faces"
            value={!isMonitoring || status.isCalibrating ? "—" : status.multipleFaces ? "Detected!" : "Clear"}
            ok={!isMonitoring || status.isCalibrating || !status.multipleFaces}
          />
        </div>
      </CardContent>
    </Card>
  );
}

// ─── Stat card ────────────────────────────────────────────────────────────────

function StatCard({
  title,
  value,
  sub,
  warn,
  valueClass,
}: {
  title: string;
  value: string | number;
  sub?: React.ReactNode;
  warn?: boolean;
  valueClass?: string;
}) {
  return (
    <Card className={warn ? "border-orange-400 dark:border-orange-600" : ""}>
      <CardHeader className="pb-2 pt-4 px-4">
        <CardTitle className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
          {title}
        </CardTitle>
      </CardHeader>
      <CardContent className="px-4 pb-4">
        <div className={`text-2xl font-bold ${warn ? "text-orange-500" : ""} ${valueClass ?? ""}`}>
          {value}
        </div>
        {sub}
      </CardContent>
    </Card>
  );
}

// ─── Main page ────────────────────────────────────────────────────────────────

const Proctoring = () => {
  const {
    isMonitoring,
    isLoading,
    alerts,
    sessionTime,
    trustScore,
    stats,
    currentViolation,
    liveStatus,
    videoRef,
    verifyEnabled,
    setVerifyEnabled,
    phoneDetectorStatus,
    startMonitoring,
    stopMonitoring,
    exportCSV,
    exportJSON,
  } = useProctoring();

  const formatTime = (s: number) => {
    const m = Math.floor(s / 60).toString().padStart(2, "0");
    return `${m}:${(s % 60).toString().padStart(2, "0")}`;
  };

  const trustColor =
    trustScore >= 80 ? "text-green-500" :
    trustScore >= 60 ? "text-yellow-500" :
    trustScore >= 40 ? "text-orange-500" : "text-red-500";

  const totalViolations =
    stats.headTurns + stats.gazeAways + stats.noFaceEvents +
    stats.multipleFaceEvents + stats.lookingDownEvents +
    stats.phoneDetectedEvents + stats.newObjectEvents + stats.tabSwitches;

  const hasData = alerts.length > 0;

  return (
    <div className="min-h-screen bg-gradient-hero">
      <Navigation />

      {/* Full-screen gaze calibration overlay */}
      {isMonitoring && <CalibrationOverlay status={liveStatus} />}

      {/* Violation banner */}
      {currentViolation && (
        <div className="bg-red-500 text-white py-2.5 px-6 text-center font-semibold text-sm animate-fade-in flex items-center justify-center gap-2">
          <ShieldAlert className="h-4 w-4 flex-shrink-0" />
          {currentViolation}
        </div>
      )}

      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="mb-8 animate-fade-in">
          <h1 className="text-4xl font-bold mb-2 flex items-center gap-3">
            <ShieldCheck className="h-9 w-9 text-primary" />
            Verification-First Proctoring
          </h1>
          <p className="text-muted-foreground">
            Two-stage AI monitoring — real-time detectors propose, an agentic VLM verifier disposes.
            Every mark against a candidate carries visual evidence.
          </p>
        </div>

        <div className="grid lg:grid-cols-3 gap-6">
          {/* ─── Left column: camera + stats ─────────────────────────────── */}
          <div className="lg:col-span-2 space-y-6">
            {/* Camera card — no canvas overlay */}
            <Card className="shadow-medium animate-slide-up">
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle className="flex items-center gap-2">
                      <Video className="h-5 w-5" />
                      Live Video Feed
                    </CardTitle>
                    <CardDescription>
                      MediaPipe Face Mesh · head pose · iris gaze · multi-face
                    </CardDescription>
                  </div>
                  {isMonitoring && (
                    <Badge variant="default" className="animate-pulse-glow gap-1.5">
                      <span className="h-2 w-2 rounded-full bg-red-400 inline-block" />
                      LIVE
                    </Badge>
                  )}
                </div>
              </CardHeader>
              <CardContent>
                {/* Clean video feed — detection runs internally, no overlay boxes */}
                <div className="relative aspect-video bg-muted rounded-lg overflow-hidden">
                  <video
                    ref={videoRef}
                    autoPlay
                    playsInline
                    muted
                    className="w-full h-full object-cover"
                    style={{ transform: "scaleX(-1)" }}
                  />

                  {/* Idle state */}
                  {!isMonitoring && !isLoading && (
                    <div className="absolute inset-0 flex flex-col items-center justify-center bg-muted/90">
                      <Video className="h-20 w-20 text-muted-foreground/30 mb-4" />
                      <p className="text-muted-foreground font-medium">Camera inactive</p>
                      <p className="text-muted-foreground/60 text-sm mt-1">
                        Press Start to begin AI monitoring
                      </p>
                    </div>
                  )}

                  {/* Loading */}
                  {isLoading && (
                    <div className="absolute inset-0 flex flex-col items-center justify-center bg-background/80 backdrop-blur-sm">
                      <div className="h-12 w-12 border-4 border-primary border-t-transparent rounded-full animate-spin mb-4" />
                      <p className="text-sm font-medium">Loading AI models…</p>
                      <p className="text-xs text-muted-foreground mt-1">
                        Downloading MediaPipe Face Mesh
                      </p>
                    </div>
                  )}

                  {/* Minimal corner badge — trust score only */}
                  {isMonitoring && (
                    <div className="absolute top-3 left-3">
                      <Badge variant="secondary" className="bg-black/60 text-white border-0">
                        Trust: {trustScore}%
                      </Badge>
                    </div>
                  )}

                  {/* Red pulse border on violation */}
                  {currentViolation && (
                    <div className="absolute inset-0 border-4 border-red-500 rounded-lg pointer-events-none animate-pulse" />
                  )}
                </div>

                {/* Agentic verification consent */}
                <div className="mt-4 flex items-start justify-between gap-3 p-3 rounded-lg border bg-muted/40">
                  <div className="flex items-start gap-2 min-w-0">
                    <Sparkles className="h-4 w-4 text-primary mt-0.5 flex-shrink-0" />
                    <div className="min-w-0">
                      <p className="text-sm font-medium leading-tight">Agentic flag verification</p>
                      <p className="text-xs text-muted-foreground mt-0.5 leading-snug">
                        High-severity flags are fact-checked by an AI verifier before any trust
                        penalty applies. Sends the single flagged frame (never the video stream)
                        to the AI backend. Turn off for fully-offline monitoring.
                      </p>
                    </div>
                  </div>
                  <Switch
                    checked={verifyEnabled}
                    onCheckedChange={setVerifyEnabled}
                    disabled={isMonitoring}
                    aria-label="Toggle agentic flag verification"
                  />
                </div>

                {/* Controls */}
                <div className="mt-4 flex flex-wrap gap-2">
                  {!isMonitoring ? (
                    <Button
                      onClick={startMonitoring}
                      disabled={isLoading}
                      size="lg"
                      className="flex-1 min-w-36"
                    >
                      <Play className="mr-2 h-4 w-4" />
                      {isLoading ? "Initializing…" : "Start Proctoring"}
                    </Button>
                  ) : (
                    <Button
                      onClick={stopMonitoring}
                      variant="destructive"
                      size="lg"
                      className="flex-1 min-w-36"
                    >
                      <Square className="mr-2 h-4 w-4" />
                      Stop Session
                    </Button>
                  )}
                  <Button onClick={exportCSV} variant="outline" disabled={!hasData} title="Download CSV">
                    <Download className="mr-2 h-4 w-4" />
                    Export CSV
                  </Button>
                  <Button onClick={exportJSON} variant="outline" disabled={!hasData} title="Download JSON">
                    <FileJson className="mr-2 h-4 w-4" />
                    Export JSON
                  </Button>
                </div>
              </CardContent>
            </Card>

            {/* Primary stats */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 animate-slide-up" style={{ animationDelay: "0.1s" }}>
              <StatCard title="Session Duration" value={formatTime(sessionTime)} />
              <StatCard
                title="Trust Score"
                value={`${trustScore}%`}
                valueClass={trustColor}
                sub={<Progress value={trustScore} className="mt-2 h-1.5" />}
              />
              <StatCard title="Head Turns" value={stats.headTurns} warn={stats.headTurns > 3} />
              <StatCard title="Total Violations" value={totalViolations} warn={totalViolations > 5} />
            </div>

            {/* Secondary stats */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 animate-slide-up" style={{ animationDelay: "0.15s" }}>
              <StatCard title="Gaze Away" value={stats.gazeAways} />
              <StatCard title="No Face" value={stats.noFaceEvents} warn={stats.noFaceEvents > 2} />
              <StatCard title="Looking Down" value={stats.lookingDownEvents} warn={stats.lookingDownEvents > 2} />
              <StatCard title="Tab Switches" value={stats.tabSwitches} warn={stats.tabSwitches > 0} />
              <StatCard title="Phone Detected" value={stats.phoneDetectedEvents} warn={stats.phoneDetectedEvents > 0} />
              <StatCard title="New Objects" value={stats.newObjectEvents} warn={stats.newObjectEvents > 0} />
              <StatCard
                title="Dismissed by AI"
                value={stats.dismissedFlags}
                valueClass="text-green-500"
                sub={<p className="text-xs text-muted-foreground mt-1">false flags, no penalty</p>}
              />
            </div>
          </div>

          {/* ─── Right column: live status + alert log ───────────────────── */}
          <div className="space-y-4">
            {/* Live proctoring status — replaces canvas overlay */}
            <div className="animate-slide-up" style={{ animationDelay: "0.1s" }}>
              <LiveStatusPanel status={liveStatus} isMonitoring={isMonitoring} phoneDetectorStatus={phoneDetectorStatus} />
            </div>

            {/* Alert log */}
            <Card className="shadow-medium animate-slide-up" style={{ animationDelay: "0.2s" }}>
              <CardHeader className="pb-3">
                <CardTitle className="flex items-center gap-2">
                  <AlertTriangle className="h-5 w-5" />
                  Activity Log
                </CardTitle>
                <CardDescription>
                  {alerts.length} event{alerts.length !== 1 ? "s" : ""} recorded
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-2 max-h-[440px] overflow-y-auto pr-0.5">
                  {alerts.length === 0 ? (
                    <div className="text-center py-8 text-muted-foreground">
                      <CheckCircle className="h-10 w-10 mx-auto mb-3 opacity-40" />
                      <p className="text-sm font-medium">No events yet</p>
                      <p className="text-xs mt-1">Start monitoring to track activity</p>
                    </div>
                  ) : (
                    alerts.map((alert) => <AlertItem key={alert.id} alert={alert} />)
                  )}
                </div>
              </CardContent>
            </Card>
          </div>
        </div>

        {/* Detection legend */}
        <div className="mt-8 animate-slide-up" style={{ animationDelay: "0.25s" }}>
          <Card className="shadow-soft">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-semibold">Detection Capabilities</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-3 text-xs text-center">
                {[
                  { icon: <ArrowLeft className="h-5 w-5" />, label: "Head Turns", desc: "Yaw angle" },
                  { icon: <EyeOff className="h-5 w-5" />,   label: "Gaze Away",  desc: "Iris tracking" },
                  { icon: <ArrowDown className="h-5 w-5" />,    label: "Head Down",   desc: "Face compression" },
                  { icon: <Smartphone className="h-5 w-5" />,  label: "Phone Seen",  desc: "COCO-SSD AI" },
                  { icon: <UserX className="h-5 w-5" />,    label: "No Face",    desc: "Left frame" },
                  { icon: <Users className="h-5 w-5" />,    label: "Multi-Face", desc: "Extra person" },
                  { icon: <ExternalLink className="h-5 w-5" />, label: "Tab Switch", desc: "Focus loss" },
                  { icon: <Package className="h-5 w-5" />, label: "New Object", desc: "Scene baseline" },
                  { icon: <Sparkles className="h-5 w-5" />, label: "AI Verifier", desc: "Fact-checks flags" },
                ].map(({ icon, label, desc }) => (
                  <div key={label} className="flex flex-col items-center gap-1 p-2 rounded-lg bg-muted/50">
                    <span className="text-primary">{icon}</span>
                    <span className="font-semibold">{label}</span>
                    <span className="text-muted-foreground">{desc}</span>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default Proctoring;
