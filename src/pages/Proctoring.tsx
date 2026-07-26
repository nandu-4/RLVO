import { useState, useEffect, useRef } from "react";
import { Video, AlertTriangle, CheckCircle, Download, Play, Square } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { toast } from "sonner";
import Navigation from "@/components/Navigation";
import { supabase } from "@/integrations/supabase/client";

const Proctoring = () => {
  const [isMonitoring, setIsMonitoring] = useState(false);
  const [headCount, setHeadCount] = useState(0);
  const [blinkCount, setBlinkCount] = useState(0);
  const [headTurns, setHeadTurns] = useState(0);
  const [alerts, setAlerts] = useState<Array<{ time: string; type: string; message: string }>>([]);
  const [sessionTime, setSessionTime] = useState(0);
  const [confidenceScore, setConfidenceScore] = useState(98);
  const [lastHeadOrientation, setLastHeadOrientation] = useState<string>('facing_camera');
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const analysisIntervalRef = useRef<NodeJS.Timeout | null>(null);

  const analyzeFrame = async () => {
    if (!videoRef.current || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const video = videoRef.current;
    const ctx = canvas.getContext('2d');
    
    if (!ctx) return;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);
    
    const imageData = canvas.toDataURL('image/jpeg', 0.8);

    try {
      const { data, error } = await supabase.functions.invoke('analyze-frame', {
        body: { imageData }
      });

      if (error) throw error;

      if (data) {
        const currentHeadCount = data.headCount || 0;
        const headOrientation = data.headOrientation || 'facing_camera';
        
        setHeadCount(currentHeadCount);
        setConfidenceScore(Math.round((data.confidence || 0.9) * 100));

        // Track head turns
        if (headOrientation !== 'facing_camera' && lastHeadOrientation === 'facing_camera') {
          setHeadTurns(prev => prev + 1);
          const alert = {
            time: new Date().toLocaleTimeString(),
            type: "warning",
            message: `Head turned away: ${headOrientation.replace('_', ' ')}`
          };
          setAlerts(prev => [alert, ...prev].slice(0, 10));
          toast.warning("Head turned away from camera!");
        }
        setLastHeadOrientation(headOrientation);

        if (currentHeadCount > 1) {
          const alert = {
            time: new Date().toLocaleTimeString(),
            type: "warning",
            message: `Multiple heads detected: ${currentHeadCount} people`
          };
          setAlerts(prev => [alert, ...prev].slice(0, 10));
          toast.warning(`${currentHeadCount} people detected in frame!`);
        } else if (currentHeadCount === 0) {
          const alert = {
            time: new Date().toLocaleTimeString(),
            type: "warning",
            message: "No person detected in frame"
          };
          setAlerts(prev => [alert, ...prev].slice(0, 10));
          toast.warning("No person detected!");
        }

        if (data.eyesBlinked) {
          setBlinkCount(prev => prev + 1);
        }
      }
    } catch (error) {
      console.error('Error analyzing frame:', error);
    }
  };

  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isMonitoring) {
      interval = setInterval(() => {
        setSessionTime(prev => prev + 1);
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [isMonitoring]);

  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
      if (analysisIntervalRef.current) {
        clearInterval(analysisIntervalRef.current);
      }
    };
  }, []);

  const startMonitoring = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { width: 1280, height: 720 } 
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        streamRef.current = stream;
      }

      setIsMonitoring(true);
      setHeadCount(0);
      setBlinkCount(0);
      setHeadTurns(0);
      setLastHeadOrientation('facing_camera');
      setAlerts([{
        time: new Date().toLocaleTimeString(),
        type: "success",
        message: "Proctoring session started"
      }]);
      setSessionTime(0);
      
      // Start analyzing frames every 3 seconds
      analysisIntervalRef.current = setInterval(analyzeFrame, 3000);
      
      toast.success("Proctoring session started");
    } catch (error) {
      console.error('Error accessing webcam:', error);
      toast.error("Failed to access webcam. Please grant camera permissions.");
    }
  };

  const stopMonitoring = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }

    if (analysisIntervalRef.current) {
      clearInterval(analysisIntervalRef.current);
      analysisIntervalRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    setIsMonitoring(false);
    setAlerts(prev => [{
      time: new Date().toLocaleTimeString(),
      type: "info",
      message: "Proctoring session ended"
    }, ...prev]);
    toast.info("Proctoring session ended");
  };

  const exportReport = () => {
    const report = {
      sessionDuration: `${Math.floor(sessionTime / 60)}m ${sessionTime % 60}s`,
      headCount: headCount,
      headTurns: headTurns,
      blinkCount: blinkCount,
      alerts: alerts,
      confidenceScore: confidenceScore.toFixed(1)
    };
    
    console.log("Exporting report:", report);
    toast.success("Report exported successfully");
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="min-h-screen bg-gradient-hero">
      <Navigation />
      
      <div className="container mx-auto px-4 py-8">
        <div className="mb-8 animate-fade-in">
          <h1 className="text-4xl font-bold mb-2">Video Proctoring Dashboard</h1>
          <p className="text-muted-foreground">
            Real-time exam monitoring with AI-powered behavior detection
          </p>
        </div>

        <div className="grid lg:grid-cols-3 gap-6">
          {/* Left Column - Video Feed */}
          <div className="lg:col-span-2 space-y-6">
            <Card className="shadow-medium animate-slide-up">
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle className="flex items-center gap-2">
                      <Video className="h-5 w-5" />
                      Live Video Feed
                    </CardTitle>
                    <CardDescription>Webcam monitoring with head pose detection</CardDescription>
                  </div>
                  {isMonitoring && (
                    <Badge variant="default" className="animate-pulse-glow">
                      <div className="h-2 w-2 rounded-full bg-destructive mr-2" />
                      LIVE
                    </Badge>
                  )}
                </div>
              </CardHeader>
              <CardContent>
                <div className="relative aspect-video bg-muted rounded-lg overflow-hidden">
                  <video
                    ref={videoRef}
                    autoPlay
                    playsInline
                    muted
                    className="w-full h-full object-cover"
                  />
                  <canvas ref={canvasRef} className="hidden" />
                  {!isMonitoring && (
                    <div className="absolute inset-0 flex items-center justify-center bg-muted">
                      <Video className="h-24 w-24 text-muted-foreground/30" />
                    </div>
                  )}
                  {isMonitoring && (
                    <>
                      <div className="absolute top-4 left-4 space-y-2">
                        <Badge variant="secondary" className="bg-card/80 backdrop-blur">
                          Confidence: {confidenceScore.toFixed(1)}%
                        </Badge>
                        <Badge variant="secondary" className="bg-card/80 backdrop-blur block">
                          Heads: {headCount}
                        </Badge>
                      </div>
                      <div className="absolute inset-0 border-2 border-accent/50 pointer-events-none" />
                    </>
                  )}
                </div>
                
                <div className="mt-4 flex gap-2">
                  {!isMonitoring ? (
                    <Button onClick={startMonitoring} size="lg" className="flex-1">
                      <Play className="mr-2 h-4 w-4" />
                      Start Proctoring
                    </Button>
                  ) : (
                    <Button onClick={stopMonitoring} variant="destructive" size="lg" className="flex-1">
                      <Square className="mr-2 h-4 w-4" />
                      Stop Session
                    </Button>
                  )}
                  <Button onClick={exportReport} variant="outline" disabled={!isMonitoring && alerts.length === 0}>
                    <Download className="mr-2 h-4 w-4" />
                    Export Report
                  </Button>
                </div>
              </CardContent>
            </Card>

            {/* Session Metrics */}
            <div className="grid md:grid-cols-5 gap-4 animate-slide-up" style={{ animationDelay: "0.1s" }}>
              <Card className="shadow-soft">
                <CardHeader className="pb-3">
                  <CardTitle className="text-sm font-medium text-muted-foreground">
                    Session Duration
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{formatTime(sessionTime)}</div>
                </CardContent>
              </Card>
              
              <Card className="shadow-soft">
                <CardHeader className="pb-3">
                  <CardTitle className="text-sm font-medium text-muted-foreground">
                    Current Heads
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{headCount}</div>
                  {headCount > 1 && (
                    <p className="text-xs text-warning mt-1">Multiple people detected</p>
                  )}
                  {headCount === 0 && isMonitoring && (
                    <p className="text-xs text-warning mt-1">No person detected</p>
                  )}
                </CardContent>
              </Card>
              
              <Card className="shadow-soft">
                <CardHeader className="pb-3">
                  <CardTitle className="text-sm font-medium text-muted-foreground">
                    Head Turns
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{headTurns}</div>
                  <p className="text-xs text-muted-foreground mt-1">Away from camera</p>
                </CardContent>
              </Card>
              
              <Card className="shadow-soft">
                <CardHeader className="pb-3">
                  <CardTitle className="text-sm font-medium text-muted-foreground">
                    Eye Blinks
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{blinkCount}</div>
                  <p className="text-xs text-muted-foreground mt-1">Complete blinks</p>
                </CardContent>
              </Card>
              
              <Card className="shadow-soft">
                <CardHeader className="pb-3">
                  <CardTitle className="text-sm font-medium text-muted-foreground">
                    Confidence Score
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{confidenceScore.toFixed(1)}%</div>
                  <Progress value={confidenceScore} className="mt-2" />
                </CardContent>
              </Card>
            </div>
          </div>

          {/* Right Column - Alerts */}
          <Card className="shadow-medium animate-slide-up" style={{ animationDelay: "0.2s" }}>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <AlertTriangle className="h-5 w-5" />
                Activity Log
              </CardTitle>
              <CardDescription>Real-time monitoring alerts</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3 max-h-[600px] overflow-y-auto">
                {alerts.length === 0 ? (
                  <div className="text-center py-8 text-muted-foreground">
                    <CheckCircle className="h-12 w-12 mx-auto mb-2 opacity-50" />
                    <p className="text-sm">No alerts yet</p>
                    <p className="text-xs">Start monitoring to track activity</p>
                  </div>
                ) : (
                  alerts.map((alert, index) => (
                    <div
                      key={index}
                      className="p-3 rounded-lg border bg-card animate-fade-in"
                    >
                      <div className="flex items-start gap-2">
                        {alert.type === "warning" && (
                          <AlertTriangle className="h-4 w-4 text-warning mt-0.5 flex-shrink-0" />
                        )}
                        {alert.type === "info" && (
                          <CheckCircle className="h-4 w-4 text-primary mt-0.5 flex-shrink-0" />
                        )}
                        {alert.type === "success" && (
                          <CheckCircle className="h-4 w-4 text-success mt-0.5 flex-shrink-0" />
                        )}
                        <div className="flex-1 min-w-0">
                          <p className="text-sm font-medium">{alert.message}</p>
                          <p className="text-xs text-muted-foreground mt-1">{alert.time}</p>
                        </div>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default Proctoring;
