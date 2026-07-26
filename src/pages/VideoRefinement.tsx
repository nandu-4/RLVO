import { useState, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { useToast } from "@/hooks/use-toast";
import { Loader2, Video, Clock, Upload } from "lucide-react";
import { invokeAi } from "@/integrations/aiClient";
import Navigation from "@/components/Navigation";

interface CaptionedFrame {
  frameUrl: string;
  caption: string;
  timestamp: number;
}

const VideoRefinement = () => {
  const [selectedVideo, setSelectedVideo] = useState<File | null>(null);
  const [videoPreview, setVideoPreview] = useState<string>("");
  const [mode, setMode] = useState<"summary" | "timecapsule">("summary");
  const [isProcessing, setIsProcessing] = useState(false);
  const [summary, setSummary] = useState<string>("");
  const [timeCapsuleFrames, setTimeCapsuleFrames] = useState<CaptionedFrame[]>([]);
  const videoRef = useRef<HTMLVideoElement>(null);
  const { toast } = useToast();

  const handleVideoSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      if (!file.type.startsWith('video/')) {
        toast({
          title: "Invalid file type",
          description: "Please select a video file",
          variant: "destructive",
        });
        return;
      }
      setSelectedVideo(file);
      setVideoPreview(URL.createObjectURL(file));
      setSummary("");
      setTimeCapsuleFrames([]);
    }
  };

  const extractFrameAtTime = (video: HTMLVideoElement, time: number): Promise<string> => {
    return new Promise((resolve) => {
      video.currentTime = time;
      video.onseeked = () => {
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx?.drawImage(video, 0, 0);
        resolve(canvas.toDataURL('image/jpeg', 0.8));
      };
    });
  };

  const extractVideoFrames = async (video: HTMLVideoElement, numFrames: number = 6): Promise<string[]> => {
    const duration = video.duration;
    const interval = duration / numFrames;
    const frames: string[] = [];

    for (let i = 0; i < numFrames; i++) {
      const time = i * interval;
      const frame = await extractFrameAtTime(video, time);
      frames.push(frame);
    }

    return frames;
  };

  const processVideo = async () => {
    if (!selectedVideo || !videoRef.current) {
      toast({
        title: "No video selected",
        description: "Please select a video file first",
        variant: "destructive",
      });
      return;
    }

    setIsProcessing(true);
    setSummary("");
    setTimeCapsuleFrames([]);

    try {
      const video = videoRef.current;

      await new Promise<void>((resolve) => {
        if (video.readyState >= 2) {
          resolve();
        } else {
          video.onloadedmetadata = () => resolve();
        }
      });

      if (mode === "summary") {
        const frames = await extractVideoFrames(video, 6);

        const { data, error } = await invokeAi<{ summary: string }>('analyze-video', {
          frames,
          mode: 'summary',
        });

        if (error || !data) throw error ?? new Error("No data");
        setSummary(data.summary);

        toast({
          title: "Video analyzed successfully",
          description: "Summary generated",
        });
      } else {
        const numFrames = 8;
        const frames = await extractVideoFrames(video, numFrames);
        const duration = video.duration;
        const interval = duration / numFrames;

        const { data, error } = await invokeAi<{ captions: string[] }>('analyze-video', {
          frames,
          mode: 'timecapsule',
        });

        if (error || !data) throw error ?? new Error("No data");

        const captionedFrames: CaptionedFrame[] = data.captions.map((caption: string, index: number) => ({
          frameUrl: frames[index],
          caption,
          timestamp: index * interval
        }));

        setTimeCapsuleFrames(captionedFrames);

        toast({
          title: "Time capsule created",
          description: `${numFrames} frames analyzed`,
        });
      }
    } catch (error) {
      console.error('Error processing video:', error);
      toast({
        title: "Processing failed",
        description: error instanceof Error ? error.message : "Failed to process video",
        variant: "destructive",
      });
    } finally {
      setIsProcessing(false);
    }
  };

  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="min-h-screen bg-background">
      <Navigation />

      <main className="container mx-auto px-4 py-8">
        <div className="max-w-5xl mx-auto space-y-8">
          <div className="text-center space-y-2">
            <h1 className="text-4xl font-bold tracking-tight">Video Understanding</h1>
            <p className="text-muted-foreground">
              Analyze videos and generate descriptions using AI
            </p>
          </div>

          <Card>
            <CardHeader>
              <CardTitle>Upload Video</CardTitle>
              <CardDescription>
                Select a video file to analyze
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-center w-full">
                <label className="flex flex-col items-center justify-center w-full h-64 border-2 border-dashed rounded-lg cursor-pointer hover:bg-accent/50 transition-colors">
                  {videoPreview ? (
                    <video
                      ref={videoRef}
                      src={videoPreview}
                      className="max-h-60 rounded"
                      controls
                    />
                  ) : (
                    <div className="flex flex-col items-center justify-center pt-5 pb-6">
                      <Upload className="w-10 h-10 mb-3 text-muted-foreground" />
                      <p className="mb-2 text-sm text-muted-foreground">
                        <span className="font-semibold">Click to upload</span> or drag and drop
                      </p>
                      <p className="text-xs text-muted-foreground">MP4, WebM, or MOV</p>
                    </div>
                  )}
                  <input
                    type="file"
                    className="hidden"
                    accept="video/*"
                    onChange={handleVideoSelect}
                  />
                </label>
              </div>

              <div className="flex gap-4">
                <Button
                  variant={mode === "summary" ? "default" : "outline"}
                  onClick={() => setMode("summary")}
                  className="flex-1"
                >
                  <Video className="w-4 h-4 mr-2" />
                  Video Summary
                </Button>
                <Button
                  variant={mode === "timecapsule" ? "default" : "outline"}
                  onClick={() => setMode("timecapsule")}
                  className="flex-1"
                >
                  <Clock className="w-4 h-4 mr-2" />
                  Time Capsule
                </Button>
              </div>

              <Button
                onClick={processVideo}
                disabled={!selectedVideo || isProcessing}
                className="w-full"
                size="lg"
              >
                {isProcessing ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    Processing...
                  </>
                ) : (
                  `Generate ${mode === "summary" ? "Summary" : "Time Capsule"}`
                )}
              </Button>
            </CardContent>
          </Card>

          {summary && (
            <Card>
              <CardHeader>
                <CardTitle>Video Summary</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-lg leading-relaxed">{summary}</p>
              </CardContent>
            </Card>
          )}

          {timeCapsuleFrames.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle>Time Capsule Timeline</CardTitle>
                <CardDescription>
                  Key moments from your video
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-6">
                  {timeCapsuleFrames.map((frame, index) => (
                    <div key={index} className="space-y-2">
                      <div className="flex items-center gap-2 text-sm text-muted-foreground">
                        <Clock className="w-4 h-4" />
                        <span>{formatTime(frame.timestamp)}</span>
                      </div>
                      <div className="rounded-lg overflow-hidden border">
                        <img
                          src={frame.frameUrl}
                          alt={`Frame ${index + 1}`}
                          className="w-full"
                        />
                      </div>
                      <p className="text-sm leading-relaxed bg-accent/50 p-3 rounded">
                        {frame.caption}
                      </p>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </main>
    </div>
  );
};

export default VideoRefinement;
