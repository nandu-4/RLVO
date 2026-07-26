import { useState } from "react";
import { Upload, Sparkles, CheckCircle2, AlertCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { toast } from "sonner";
import { supabase } from "@/integrations/supabase/client";
import Navigation from "@/components/Navigation";

const ImageRefinement = () => {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [rawCaption, setRawCaption] = useState<string>("");
  const [refinedCaption, setRefinedCaption] = useState<string>("");
  const [isRefining, setIsRefining] = useState(false);
  const [refinementLogs, setRefinementLogs] = useState<string[]>([]);

  const handleImageUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = async (event) => {
        const imageData = event.target?.result as string;
        setSelectedImage(imageData);
        setRefinementLogs([`[${new Date().toLocaleTimeString()}] Image uploaded successfully`]);
        
        try {
          setRefinementLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] Generating caption with AI...`]);
          
          const { data, error } = await supabase.functions.invoke('generate-caption', {
            body: { image: imageData }
          });
          
          if (error) throw error;
          
          setRawCaption(data.caption);
          setRefinementLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] Caption generated successfully`]);
          toast.success("Image analyzed with AI");
        } catch (error) {
          console.error('Error generating caption:', error);
          toast.error("Failed to generate caption");
          setRefinementLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] Error: ${error instanceof Error ? error.message : 'Unknown error'}`]);
        }
      };
      reader.readAsDataURL(file);
    }
  };

  const handleRefine = async () => {
    if (!selectedImage || !rawCaption) return;
    
    setIsRefining(true);
    setRefinementLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] Starting agentic refinement...`]);
    
    try {
      const { data, error } = await supabase.functions.invoke('refine-caption', {
        body: { 
          image: selectedImage,
          rawCaption: rawCaption
        }
      });
      
      if (error) throw error;
      
      // Add processing logs
      data.logs.forEach((log: string, index: number) => {
        setTimeout(() => {
          setRefinementLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] ${log}`]);
        }, index * 400);
      });
      
      // Set refined caption after logs
      setTimeout(() => {
        setRefinedCaption(data.refinedCaption);
        setIsRefining(false);
        toast.success("Caption refined with AI!");
      }, data.logs.length * 400 + 200);
      
    } catch (error) {
      console.error('Error refining caption:', error);
      setIsRefining(false);
      toast.error("Failed to refine caption");
      setRefinementLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] Error: ${error instanceof Error ? error.message : 'Unknown error'}`]);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-hero">
      <Navigation />
      
      <div className="container mx-auto px-4 py-8">
        <div className="mb-8 animate-fade-in">
          <h1 className="text-4xl font-bold mb-2">Image Caption Refinement</h1>
          <p className="text-muted-foreground">
            Upload an image to generate and refine captions using our AI-powered alignment system
          </p>
        </div>

        <div className="grid lg:grid-cols-2 gap-6">
          {/* Left Column - Image Upload & Display */}
          <Card className="shadow-medium animate-slide-up">
            <CardHeader>
              <CardTitle>Image Upload</CardTitle>
              <CardDescription>Upload an image to analyze</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {!selectedImage ? (
                  <label className="flex flex-col items-center justify-center w-full h-64 border-2 border-dashed border-border rounded-lg cursor-pointer bg-muted/30 hover:bg-muted/50 transition-colors">
                    <div className="flex flex-col items-center justify-center pt-5 pb-6">
                      <Upload className="w-12 h-12 mb-3 text-muted-foreground" />
                      <p className="mb-2 text-sm text-muted-foreground">
                        <span className="font-semibold">Click to upload</span> or drag and drop
                      </p>
                      <p className="text-xs text-muted-foreground">PNG, JPG or WEBP</p>
                    </div>
                    <input
                      type="file"
                      className="hidden"
                      accept="image/*"
                      onChange={handleImageUpload}
                    />
                  </label>
                ) : (
                  <div className="space-y-4">
                    <img
                      src={selectedImage}
                      alt="Uploaded"
                      className="w-full h-64 object-cover rounded-lg shadow-soft"
                    />
                    <Button
                      variant="outline"
                      onClick={() => {
                        setSelectedImage(null);
                        setRawCaption("");
                        setRefinedCaption("");
                        setRefinementLogs([]);
                      }}
                      className="w-full"
                    >
                      Upload Different Image
                    </Button>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>

          {/* Right Column - Captions & Refinement */}
          <div className="space-y-6">
            <Card className="shadow-medium animate-slide-up" style={{ animationDelay: "0.1s" }}>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  Raw Caption
                  {rawCaption && <Badge variant="secondary">Generated</Badge>}
                </CardTitle>
                <CardDescription>Initial caption from VLM</CardDescription>
              </CardHeader>
              <CardContent>
                {rawCaption ? (
                  <p className="text-sm leading-relaxed p-4 bg-muted/50 rounded-lg">
                    {rawCaption}
                  </p>
                ) : (
                  <p className="text-sm text-muted-foreground italic">
                    Upload an image to generate a caption
                  </p>
                )}
              </CardContent>
            </Card>

            <Card className="shadow-medium animate-slide-up" style={{ animationDelay: "0.2s" }}>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  Refined Caption
                  {refinedCaption && (
                    <Badge variant="default" className="bg-success">
                      <CheckCircle2 className="h-3 w-3 mr-1" />
                      Refined
                    </Badge>
                  )}
                </CardTitle>
                <CardDescription>Enhanced with agentic re-alignment</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {refinedCaption ? (
                  <p className="text-sm leading-relaxed p-4 bg-gradient-card rounded-lg border border-success/20">
                    {refinedCaption}
                  </p>
                ) : (
                  <p className="text-sm text-muted-foreground italic">
                    Click refine to improve the caption
                  </p>
                )}
                
                <Button
                  onClick={handleRefine}
                  disabled={!rawCaption || isRefining}
                  className="w-full"
                  size="lg"
                >
                  {isRefining ? (
                    <>
                      <Sparkles className="mr-2 h-4 w-4 animate-spin" />
                      Refining...
                    </>
                  ) : (
                    <>
                      <Sparkles className="mr-2 h-4 w-4" />
                      Refine Caption
                    </>
                  )}
                </Button>
              </CardContent>
            </Card>
          </div>
        </div>

        {/* Refinement Logs */}
        {refinementLogs.length > 0 && (
          <Card className="mt-6 shadow-medium animate-fade-in">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <AlertCircle className="h-5 w-5" />
                Refinement Logs
              </CardTitle>
              <CardDescription>Real-time processing information</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="bg-card rounded-lg p-4 font-mono text-xs space-y-1 max-h-48 overflow-y-auto border">
                {refinementLogs.map((log, index) => (
                  <div key={index} className="text-muted-foreground">
                    {log}
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
};

export default ImageRefinement;
