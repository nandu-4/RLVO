import { Link } from "react-router-dom";
import { Brain, Image, Video, CheckCircle, Zap, Shield, ArrowRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import Navigation from "@/components/Navigation";

const Index = () => {
  return (
    <div className="min-h-screen bg-gradient-hero">
      <Navigation />
      
      {/* Hero Section */}
      <section className="container mx-auto px-4 py-20 animate-fade-in">
        <div className="max-w-4xl mx-auto text-center space-y-6">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-accent/10 border border-accent/20 mb-4">
            <Brain className="h-4 w-4 text-accent" />
            <span className="text-sm font-medium">Agentic Re-alignment for Vision-Language</span>
          </div>
          
          <h1 className="text-5xl md:text-6xl font-bold leading-tight">
            Reduce AI Hallucinations with{" "}
            <span className="bg-gradient-primary bg-clip-text text-transparent">
              RLVO
            </span>
          </h1>
          
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            AI-powered vision-language alignment system that improves caption accuracy by 60% 
            and provides real-time exam proctoring with advanced behavior detection.
          </p>
          
          <div className="flex flex-wrap gap-4 justify-center pt-4">
            <Button asChild size="lg" className="shadow-glow">
              <Link to="/image-refinement" className="flex items-center gap-2">
                <Image className="h-5 w-5" />
                Try Image Refinement
                <ArrowRight className="h-4 w-4" />
              </Link>
            </Button>
            <Button asChild variant="outline" size="lg">
              <Link to="/proctoring" className="flex items-center gap-2">
                <Video className="h-5 w-5" />
                View Proctoring Dashboard
              </Link>
            </Button>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="container mx-auto px-4 py-16">
        <div className="text-center mb-12 animate-slide-up">
          <h2 className="text-3xl md:text-4xl font-bold mb-4">Key Features</h2>
          <p className="text-muted-foreground max-w-2xl mx-auto">
            Powerful AI capabilities designed to ensure accurate visual-language alignment and secure online examinations
          </p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6 max-w-6xl mx-auto">
          <Card className="shadow-medium animate-slide-up hover:shadow-strong transition-shadow" style={{ animationDelay: "0.1s" }}>
            <CardHeader>
              <div className="h-12 w-12 rounded-lg bg-gradient-primary flex items-center justify-center mb-4">
                <Image className="h-6 w-6 text-primary-foreground" />
              </div>
              <CardTitle>Image Caption Refinement</CardTitle>
              <CardDescription>
                Upload images and refine captions through our LLM-driven agentic workflow
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ul className="space-y-2 text-sm">
                <li className="flex items-start gap-2">
                  <CheckCircle className="h-4 w-4 text-success mt-0.5 flex-shrink-0" />
                  <span>60% improvement in alignment accuracy</span>
                </li>
                <li className="flex items-start gap-2">
                  <CheckCircle className="h-4 w-4 text-success mt-0.5 flex-shrink-0" />
                  <span>Multi-cycle refinement process</span>
                </li>
                <li className="flex items-start gap-2">
                  <CheckCircle className="h-4 w-4 text-success mt-0.5 flex-shrink-0" />
                  <span>Real-time processing logs</span>
                </li>
              </ul>
            </CardContent>
          </Card>

          <Card className="shadow-medium animate-slide-up hover:shadow-strong transition-shadow" style={{ animationDelay: "0.2s" }}>
            <CardHeader>
              <div className="h-12 w-12 rounded-lg bg-gradient-primary flex items-center justify-center mb-4">
                <Video className="h-6 w-6 text-primary-foreground" />
              </div>
              <CardTitle>Video Proctoring</CardTitle>
              <CardDescription>
                Real-time webcam monitoring with AI-powered behavior detection
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ul className="space-y-2 text-sm">
                <li className="flex items-start gap-2">
                  <CheckCircle className="h-4 w-4 text-success mt-0.5 flex-shrink-0" />
                  <span>Head turn detection and tracking</span>
                </li>
                <li className="flex items-start gap-2">
                  <CheckCircle className="h-4 w-4 text-success mt-0.5 flex-shrink-0" />
                  <span>&lt;3s real-time latency</span>
                </li>
                <li className="flex items-start gap-2">
                  <CheckCircle className="h-4 w-4 text-success mt-0.5 flex-shrink-0" />
                  <span>Automated alert system</span>
                </li>
              </ul>
            </CardContent>
          </Card>

          <Card className="shadow-medium animate-slide-up hover:shadow-strong transition-shadow" style={{ animationDelay: "0.3s" }}>
            <CardHeader>
              <div className="h-12 w-12 rounded-lg bg-gradient-primary flex items-center justify-center mb-4">
                <Zap className="h-6 w-6 text-primary-foreground" />
              </div>
              <CardTitle>High Performance</CardTitle>
              <CardDescription>
                Optimized for speed and reliability with enterprise-grade infrastructure
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ul className="space-y-2 text-sm">
                <li className="flex items-start gap-2">
                  <CheckCircle className="h-4 w-4 text-success mt-0.5 flex-shrink-0" />
                  <span>&lt;2.5s refinement latency</span>
                </li>
                <li className="flex items-start gap-2">
                  <CheckCircle className="h-4 w-4 text-success mt-0.5 flex-shrink-0" />
                  <span>100 concurrent users supported</span>
                </li>
                <li className="flex items-start gap-2">
                  <CheckCircle className="h-4 w-4 text-success mt-0.5 flex-shrink-0" />
                  <span>99% uptime guarantee</span>
                </li>
              </ul>
            </CardContent>
          </Card>

          <Card className="shadow-medium animate-slide-up hover:shadow-strong transition-shadow" style={{ animationDelay: "0.4s" }}>
            <CardHeader>
              <div className="h-12 w-12 rounded-lg bg-gradient-primary flex items-center justify-center mb-4">
                <Shield className="h-6 w-6 text-primary-foreground" />
              </div>
              <CardTitle>Privacy & Security</CardTitle>
              <CardDescription>
                GDPR compliant with enterprise-grade encryption and data protection
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ul className="space-y-2 text-sm">
                <li className="flex items-start gap-2">
                  <CheckCircle className="h-4 w-4 text-success mt-0.5 flex-shrink-0" />
                  <span>TLS encryption for all data</span>
                </li>
                <li className="flex items-start gap-2">
                  <CheckCircle className="h-4 w-4 text-success mt-0.5 flex-shrink-0" />
                  <span>Explicit consent required</span>
                </li>
                <li className="flex items-start gap-2">
                  <CheckCircle className="h-4 w-4 text-success mt-0.5 flex-shrink-0" />
                  <span>30-day auto-deletion policy</span>
                </li>
              </ul>
            </CardContent>
          </Card>

          <Card className="shadow-medium animate-slide-up hover:shadow-strong transition-shadow" style={{ animationDelay: "0.5s" }}>
            <CardHeader>
              <div className="h-12 w-12 rounded-lg bg-gradient-primary flex items-center justify-center mb-4">
                <Brain className="h-6 w-6 text-primary-foreground" />
              </div>
              <CardTitle>Advanced AI Models</CardTitle>
              <CardDescription>
                Powered by state-of-the-art vision-language and reflection models
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ul className="space-y-2 text-sm">
                <li className="flex items-start gap-2">
                  <CheckCircle className="h-4 w-4 text-success mt-0.5 flex-shrink-0" />
                  <span>LLaVA-v1.6-34B for captioning</span>
                </li>
                <li className="flex items-start gap-2">
                  <CheckCircle className="h-4 w-4 text-success mt-0.5 flex-shrink-0" />
                  <span>Vicuna-13B for reflection</span>
                </li>
                <li className="flex items-start gap-2">
                  <CheckCircle className="h-4 w-4 text-success mt-0.5 flex-shrink-0" />
                  <span>SigLIP alignment scoring</span>
                </li>
              </ul>
            </CardContent>
          </Card>

          <Card className="shadow-medium animate-slide-up hover:shadow-strong transition-shadow" style={{ animationDelay: "0.6s" }}>
            <CardHeader>
              <div className="h-12 w-12 rounded-lg bg-gradient-primary flex items-center justify-center mb-4">
                <CheckCircle className="h-6 w-6 text-primary-foreground" />
              </div>
              <CardTitle>Reports & Audit Logs</CardTitle>
              <CardDescription>
                Comprehensive reporting and transparent audit trails for compliance
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ul className="space-y-2 text-sm">
                <li className="flex items-start gap-2">
                  <CheckCircle className="h-4 w-4 text-success mt-0.5 flex-shrink-0" />
                  <span>Exportable CSV/JSON reports</span>
                </li>
                <li className="flex items-start gap-2">
                  <CheckCircle className="h-4 w-4 text-success mt-0.5 flex-shrink-0" />
                  <span>Real-time activity logging</span>
                </li>
                <li className="flex items-start gap-2">
                  <CheckCircle className="h-4 w-4 text-success mt-0.5 flex-shrink-0" />
                  <span>Full audit trail access</span>
                </li>
              </ul>
            </CardContent>
          </Card>
        </div>
      </section>

      {/* CTA Section */}
      <section className="container mx-auto px-4 py-16">
        <Card className="max-w-4xl mx-auto shadow-strong bg-gradient-card animate-slide-up">
          <CardContent className="p-12 text-center space-y-6">
            <h2 className="text-3xl md:text-4xl font-bold">Ready to Get Started?</h2>
            <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
              Experience the power of AI-driven vision-language alignment and secure exam proctoring.
            </p>
            <div className="flex flex-wrap gap-4 justify-center pt-4">
              <Button asChild size="lg" className="shadow-glow">
                <Link to="/image-refinement" className="flex items-center gap-2">
                  Start Refining Images
                  <ArrowRight className="h-4 w-4" />
                </Link>
              </Button>
              <Button asChild variant="outline" size="lg">
                <Link to="/proctoring">
                  Launch Proctoring
                </Link>
              </Button>
            </div>
          </CardContent>
        </Card>
      </section>

      {/* Footer */}
      <footer className="border-t border-border mt-16 py-8">
        <div className="container mx-auto px-4 text-center text-sm text-muted-foreground">
          <p>© 2025 RLVO. All rights reserved. Built with cutting-edge AI technology.</p>
        </div>
      </footer>
    </div>
  );
};

export default Index;
