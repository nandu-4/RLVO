import { Link, useLocation } from "react-router-dom";
import { Brain, Image, Video, Home } from "lucide-react";
import { Button } from "@/components/ui/button";

const Navigation = () => {
  const location = useLocation();

  const isActive = (path: string) => location.pathname === path;

  return (
    <nav className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-50">
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          <Link to="/" className="flex items-center gap-2 group">
            <div className="p-2 rounded-lg bg-gradient-primary">
              <Brain className="h-6 w-6 text-primary-foreground" />
            </div>
            <span className="text-2xl font-bold bg-gradient-primary bg-clip-text text-transparent">
              RLVO
            </span>
          </Link>

          <div className="flex items-center gap-2">
            <Button
              variant={isActive("/") ? "default" : "ghost"}
              asChild
              size="sm"
            >
              <Link to="/" className="flex items-center gap-2">
                <Home className="h-4 w-4" />
                Home
              </Link>
            </Button>
            <Button
              variant={isActive("/image-refinement") ? "default" : "ghost"}
              asChild
              size="sm"
            >
              <Link to="/image-refinement" className="flex items-center gap-2">
                <Image className="h-4 w-4" />
                Image Refinement
              </Link>
            </Button>
            <Button
              variant={isActive("/proctoring") ? "default" : "ghost"}
              asChild
              size="sm"
            >
              <Link to="/proctoring" className="flex items-center gap-2">
                <Video className="h-4 w-4" />
                Proctoring
              </Link>
            </Button>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navigation;
