import { useState, useEffect } from "react";
import { Link, useLocation } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import { Shield, Menu, X, LayoutDashboard, Upload, Settings, Image, Film, Video, Scale, ClipboardList, History } from "lucide-react";
import { cn } from "@/lib/utils";
import AccountMenu from "./AccountMenu";

const navItems = [
  { label: "Home", href: "/" },
  { label: "Verify", href: "/verify", icon: <Upload className="w-4 h-4" /> },
  { label: "Review", href: "/review", icon: <ClipboardList className="w-4 h-4" /> },
  { label: "Dashboard", href: "/dashboard", icon: <LayoutDashboard className="w-4 h-4" /> },
  { label: "History", href: "/history", icon: <History className="w-4 h-4" /> },
  { label: "Benchmark", href: "/benchmark", icon: <Scale className="w-4 h-4" /> },
  { label: "Admin", href: "/admin", icon: <Settings className="w-4 h-4" /> },
];

const labItems = [
  { label: "Image RLVO", href: "/image-refinement", icon: <Image className="w-3.5 h-3.5" /> },
  { label: "Video RLVO", href: "/video-refinement", icon: <Film className="w-3.5 h-3.5" /> },
  { label: "Proctoring", href: "/proctoring", icon: <Video className="w-3.5 h-3.5" /> },
];

export default function TruthLensNavbar() {
  const location = useLocation();
  const [scrolled, setScrolled] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);

  useEffect(() => {
    const handleScroll = () => setScrolled(window.scrollY > 20);
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  return (
    <header
      className={cn(
        "fixed top-0 left-0 right-0 z-50 transition-all duration-500",
        scrolled ? "glass-strong py-3" : "py-4",
      )}
    >
      <a href="#main" className="skip-link">Skip to main content</a>
      <div className="max-w-7xl mx-auto px-6 flex items-center justify-between">
        {/* Logo */}
        <Link to="/" className="flex items-center gap-2.5 group">
          <div className="relative">
            <Shield className="w-8 h-8 text-primary transition-transform duration-300 group-hover:scale-110" />
            <div className="absolute inset-0 bg-primary/20 blur-xl rounded-full opacity-0 group-hover:opacity-100 transition-opacity" />
          </div>
          <span className="text-xl font-bold tracking-tight">
            <span className="gradient-text">Truth</span>
            <span className="text-foreground">Lens</span>
            <span className="text-muted-foreground text-xs font-normal ml-1.5 px-2 py-0.5 rounded-full glass-light border border-border">AI Enterprise</span>
          </span>
        </Link>

        {/* Desktop Nav */}
        <nav aria-label="Primary" className="hidden md:flex items-center gap-1">
          {navItems.map((item) => {
            const isActive = location.pathname === item.href;
            return (
              <Link
                key={item.href}
                to={item.href}
                className={cn(
                  "relative px-4 py-2 rounded-lg text-sm font-medium transition-all duration-300 flex items-center gap-2",
                  isActive
                    ? "text-white bg-primary/15 border border-primary/30"
                    : "text-muted-foreground hover:text-foreground hover:bg-surface-light",
                )}
                aria-current={isActive ? "page" : undefined}
              >
                <span className="relative z-10 flex items-center gap-2">
                  {item.icon}
                  {item.label}
                </span>
              </Link>
            );
          })}

          {/* RLVO research tools — a separate project, kept visible and clearly grouped. */}
          <span className="h-4 w-px bg-border mx-2" aria-hidden="true" />
          {labItems.map((item) => {
            const isActive = location.pathname === item.href;
            return (
              <Link
                key={item.href}
                to={item.href}
                aria-current={isActive ? "page" : undefined}
                className={cn(
                  "px-3 py-1.5 rounded-lg text-xs font-medium transition-all flex items-center gap-1.5",
                  isActive
                    ? "text-accent bg-accent/15 border border-accent/30"
                    : "text-muted-foreground hover:text-foreground hover:bg-surface-light",
                )}
              >
                {item.icon}
                {item.label}
              </Link>
            );
          })}
        </nav>

        {/* Account + CTA */}
        <div className="hidden md:flex items-center gap-3">
          <AccountMenu />
        </div>

        {/* Mobile Toggle */}
        <button
          className="md:hidden text-foreground p-2"
          onClick={() => setMobileOpen(!mobileOpen)}
          aria-label="Toggle navigation menu"
          aria-expanded={mobileOpen}
          aria-controls="mobile-nav"
        >
          {mobileOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
        </button>
      </div>

      {/* Mobile Menu */}
      <AnimatePresence>
        {mobileOpen && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            id="mobile-nav"
            className="md:hidden glass-strong border-t border-border mt-2"
          >
            <div className="px-6 py-4 flex flex-col gap-2">
              {navItems.map((item) => (
                <Link
                  key={item.href}
                  to={item.href}
                  onClick={() => setMobileOpen(false)}
                  className={cn(
                    "px-4 py-3 rounded-lg text-sm font-medium transition-colors flex items-center gap-3",
                    location.pathname === item.href
                      ? "bg-primary/15 text-white"
                      : "text-muted-foreground hover:text-foreground hover:bg-surface-light",
                  )}
                >
                  {item.icon}
                  {item.label}
                </Link>
              ))}
              <div className="h-px bg-border my-2" />
              <div className="text-xs font-semibold text-muted-foreground uppercase px-4 mb-1">Research demos (separate project)</div>
              {labItems.map((item) => (
                <Link
                  key={item.href}
                  to={item.href}
                  onClick={() => setMobileOpen(false)}
                  className="px-4 py-2 rounded-lg text-xs font-medium text-muted-foreground hover:text-foreground flex items-center gap-2"
                >
                  {item.icon}
                  {item.label}
                </Link>
              ))}
              <Link
                to="/verify"
                onClick={() => setMobileOpen(false)}
                className="btn-primary text-sm text-center mt-3 relative z-10"
              >
                Verify Document →
              </Link>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </header>
  );
}
