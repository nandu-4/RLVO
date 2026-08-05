import { motion, type Variants } from "framer-motion";
import { Link } from "react-router-dom";
import {
  Shield,
  ArrowRight,
  FileText,
  Brain,
  Search,
  CheckCircle2,
  AlertTriangle,
  Sparkles,
  Eye,
  BarChart3,
  Lock,
  Zap,
  Target,
  Layers,
} from "lucide-react";
import GlassCard from "./GlassCard";

/* ── Fade-in animation wrapper ── */
const fadeUp: Variants = {
  hidden: { opacity: 0, y: 30 },
  visible: (i: number = 0) => ({
    opacity: 1,
    y: 0,
    transition: { delay: i * 0.1, duration: 0.6, ease: "easeOut" },
  }),
};

const stagger: Variants = {
  visible: { transition: { staggerChildren: 0.08 } },
};

/* ══════════════════════════════════════════════════════
   HERO
   ══════════════════════════════════════════════════════ */
export function Hero() {
  return (
    <section className="relative min-h-screen flex items-center justify-center section-padding pt-32">
      <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-primary/10 rounded-full blur-3xl animate-pulse-glow" />
      <div className="absolute bottom-1/4 right-1/4 w-80 h-80 bg-secondary/10 rounded-full blur-3xl animate-pulse-glow" style={{ animationDelay: "1.5s" }} />

      <div className="relative z-10 max-w-5xl mx-auto text-center">
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5 }}
          className="inline-flex items-center gap-2 glass rounded-full px-4 py-2 mb-8"
        >
          <Sparkles className="w-4 h-4 text-accent" />
          <span className="text-xs md:text-sm text-muted-foreground">
            Inspired by Real-LOD (ICLR 2025) · Agentic Verification Platform
          </span>
        </motion.div>

        <motion.h1
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.15, duration: 0.7 }}
          className="text-5xl md:text-7xl lg:text-8xl font-bold leading-[1.05] tracking-tight mb-6"
        >
          Can You Trust
          <br />
          <span className="gradient-text">What AI Sees?</span>
        </motion.h1>

        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3, duration: 0.7 }}
          className="text-lg md:text-xl text-muted-foreground max-w-2xl mx-auto mb-10 leading-relaxed"
        >
          Every AI-generated visual answer should be{" "}
          <span className="text-foreground font-medium">verified</span> before humans make
          decisions. TruthLens detects hallucinations, provides{" "}
          <span className="text-foreground font-medium">visual evidence</span>, and delivers{" "}
          <span className="text-foreground font-medium">confidence scores</span> you can trust.
        </motion.p>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.45, duration: 0.7 }}
          className="flex flex-col sm:flex-row items-center justify-center gap-4 mb-16"
        >
          <Link to="/verify" className="btn-primary text-base py-3.5 px-8 flex items-center gap-2 relative z-10">
            Try Verification Demo <ArrowRight className="w-4 h-4" />
          </Link>
          <Link to="/dashboard" className="btn-secondary flex items-center gap-2">
            <BarChart3 className="w-4 h-4" /> Enterprise Dashboard
          </Link>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6, duration: 0.8 }}
        >
          <VerificationFlowAnimation />
        </motion.div>
      </div>
    </section>
  );
}

function VerificationFlowAnimation() {
  // Mirrors the stages the server measures and returns; see PipelineTimeline.
  const steps = [
    { icon: <FileText className="w-5 h-5" />, label: "Intake", color: "text-muted-foreground" },
    { icon: <Brain className="w-5 h-5" />, label: "Vision Provider", color: "text-primary" },
    { icon: <Search className="w-5 h-5" />, label: "Evidence Retrieval", color: "text-warning" },
    { icon: <Eye className="w-5 h-5" />, label: "Trust Scoring", color: "text-accent" },
    { icon: <CheckCircle2 className="w-5 h-5" />, label: "Audited Decision", color: "text-success" },
  ];

  return (
    <div className="glass rounded-2xl p-6 md:p-8 max-w-3xl mx-auto">
      <div className="flex items-center justify-between gap-2 md:gap-4">
        {steps.map((step, i) => (
          <motion.div
            key={step.label}
            initial={{ opacity: 0, scale: 0.5 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.8 + i * 0.15, type: "spring", stiffness: 200 }}
            className="flex flex-col items-center gap-2 flex-1"
          >
            <div
              className={`w-12 h-12 md:w-14 md:h-14 rounded-xl glass-light flex items-center justify-center ${step.color}`}
            >
              {step.icon}
            </div>
            <span className="text-xs md:text-sm text-muted-foreground font-medium">
              {step.label}
            </span>
          </motion.div>
        ))}
      </div>

      {/* Illustrative only — labelled as such so nobody reads these numbers as measured results. */}
      <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-3 text-left">
        <div className="glass-light rounded-xl p-4">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <AlertTriangle className="w-4 h-4 text-warning" />
              <span className="text-xs font-semibold text-warning">AI claim contradicted by evidence</span>
            </div>
            <span className="text-[10px] px-2 py-0.5 rounded bg-warning/20 text-warning font-bold">CORRECTED</span>
          </div>
          <p className="text-xs text-muted-foreground font-mono">
            Vendor: <span className="line-through text-danger/70">Microsoft</span> → <span className="text-success font-bold">Oracle Corporation</span>
          </p>
          <p className="text-[10px] text-muted-foreground mt-1">Header text on page 1 disagrees with the model's answer.</p>
        </div>

        <div className="glass-light rounded-xl p-4">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 text-success" />
              <span className="text-xs font-semibold text-success">AI claim supported by evidence</span>
            </div>
            <span className="text-[10px] px-2 py-0.5 rounded bg-success/20 text-success font-bold">VERIFIED</span>
          </div>
          <p className="text-xs text-foreground font-mono">
            Jurisdiction: <span className="text-success font-semibold">State of Delaware</span>
          </p>
          <p className="text-[10px] text-muted-foreground mt-1">Matched to clause text with page and bounding-box provenance.</p>
        </div>
      </div>

      <p className="mt-3 text-[10px] text-muted-foreground/70 text-center">
        Illustrative examples of the two decision paths — not results from a benchmark run.
      </p>
    </div>
  );
}

export function HowItWorks() {
  const steps = [
    {
      num: "01",
      title: "Upload the source document",
      desc: "Any document type — resume, contract, medical report, diagram, financial PDF, purchase order. No schema is assumed and no field is hardcoded.",
      icon: <FileText className="w-6 h-6" />,
    },
    {
      num: "02",
      title: "Supply the AI's claims",
      desc: "Paste the statements your model produced about that document. TruthLens verifies what an AI already said — it does not generate answers of its own.",
      icon: <Brain className="w-6 h-6" />,
    },
    {
      num: "03",
      title: "Evidence retrieval & verification",
      desc: "Each claim is matched to visible document evidence with page provenance. Anything unsupported is returned for human review rather than guessed.",
      icon: <Shield className="w-6 h-6" />,
    },
    {
      num: "04",
      title: "Trust score & audit trail",
      desc: "Verified, corrected, unsupported, or needs-review — each with an explainable signal breakdown, measured stage timings, and an exportable audit record.",
      icon: <Target className="w-6 h-6" />,
    },
  ];

  return (
    <section className="section-padding relative" id="how-it-works">
      <div className="max-w-6xl mx-auto">
        <motion.div
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, amount: 0.3 }}
          variants={stagger}
          className="text-center mb-16"
        >
          <motion.p variants={fadeUp} custom={0} className="text-sm font-semibold text-primary uppercase tracking-wider mb-3">
            How It Works
          </motion.p>
          <motion.h2 variants={fadeUp} custom={1} className="text-4xl md:text-5xl font-bold mb-4">
            Verification in <span className="gradient-text">Four Steps</span>
          </motion.h2>
        </motion.div>

        <motion.div
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, amount: 0.2 }}
          variants={stagger}
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6"
        >
          {steps.map((step, i) => (
            <motion.div key={step.num} variants={fadeUp} custom={i}>
              <GlassCard className="h-full relative overflow-hidden group">
                <div className="text-5xl font-black text-white/10 mb-4 select-none">
                  {step.num}
                </div>
                <div className="w-12 h-12 rounded-xl bg-surface-light flex items-center justify-center text-primary mb-4">
                  {step.icon}
                </div>
                <h3 className="text-lg font-semibold mb-2">{step.title}</h3>
                <p className="text-sm text-muted-foreground leading-relaxed">{step.desc}</p>
              </GlassCard>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </section>
  );
}

export function Features() {
  // Each line below describes behaviour the code actually implements. Unearned compliance and
  // performance claims ("SOC 2 compliant", "sub-3-second") were removed rather than softened —
  // an enterprise buyer treats an unverifiable claim on a marketing page as a diligence finding.
  const features = [
    { icon: <Shield className="w-5 h-5" />, title: "Hallucination Detection", desc: "Every supplied AI claim is checked against evidence retrieved from the uploaded document" },
    { icon: <Eye className="w-5 h-5" />, title: "Evidence Provenance", desc: "Each decision carries page numbers, extracted text, and bounding boxes where available" },
    { icon: <BarChart3 className="w-5 h-5" />, title: "Explainable Scoring", desc: "Trust is broken into OCR, vision, layout, semantic, and evidence-strength signals" },
    { icon: <Layers className="w-5 h-5" />, title: "Comparison Mode", desc: "Side-by-side view of the original AI answer, the evidence, and the verified output" },
    { icon: <Zap className="w-5 h-5" />, title: "Measured Pipeline", desc: "Real per-stage timings are returned with every verification — nothing is simulated" },
    { icon: <Lock className="w-5 h-5" />, title: "No Sign-Up Required", desc: "An anonymous workspace is created on first use — your data is scoped to it, and only you hold the key" },
    { icon: <Target className="w-5 h-5" />, title: "Evidence-First Decisions", desc: "A claim with no retrievable evidence is returned for human review, never guessed" },
    { icon: <Sparkles className="w-5 h-5" />, title: "Auditable Overrides", desc: "Human approvals, rejections, and corrections are recorded against the claim they changed" },
  ];

  return (
    <section className="section-padding relative" id="features">
      <div className="max-w-6xl mx-auto">
        <motion.div
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, amount: 0.3 }}
          variants={stagger}
          className="text-center mb-16"
        >
          <motion.p variants={fadeUp} custom={0} className="text-sm font-semibold text-accent uppercase tracking-wider mb-3">
            Features
          </motion.p>
          <motion.h2 variants={fadeUp} custom={1} className="text-4xl md:text-5xl font-bold mb-4">
            Built for <span className="gradient-text">Enterprise AI</span>
          </motion.h2>
        </motion.div>

        <motion.div
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, amount: 0.1 }}
          variants={stagger}
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5"
        >
          {features.map((f, i) => (
            <motion.div key={f.title} variants={fadeUp} custom={i}>
              <GlassCard className="h-full group">
                <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center text-primary mb-4 group-hover:bg-primary/20 transition-colors">
                  {f.icon}
                </div>
                <h3 className="text-base font-semibold mb-2">{f.title}</h3>
                <p className="text-sm text-muted-foreground leading-relaxed">{f.desc}</p>
              </GlassCard>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </section>
  );
}

export function Footer() {
  return (
    <footer className="border-t border-border py-12 px-6">
      <div className="max-w-6xl mx-auto flex flex-col md:flex-row items-center justify-between gap-6">
        <div className="flex items-center gap-2">
          <Shield className="w-5 h-5 text-primary" />
          <span className="font-bold">
            <span className="gradient-text">Truth</span>Lens
            <span className="text-muted-foreground text-sm font-normal ml-1">AI</span>
          </span>
        </div>
        <div className="flex items-center gap-5 text-sm text-muted-foreground flex-wrap justify-center">
          <Link to="/verify" className="hover:text-foreground transition-colors">Verify</Link>
          <Link to="/review" className="hover:text-foreground transition-colors">Review</Link>
          <Link to="/dashboard" className="hover:text-foreground transition-colors">Dashboard</Link>
          <Link to="/benchmark" className="hover:text-foreground transition-colors">Benchmark</Link>
          <Link to="/admin" className="hover:text-foreground transition-colors">Admin</Link>
          <span className="w-px h-3 bg-border" aria-hidden="true" />
          <Link to="/image-refinement" className="hover:text-foreground transition-colors">Image RLVO</Link>
          <Link to="/video-refinement" className="hover:text-foreground transition-colors">Video RLVO</Link>
          <Link to="/proctoring" className="hover:text-foreground transition-colors">Proctoring</Link>
        </div>
        <p className="text-xs text-muted-foreground">
          © 2025 TruthLens AI. The trust layer missing from modern AI.
        </p>
      </div>
    </footer>
  );
}
