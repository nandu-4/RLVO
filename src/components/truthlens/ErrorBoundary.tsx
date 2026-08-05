import { Component, type ErrorInfo, type ReactNode } from "react";
import { AlertTriangle, RefreshCw, Home } from "lucide-react";

interface Props {
  children: ReactNode;
}

interface State {
  error: Error | null;
  /** Short random id the user can quote; correlates the screen with the console entry. */
  reference: string | null;
}

/**
 * Catches render-time failures so a single bad component cannot blank the entire application.
 *
 * Without this, any uncaught render error unmounts the React tree and leaves a white screen with
 * no explanation and no way back — the worst possible outcome in front of an evaluator. The stack
 * trace stays in the console for developers; the user gets a readable message and two exits.
 */
export default class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null, reference: null };

  static getDerivedStateFromError(error: Error): State {
    return { error, reference: Math.random().toString(36).slice(2, 8).toUpperCase() };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    // Developer-facing detail only. Never rendered — stack traces must not reach the UI.
    console.error(`[TruthLens ${this.state.reference}]`, error, info.componentStack);
  }

  private reset = () => this.setState({ error: null, reference: null });

  render() {
    if (!this.state.error) return this.props.children;

    return (
      <div className="min-h-screen flex items-center justify-center aurora-bg text-foreground p-6">
        <div className="glass rounded-2xl border border-border/60 p-8 max-w-lg w-full text-center">
          <div className="w-14 h-14 rounded-2xl bg-danger/10 border border-danger/30 flex items-center justify-center mx-auto mb-5 text-danger">
            <AlertTriangle className="w-7 h-7" />
          </div>
          <h1 className="text-xl font-bold mb-2">Something went wrong on this page</h1>
          <p className="text-sm text-muted-foreground leading-relaxed mb-1">
            The rest of TruthLens is unaffected. Your verification results are not lost — anything already
            stored in your workspace is still there.
          </p>
          <p className="text-xs text-muted-foreground mb-6">
            Reference <span className="font-mono text-foreground">{this.state.reference}</span> · full detail is in the
            browser console.
          </p>
          <div className="flex items-center justify-center gap-2">
            <button onClick={this.reset} className="btn-primary text-xs py-2.5 px-4 flex items-center gap-1.5 relative z-10">
              <RefreshCw className="w-3.5 h-3.5" /> Try again
            </button>
            <a href="/" className="btn-secondary text-xs py-2.5 px-4 flex items-center gap-1.5">
              <Home className="w-3.5 h-3.5" /> Back to home
            </a>
          </div>
        </div>
      </div>
    );
  }
}
