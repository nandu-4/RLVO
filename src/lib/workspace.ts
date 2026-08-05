/**
 * Anonymous workspace token.
 *
 * TruthLens has no sign-up. The browser mints a high-entropy token on first use and sends it with
 * every request; the server stores only its SHA-256 hash and scopes all data to it.
 *
 * The token is a bearer secret, like an unlisted share link: whoever holds it has the workspace,
 * and it cannot be recovered. The Admin page says so, and offers export/import so the user can
 * move a workspace between browsers deliberately rather than losing it by accident.
 */

const STORAGE_KEY = "truthlens.workspace";
const TOKEN_RE = /^[A-Za-z0-9_-]{24,128}$/;

function mint(): string {
  const bytes = new Uint8Array(32);
  crypto.getRandomValues(bytes);
  // URL-safe base64 without padding — 256 bits of entropy in 43 characters.
  return btoa(String.fromCharCode(...bytes)).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function read(): string | null {
  try {
    const stored = window.localStorage.getItem(STORAGE_KEY);
    return stored && TOKEN_RE.test(stored) ? stored : null;
  } catch {
    return null; // Private mode or blocked storage: the app still verifies, just without persistence.
  }
}

/** Returns the workspace token, creating one on first call. Null when storage is unavailable. */
export function workspaceToken(): string | null {
  const existing = read();
  if (existing) return existing;
  const token = mint();
  try {
    window.localStorage.setItem(STORAGE_KEY, token);
    return token;
  } catch {
    return null;
  }
}

export function replaceWorkspaceToken(token: string): boolean {
  const trimmed = token.trim();
  if (!TOKEN_RE.test(trimmed)) return false;
  try {
    window.localStorage.setItem(STORAGE_KEY, trimmed);
    return true;
  } catch {
    return false;
  }
}

/** Abandons the current workspace locally. The stored data is untouched and becomes unreachable. */
export function forgetWorkspace(): void {
  try {
    window.localStorage.removeItem(STORAGE_KEY);
  } catch {
    /* nothing to forget */
  }
}

export const workspaceStorageAvailable = (): boolean => {
  try {
    const probe = "__truthlens_probe__";
    window.localStorage.setItem(probe, "1");
    window.localStorage.removeItem(probe);
    return true;
  } catch {
    return false;
  }
};
