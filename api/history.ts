/**
 * Verification history and no-model replay.
 *
 * Replay returns the stored snapshot verbatim. No provider is contacted, no quota is spent, and
 * nothing is recomputed — which is the point: a result you have already paid for should be
 * readable forever without paying again.
 *
 * Both storage drivers are served. Under Supabase, sessions are scoped to a verified user id;
 * under the local driver, to a browser-held device id. The scope key differs, the response shape
 * does not, so the page renders identically either way.
 */
import { clientSafeError, sendJson } from "./_gemini.js";
import { statusOf } from "./_identity.js";
import { listHistory, readReplay, resolveSession, storageDriver, storageDurability, storageUnavailableReason } from "./_store.js";
import { deleteSession } from "./_localstore.js";

export default async function handler(req: any, res: any) {
  if (req.method !== "POST") return sendJson(res, 405, { error: "Method not allowed" });

  try {
    const session = await resolveSession(req);
    if (!session) {
      // 200, not 401: "nothing is stored here" is a supported state, and the page needs to explain
      // it rather than render an error. `signInRequired` tells the UI which of the two it is.
      return sendJson(res, 200, {
        sessions: [],
        storage: { driver: storageDriver(), available: false, signInRequired: storageDriver() === "supabase" },
        reason: storageUnavailableReason(),
      });
    }

    const storage = {
      driver: session.driver,
      available: true,
      signInRequired: false,
      attributable: session.attributable,
      durability: session.driver === "local" ? storageDurability() : "durable",
    };

    const id = typeof req.body?.id === "string" ? req.body.id : null;

    if (id && req.body?.action === "delete") {
      if (session.driver !== "local") return sendJson(res, 501, { error: "Deleting stored sessions is not supported on this storage driver yet." });
      const removed = await deleteSession(session.deviceId as string, id);
      return sendJson(res, removed ? 200 : 404, removed ? { deleted: id } : { error: "Session not found." });
    }

    if (id) {
      const snapshot = await readReplay(session, id);
      if (!snapshot) return sendJson(res, 404, { error: "That verification is not in your history." });
      return sendJson(res, 200, {
        result: {
          ...snapshot,
          replayMode: true,
          replay: { replayedAt: new Date().toISOString(), aiCallsUsed: 0, detail: "Replayed from storage. No AI provider was contacted." },
        },
        storage,
      });
    }

    return sendJson(res, 200, { sessions: await listHistory(session), storage });
  } catch (error) {
    return sendJson(res, statusOf(error, 500), { error: clientSafeError(error, "history").message });
  }
}
