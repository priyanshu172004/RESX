"use client";

/**
 * The session provider.
 *
 * On mount it attempts one silent refresh. That is what makes a reload keep
 * the user signed in: the access token only ever lived in memory and is gone,
 * but the HttpOnly refresh cookie survived, so the session can be rebuilt
 * without the user typing anything.
 *
 * The four-state `status` matters. Collapsing "checking" into "signed out"
 * would flash the login screen on every reload before the refresh completes,
 * and redirect the user away from the page they asked for. And "unreachable"
 * is kept apart from "anonymous" because they call for opposite advice: one
 * means sign in, the other means the API is down and signing in cannot work
 * either.
 *
 * Every path out of the bootstrap sets a status. That is the invariant this
 * file exists to keep — it was broken, and the app hung on "Restoring your
 * session…" on every reload in development. See the effect below.
 */

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import {
  ApiError,
  auth as authApi,
  refreshSessionDetailed,
  setAccessToken,
  type User,
} from "@/lib/api";

type Status = "checking" | "authenticated" | "anonymous" | "unreachable";

/**
 * A failure to reach the API, as distinct from a rejection by it.
 *
 * The distinction changes what the user should be told. "Sign in again" is
 * useless advice when the API is down — the login form posts to the same
 * origin and will fail identically — so an unreachable API says so and offers
 * a retry instead of bouncing the user to a form that cannot work.
 */
function isUnreachable(error: unknown): boolean {
  if (error instanceof ApiError) {
    // The server answered, so it is reachable. 5xx included: that is a broken
    // server rather than an absent one, but retrying is still the right move.
    return error.status >= 500;
  }
  // `fetch` rejects with a TypeError for a network failure and an AbortError
  // (DOMException) for the timeout added in `lib/api`.
  return true;
}

interface AuthState {
  status: Status;
  user: User | null;
  /** Retry the session bootstrap after an unreachable API. */
  retry: () => Promise<void>;
  signIn: (email: string, password: string) => Promise<void>;
  signUp: (input: {
    email: string;
    password: string;
    name: string;
    workspaceName?: string;
  }) => Promise<void>;
  signOut: () => Promise<void>;
  reload: () => Promise<void>;
}

const AuthContext = createContext<AuthState | null>(null);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [status, setStatus] = useState<Status>("checking");
  const [user, setUser] = useState<User | null>(null);
  // Guards against the double-invoked effect in React's development strict
  // mode. Two refreshes in flight would rotate the token twice, and the
  // second would present an already-used token — which the server correctly
  // treats as replay and answers by revoking the whole family.
  const bootstrapped = useRef(false);

  const loadUser = useCallback(async () => {
    const { user: me } = await authApi.me();
    setUser(me);
    setStatus("authenticated");
  }, []);

  useEffect(() => {
    if (bootstrapped.current) return;
    bootstrapped.current = true;

    // No `cancelled` flag, and that is the fix rather than an omission.
    //
    // There was one, and together with the ref guard above it wedged the app
    // on "Restoring your session…" every single reload in development:
    //
    //   1. The effect runs. `bootstrapped` flips to true and the refresh
    //      starts.
    //   2. React's strict mode immediately runs the cleanup, setting
    //      `cancelled = true`.
    //   3. The effect runs a second time and returns at once, because
    //      `bootstrapped` is already true.
    //   4. The first refresh resolves, sees `cancelled`, and returns without
    //      touching the status.
    //
    // Nobody ever left "checking", so the spinner ran forever with nothing in
    // the console. Two mechanisms were guarding the same thing and cancelled
    // each other: the ref already guarantees exactly one bootstrap for the
    // life of the provider, which is what actually matters here — a second
    // refresh would rotate the token twice and the server would correctly read
    // the reused one as replay and revoke the family.
    //
    // Setting state after an unmount is a no-op in React 18+, so dropping the
    // flag costs nothing and removes the deadlock.
    (async () => {
      try {
        const { token, reachable } = await refreshSessionDetailed();
        if (!token) {
          // Reachable and declined means there is genuinely no session.
          // Unreachable means we do not know, and saying "sign in" would be a
          // guess presented as a fact.
          setStatus(reachable ? "anonymous" : "unreachable");
          return;
        }
        await loadUser();
      } catch (error) {
        // `refreshSession` swallows its own failures, so reaching here means
        // /auth/me failed. A network error is not the same as being signed
        // out: telling someone with a valid session to log in again, when the
        // API is simply unreachable, sends them to a login form that cannot
        // work either.
        setAccessToken(null);
        setStatus(isUnreachable(error) ? "unreachable" : "anonymous");
      }
    })();
  }, [loadUser]);

  const signIn = useCallback(
    async (email: string, password: string) => {
      const session = await authApi.login({ email, password });
      setAccessToken(session.access_token);
      setUser(session.user);
      setStatus("authenticated");
    },
    [],
  );

  const signUp = useCallback(
    async (input: {
      email: string;
      password: string;
      name: string;
      workspaceName?: string;
    }) => {
      const session = await authApi.register({
        email: input.email,
        password: input.password,
        name: input.name,
        workspace_name: input.workspaceName,
      });
      setAccessToken(session.access_token);
      setUser(session.user);
      setStatus("authenticated");
    },
    [],
  );

  const signOut = useCallback(async () => {
    try {
      await authApi.logout();
    } finally {
      // Local state is cleared regardless. If the network call failed the
      // server-side session may survive, but leaving the UI signed in would
      // be worse: the user asked to be signed out.
      setAccessToken(null);
      setUser(null);
      setStatus("anonymous");
    }
  }, []);

  const retry = useCallback(async () => {
    setStatus("checking");
    try {
      const { token, reachable } = await refreshSessionDetailed();
      if (!token) {
        setStatus(reachable ? "anonymous" : "unreachable");
        return;
      }
      await loadUser();
    } catch (error) {
      setAccessToken(null);
      setStatus(isUnreachable(error) ? "unreachable" : "anonymous");
    }
  }, [loadUser]);

  const value = useMemo<AuthState>(
    () => ({ status, user, signIn, signUp, signOut, retry, reload: loadUser }),
    [status, user, signIn, signUp, signOut, retry, loadUser],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used inside <AuthProvider>");
  }
  return context;
}
