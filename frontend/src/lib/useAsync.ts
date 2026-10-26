import { useEffect, useState } from "react";

export interface AsyncState<T> {
  data: T | null;
  error: string | null;
  loading: boolean;
}

/** Runs `fn` whenever `deps` change; aborts the previous request. */
export function useAsync<T>(
  fn: (signal: AbortSignal) => Promise<T>,
  deps: unknown[],
): AsyncState<T> {
  const [state, setState] = useState<AsyncState<T>>({
    data: null,
    error: null,
    loading: true,
  });

  useEffect(() => {
    const ctrl = new AbortController();
    setState((s) => ({ data: s.data, error: null, loading: true }));
    fn(ctrl.signal)
      .then((data) => setState({ data, error: null, loading: false }))
      .catch((e: Error) => {
        if (e.name === "AbortError") return;
        setState({ data: null, error: e.message, loading: false });
      });
    return () => ctrl.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  return state;
}
