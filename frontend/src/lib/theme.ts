import { useEffect, useState } from "react";

export type Theme = "light" | "dark";
const KEY = "kltn11-theme";

function initial(): Theme {
  try {
    const v = localStorage.getItem(KEY);
    if (v === "dark" || v === "light") return v;
  } catch {
    /* storage unavailable */
  }
  return "light"; // Default to Titanium Pearl palette
}

let current: Theme = initial();
const subs = new Set<() => void>();
const apply = (t: Theme) => {
  document.documentElement.dataset.theme = t;
  document.documentElement.classList.toggle("dark", t === "dark");
};
apply(current);

export function setTheme(t: Theme) {
  current = t;
  apply(t);
  try {
    localStorage.setItem(KEY, t);
  } catch {
    /* ignore */
  }
  subs.forEach((f) => f());
}

export function useTheme() {
  const [t, setT] = useState<Theme>(current);
  useEffect(() => {
    const f = () => setT(current);
    subs.add(f);
    f();
    return () => {
      subs.delete(f);
    };
  }, []);
  return [t, () => setTheme(t === "dark" ? "light" : "dark")] as const;
}
