import { useEffect, useState } from "react";
import { Logo } from "./Logo";

const KEY = "flymax.splashShown";
const SHOW_MS = 1400;
const FADE_MS = 700;

function alreadyShown(): boolean {
  try {
    return sessionStorage.getItem(KEY) === "1";
  } catch {
    return false;
  }
}

/** 앱을 열 때 화면 가운데 로고가 떴다가 서서히 사라지는 시작 효과 (탭마다 한 번). */
export function Splash() {
  const [phase, setPhase] = useState<"show" | "fade" | "done">(() =>
    alreadyShown() ? "done" : "show",
  );

  useEffect(() => {
    if (phase === "done") return;
    try {
      sessionStorage.setItem(KEY, "1");
    } catch {
      /* 무시 */
    }
    const reduced = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    const t1 = window.setTimeout(() => setPhase("fade"), reduced ? 500 : SHOW_MS);
    const t2 = window.setTimeout(() => setPhase("done"), (reduced ? 500 : SHOW_MS) + FADE_MS);
    const skip = () => setPhase("fade");
    window.addEventListener("keydown", skip, { once: true });
    window.addEventListener("pointerdown", skip, { once: true });
    return () => {
      window.clearTimeout(t1);
      window.clearTimeout(t2);
      window.removeEventListener("keydown", skip);
      window.removeEventListener("pointerdown", skip);
    };
    // phase 가 바뀔 때마다 다시 걸지 않도록 처음 한 번만
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (phase !== "fade") return;
    const t = window.setTimeout(() => setPhase("done"), FADE_MS);
    return () => window.clearTimeout(t);
  }, [phase]);

  if (phase === "done") return null;
  return (
    <div className={`splash ${phase === "fade" ? "out" : ""}`} aria-hidden>
      <div className="splash-inner">
        <Logo big />
        <div className="splash-sub">HUMAN VS FLY</div>
      </div>
    </div>
  );
}
