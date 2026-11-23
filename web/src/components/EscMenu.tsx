/**
 * Esc 메뉴: 어느 화면에서든 Esc 로 열고 닫는다. 열려 있는 동안 키 입력은 메뉴만 받는다.
 */
import { useEffect, useRef, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { Logo } from "./Logo";

const ITEMS = [
  { to: "/", label: "로비", desc: "곡을 골라 대결하기" },
  { to: "/train", label: "학습", desc: "상대를 새로 학습시키기" },
  { to: "/brain", label: "뇌 보기", desc: "초파리 커넥톰과 학습으로 바뀐 시냅스" },
] as const;

export function EscMenu() {
  const [open, setOpen] = useState(false);
  const [index, setIndex] = useState(0);
  const navigate = useNavigate();
  const location = useLocation();
  const listRef = useRef<HTMLDivElement>(null);

  // 다른 화면으로 이동하면 메뉴를 닫는다
  const [lastPath, setLastPath] = useState(location.pathname);
  if (lastPath !== location.pathname) {
    setLastPath(location.pathname);
    setOpen(false);
  }

  useEffect(() => {
    const total = ITEMS.length + 1; // + 돌아가기
    const choose = (i: number) => {
      setOpen(false);
      const item = ITEMS[i];
      if (item && item.to !== location.pathname) navigate(item.to);
    };
    // 캡처 단계에서 먼저 받아, 메뉴가 열려 있으면 페이지의 키 처리(게임 키·로비 조작)를 막는다
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        e.stopImmediatePropagation();
        setOpen((o) => !o);
        setIndex(0);
        return;
      }
      if (!open) return;
      e.stopImmediatePropagation();
      if (e.key === "ArrowDown" || e.key === "Tab") {
        e.preventDefault();
        setIndex((i) => (i + 1) % total);
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        setIndex((i) => (i - 1 + total) % total);
      } else if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        if (index < ITEMS.length) choose(index);
        else setOpen(false);
      }
    };
    window.addEventListener("keydown", onKey, { capture: true });
    return () => window.removeEventListener("keydown", onKey, { capture: true });
  }, [open, index, navigate, location.pathname]);

  useEffect(() => {
    if (open) listRef.current?.querySelectorAll("button")[index]?.focus();
  }, [open, index]);

  if (!open) return null;
  return (
    <div className="esc-backdrop" onClick={() => setOpen(false)}>
      <div
        className="esc-menu"
        role="dialog"
        aria-modal="true"
        aria-label="메뉴"
        onClick={(e) => e.stopPropagation()}
      >
        <Logo />
        <div className="esc-items" ref={listRef}>
          {ITEMS.map((it, i) => (
            <button
              key={it.to}
              className={`esc-item ${i === index ? "on" : ""} ${
                location.pathname === it.to ? "here" : ""
              }`}
              onMouseEnter={() => setIndex(i)}
              onClick={() => {
                setOpen(false);
                if (it.to !== location.pathname) navigate(it.to);
              }}
            >
              <b>{it.label}</b>
              <small>{location.pathname === it.to ? "지금 화면" : it.desc}</small>
            </button>
          ))}
          <button
            className={`esc-item back ${index === ITEMS.length ? "on" : ""}`}
            onMouseEnter={() => setIndex(ITEMS.length)}
            onClick={() => setOpen(false)}
          >
            <b>돌아가기</b>
            <small>Esc</small>
          </button>
        </div>
      </div>
    </div>
  );
}
