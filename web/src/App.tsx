import { Navigate, NavLink, Route, Routes } from "react-router-dom";
import Brain from "./pages/Brain";
import Play from "./pages/Play";
import Train from "./pages/Train";

function FlyIcon() {
  return (
    <svg
      width="22"
      height="22"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      aria-hidden
    >
      <ellipse cx="12" cy="14" rx="3" ry="5" />
      <circle cx="12" cy="7.5" r="2" />
      <path d="M9.5 12C6 9 3 9.5 3 12s3 3 6.5 1.5M14.5 12C18 9 21 9.5 21 12s-3 3-6.5 1.5" />
    </svg>
  );
}

const NAV = [
  { to: "/", label: "Play" },
  { to: "/train", label: "Train" },
  { to: "/brain", label: "Brain" },
];

export default function App() {
  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <FlyIcon /> fly-beat
        </div>
        <nav className="nav">
          {NAV.map((n) => (
            <NavLink key={n.to} to={n.to} end>
              {n.label}
            </NavLink>
          ))}
        </nav>
      </header>
      <main>
        <Routes>
          <Route path="/" element={<Play />} />
          <Route path="/watch" element={<Navigate to="/" replace />} />
          <Route path="/train" element={<Train />} />
          <Route path="/brain" element={<Brain />} />
        </Routes>
      </main>
    </div>
  );
}
