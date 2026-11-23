import { Navigate, NavLink, Route, Routes } from "react-router-dom";
import { FlyIcon } from "./components/Logo";
import { Splash } from "./components/Splash";
import Brain from "./pages/Brain";
import Lobby from "./pages/Lobby";
import Play from "./pages/Play";
import Train from "./pages/Train";

const NAV = [
  { to: "/", label: "Lobby" },
  { to: "/train", label: "Train" },
  { to: "/brain", label: "Brain" },
];

export default function App() {
  return (
    <div className="app">
      <Splash />
      <header className="topbar">
        <NavLink to="/" className="brand">
          <FlyIcon /> fly-beat
        </NavLink>
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
          <Route path="/" element={<Lobby />} />
          <Route path="/play" element={<Play />} />
          <Route path="/watch" element={<Navigate to="/" replace />} />
          <Route path="/train" element={<Train />} />
          <Route path="/brain" element={<Brain />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
    </div>
  );
}
