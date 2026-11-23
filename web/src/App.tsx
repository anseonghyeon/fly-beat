import { Navigate, Route, Routes, useLocation } from "react-router-dom";
import { EscMenu } from "./components/EscMenu";
import { Splash } from "./components/Splash";
import Brain from "./pages/Brain";
import Lobby from "./pages/Lobby";
import Play from "./pages/Play";
import Result from "./pages/Result";
import Train from "./pages/Train";

/** 학습·뇌 보기 화면에는 Esc 메뉴 안내를 작게 띄운다 (로비는 아래 키 안내에 있음) */
function EscHint() {
  const { pathname } = useLocation();
  if (pathname !== "/train" && pathname !== "/brain") return null;
  return (
    <div className="esc-hint">
      <kbd>Esc</kbd> 메뉴
    </div>
  );
}

export default function App() {
  return (
    <div className="app">
      <Splash />
      <EscMenu />
      <EscHint />
      <main>
        <Routes>
          <Route path="/" element={<Lobby />} />
          <Route path="/play" element={<Play />} />
          <Route path="/result" element={<Result />} />
          <Route path="/watch" element={<Navigate to="/" replace />} />
          <Route path="/train" element={<Train />} />
          <Route path="/brain" element={<Brain />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
    </div>
  );
}
