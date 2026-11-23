/** 결과 페이지: 플레이가 끝나면 이동해 온다. */
import { useCallback } from "react";
import { Navigate, useLocation, useNavigate } from "react-router-dom";
import { ResultView, type MatchResult } from "../components/ResultView";
import { unlockAudio } from "../game/audio";
import { loadLastResult, type PlaySettings } from "../game/settings";

export interface ResultState {
  result: MatchResult;
  settings: PlaySettings;
}

export default function Result() {
  const location = useLocation();
  const navigate = useNavigate();
  const state = (location.state as ResultState | null) ?? loadLastResult<ResultState>();

  const retry = useCallback(() => {
    if (!state) return;
    unlockAudio(); // 클릭 안에서 오디오를 열어 둬야 플레이 화면에서 바로 소리가 난다
    navigate("/play", { state: state.settings });
  }, [navigate, state]);
  const lobby = useCallback(() => navigate("/"), [navigate]);

  if (!state) return <Navigate to="/" replace />;
  return (
    <div className="result-page">
      <ResultView result={state.result} onRetry={retry} onLobby={lobby} />
    </div>
  );
}
