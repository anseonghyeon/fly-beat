import type { Stats } from "../game/types";

export function StatsView({ stats }: { stats: Stats }) {
  const items: [string, string | number][] = [
    ["정확도", `${stats.accuracy.toFixed(2)}%`],
    ["최대 콤보", stats.maxCombo],
    ["PERFECT", stats.counts.PERFECT],
    ["GREAT", stats.counts.GREAT],
    ["GOOD", stats.counts.GOOD],
    ["MISS", stats.counts.MISS],
    ["헛누름", stats.empty],
  ];
  return (
    <div className="stat-grid">
      {items.map(([k, v]) => (
        <div className="stat" key={k}>
          <div className="k">{k}</div>
          <div className="v">{v}</div>
        </div>
      ))}
    </div>
  );
}
