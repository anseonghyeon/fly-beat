import type { ChartSummary } from "../game/types";

interface Props {
  charts: ChartSummary[];
  value: string;
  onChange: (id: string) => void;
  disabled?: boolean;
}

export function ChartPicker({ charts, value, onChange, disabled }: Props) {
  return (
    <label className="field">
      채보
      <select value={value} disabled={disabled} onChange={(e) => onChange(e.target.value)}>
        {charts.map((c) => (
          <option key={c.id} value={c.id}>
            {c.audio ? "♪ " : ""}
            {c.title} · {Math.round(c.bpm)} BPM · {c.notes} notes
          </option>
        ))}
      </select>
    </label>
  );
}
