import { useEffect, useRef, useState } from "react";

export interface Series {
  name: string;
  /** CSS 색 변수 이름 (예: "--mode-a") */
  color: string;
  points: { x: number; y: number }[];
}

interface Props {
  series: Series[];
  xLabel: string;
  yLabel: string;
  yDomain?: [number, number];
  height?: number;
  formatX?: (x: number) => string;
  formatY?: (y: number) => string;
}

const M = { top: 12, right: 16, bottom: 34, left: 46 };

function ticks(min: number, max: number, n = 5): number[] {
  if (max <= min) return [min];
  const raw = (max - min) / n;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? raw;
  const out: number[] = [];
  for (let v = Math.ceil(min / step) * step; v <= max + 1e-9; v += step) out.push(v);
  return out;
}

const fmt = (v: number) =>
  Math.abs(v) >= 1e6
    ? `${(v / 1e6).toFixed(1)}M`
    : Math.abs(v) >= 1e3
      ? `${(v / 1e3).toFixed(0)}k`
      : `${Math.round(v * 10) / 10}`;

/** 단일 y축 선 그래프 + 크로스헤어 툴팁. 시리즈가 2개 이상이면 범례를 보인다. */
export function LineChart({
  series,
  xLabel,
  yLabel,
  yDomain,
  height = 240,
  formatX = fmt,
  formatY = fmt,
}: Props) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(560);
  const [hover, setHover] = useState<number | null>(null);

  const all = series.flatMap((s) => s.points);
  const xMin = Math.min(...all.map((p) => p.x), 0);
  const xMax = Math.max(...all.map((p) => p.x), 1);
  const [yMin, yMax] = yDomain ?? [
    Math.min(...all.map((p) => p.y), 0),
    Math.max(...all.map((p) => p.y), 1),
  ];

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => e && setWidth(Math.max(280, e.contentRect.width)));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const iw = width - M.left - M.right;
  const ih = height - M.top - M.bottom;
  const sx = (x: number) => M.left + ((x - xMin) / (xMax - xMin || 1)) * iw;
  const sy = (y: number) => M.top + ih - ((y - yMin) / (yMax - yMin || 1)) * ih;

  const onMove = (e: React.PointerEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = xMin + ((e.clientX - rect.left - M.left) / iw) * (xMax - xMin);
    setHover(Math.min(xMax, Math.max(xMin, x)));
  };

  const nearest = (pts: { x: number; y: number }[], x: number) =>
    pts.reduce<{ x: number; y: number } | null>(
      (best, p) => (!best || Math.abs(p.x - x) < Math.abs(best.x - x) ? p : best),
      null,
    );

  return (
    <div ref={wrapRef} className="chart">
      {series.length >= 2 && (
        <div className="legend">
          {series.map((s) => (
            <span key={s.name}>
              <i style={{ background: `var(${s.color})` }} />
              {s.name}
            </span>
          ))}
        </div>
      )}
      <svg
        width={width}
        height={height}
        role="img"
        aria-label={`${yLabel} / ${xLabel}`}
        onPointerMove={onMove}
        onPointerLeave={() => setHover(null)}
      >
        {ticks(yMin, yMax).map((v) => (
          <g key={v}>
            <line x1={M.left} x2={M.left + iw} y1={sy(v)} y2={sy(v)} className="grid" />
            <text x={M.left - 8} y={sy(v)} className="tick" textAnchor="end" dy="0.32em">
              {formatY(v)}
            </text>
          </g>
        ))}
        {ticks(xMin, xMax, 6).map((v) => (
          <text key={v} x={sx(v)} y={M.top + ih + 16} className="tick" textAnchor="middle">
            {formatX(v)}
          </text>
        ))}
        <text x={M.left + iw / 2} y={height - 4} className="axis-label" textAnchor="middle">
          {xLabel}
        </text>
        <text
          transform={`translate(12 ${M.top + ih / 2}) rotate(-90)`}
          className="axis-label"
          textAnchor="middle"
        >
          {yLabel}
        </text>
        {series.map((s) => (
          <polyline
            key={s.name}
            fill="none"
            stroke={`var(${s.color})`}
            strokeWidth={2}
            strokeLinejoin="round"
            points={s.points.map((p) => `${sx(p.x)},${sy(p.y)}`).join(" ")}
          />
        ))}
        {hover !== null && (
          <g>
            <line x1={sx(hover)} x2={sx(hover)} y1={M.top} y2={M.top + ih} className="crosshair" />
            {series.map((s) => {
              const p = nearest(s.points, hover);
              return p ? (
                <circle
                  key={s.name}
                  cx={sx(p.x)}
                  cy={sy(p.y)}
                  r={4.5}
                  fill={`var(${s.color})`}
                  stroke="var(--surface)"
                  strokeWidth={2}
                />
              ) : null;
            })}
          </g>
        )}
      </svg>
      {hover !== null && (
        <div
          className="tooltip"
          style={{ left: Math.min(sx(hover) + 12, width - 170), top: M.top + 4 }}
        >
          {series.map((s) => {
            const p = nearest(s.points, hover);
            return p ? (
              <div key={s.name}>
                <i style={{ background: `var(${s.color})` }} />
                {series.length > 1 ? `${s.name} · ` : ""}
                {xLabel} {formatX(p.x)}: <b>{formatY(p.y)}</b>
              </div>
            ) : null;
          })}
        </div>
      )}
    </div>
  );
}
