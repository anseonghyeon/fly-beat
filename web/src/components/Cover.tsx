/**
 * 곡 제목으로 만드는 자동 앨범 아트 (실제 이미지 없이, 같은 제목이면 항상 같은 그림).
 * 네온 그라데이션 + 겹눈 육각 격자 + 제목 머리글자.
 */
function hash(s: string): number {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 16777619);
  return h >>> 0;
}

function initials(title: string): string {
  const words = title.split(/\s+/).filter(Boolean);
  const a = words[0]?.[0] ?? "?";
  const b = words[1]?.[0] ?? words[0]?.[1] ?? "";
  return (a + b).toUpperCase();
}

export function Cover({ title, size = 48 }: { title: string; size?: number }) {
  const h = hash(title);
  const hue1 = 180 + (h % 120); // 시안 ~ 보라 ~ 마젠타 계열
  const hue2 = (hue1 + 60 + ((h >> 8) % 60)) % 360;
  const id = `cv${h.toString(36)}`;
  const hexes: [number, number][] = [];
  for (let row = 0; row < 6; row++)
    for (let col = 0; col < 6; col++) {
      if (((h >> ((row * 6 + col) % 30)) & 3) !== 0) continue;
      hexes.push([col * 18 + (row % 2) * 9, row * 16]);
    }
  return (
    <svg
      className="cover"
      width={size}
      height={size}
      viewBox="0 0 100 100"
      role="img"
      aria-label={`${title} 커버`}
    >
      <defs>
        <linearGradient id={id} x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor={`hsl(${hue1} 90% 45%)`} />
          <stop offset="1" stopColor={`hsl(${hue2} 85% 30%)`} />
        </linearGradient>
      </defs>
      <rect width="100" height="100" fill={`url(#${id})`} />
      {hexes.map(([x, y], i) => (
        <polygon
          key={i}
          points={[0, 1, 2, 3, 4, 5]
            .map((k) => {
              const a = (Math.PI / 3) * k + Math.PI / 6;
              return `${x + 8 * Math.cos(a)},${y + 8 * Math.sin(a)}`;
            })
            .join(" ")}
          fill="rgba(255,255,255,0.12)"
        />
      ))}
      <text
        x="50"
        y="62"
        textAnchor="middle"
        fontSize="36"
        fontWeight="900"
        fontStyle="italic"
        fill="rgba(255,255,255,0.92)"
      >
        {initials(title)}
      </text>
    </svg>
  );
}
