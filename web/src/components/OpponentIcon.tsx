import { FlyIcon } from "./Logo";

export function RobotIcon({ size = 22 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      aria-hidden
    >
      <rect x="5" y="8" width="14" height="11" rx="3" />
      <circle cx="9.5" cy="13" r="1.3" />
      <circle cx="14.5" cy="13" r="1.3" />
      <path d="M12 8V5M10 19v2M14 19v2M3 12v3M21 12v3" />
      <circle cx="12" cy="4" r="1" />
    </svg>
  );
}

export function OpponentIcon({
  kind,
  size = 22,
}: {
  kind: "robot" | "fly" | "solo";
  size?: number;
}) {
  if (kind === "robot") return <RobotIcon size={size} />;
  if (kind === "fly") return <FlyIcon size={size} />;
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      aria-hidden
    >
      <circle cx="12" cy="8" r="3.5" />
      <path d="M5 20c1-4 4-6 7-6s6 2 7 6" />
    </svg>
  );
}
