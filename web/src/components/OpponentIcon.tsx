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

export function OpponentIcon({ kind, size = 22 }: { kind: "robot" | "fly"; size?: number }) {
  return kind === "robot" ? <RobotIcon size={size} /> : <FlyIcon size={size} />;
}
