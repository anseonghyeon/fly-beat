export function FlyIcon({ size = 22 }: { size?: number }) {
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
      <ellipse cx="12" cy="14" rx="3" ry="5" />
      <circle cx="12" cy="7.5" r="2" />
      <path d="M9.5 12C6 9 3 9.5 3 12s3 3 6.5 1.5M14.5 12C18 9 21 9.5 21 12s-3 3-6.5 1.5" />
    </svg>
  );
}

/** 네온 로고: 초파리 아이콘 + FLYMAX RESPECT V */
export function Logo({ big = false }: { big?: boolean }) {
  return (
    <div className={`logo ${big ? "big" : ""}`}>
      <span className="logo-icon">
        <FlyIcon size={big ? 96 : 40} />
      </span>
      <span className="logo-word">
        <span className="fly">FLYMAX</span>
        <span className="beat">RESPECT V</span>
      </span>
    </div>
  );
}
