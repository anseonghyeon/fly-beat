import react from "@vitejs/plugin-react";
import type { Plugin } from "vite";
import { defineConfig } from "vitest/config";

// 빌드 결과물에만 CSP 를 넣는다 (개발 서버는 HMR 용 인라인 스크립트가 필요).
const csp: Plugin = {
  name: "flymax-csp",
  apply: "build",
  transformIndexHtml(html) {
    const policy =
      "default-src 'self'; connect-src 'self' ws: wss:; style-src 'self' 'unsafe-inline'; img-src 'self' data:; object-src 'none'; base-uri 'none'";
    return html.replace(
      "<head>",
      `<head>\n    <meta http-equiv="Content-Security-Policy" content="${policy}" />`,
    );
  },
};

const API = "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react(), csp],
  server: {
    host: "127.0.0.1",
    port: 5173,
    strictPort: true,
    proxy: {
      "/api": API,
      "/ws": { target: API.replace("http", "ws"), ws: true },
    },
  },
  test: {
    include: ["src/**/*.test.ts"],
  },
});
