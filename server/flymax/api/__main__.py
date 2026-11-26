"""서버 실행: python -m flymax.api [--host 127.0.0.1] [--port 8000]"""

from __future__ import annotations

import argparse
import sys

import uvicorn

from flymax.api.settings import settings

LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m flymax.api")
    p.add_argument("--host", default=settings.host)
    p.add_argument("--port", type=int, default=settings.port)
    args = p.parse_args()
    if args.host not in LOCAL_HOSTS:
        print(
            f"[경고] {args.host} 에 바인딩합니다. 이 서버는 인증이 없으므로 "
            "같은 네트워크의 누구나 학습을 시작하고 자원을 쓸 수 있습니다.",
            file=sys.stderr,
        )
    uvicorn.run("flymax.api.main:app", host=args.host, port=args.port, ws_max_size=1 << 20)


if __name__ == "__main__":
    main()
