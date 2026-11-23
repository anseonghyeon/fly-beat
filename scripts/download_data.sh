#!/usr/bin/env bash
# MaleCNS v1.0 커넥톰 데이터 다운로드 + SHA-256 검증
# 출처: https://male-cns.janelia.org/download/ (CC-BY 4.0)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${ROOT_DIR}/data"
CHECKSUMS="${ROOT_DIR}/scripts/checksums.sha256"
BASE_URL="https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome"

FILES=(
  "body-annotations-male-cns-v1.0-minconf-0.5.feather"
  "body-neurotransmitters-male-cns-v1.0.feather"
  "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
)

if command -v sha256sum >/dev/null 2>&1; then
  SHA_CMD=(sha256sum)
elif command -v shasum >/dev/null 2>&1; then
  SHA_CMD=(shasum -a 256)
else
  echo "sha256sum 또는 shasum 이 필요합니다." >&2
  exit 1
fi

expected_hash() {
  awk -v f="$1" '$2 == f { print $1 }' "${CHECKSUMS}"
}

verify() {
  local file="$1" expected actual
  expected="$(expected_hash "$(basename "${file}")")"
  if [[ -z "${expected}" ]]; then
    echo "체크섬 목록에 없는 파일: ${file}" >&2
    return 1
  fi
  actual="$("${SHA_CMD[@]}" "${file}" | awk '{ print $1 }')"
  [[ "${actual}" == "${expected}" ]]
}

mkdir -p "${DATA_DIR}"

for name in "${FILES[@]}"; do
  target="${DATA_DIR}/${name}"
  if [[ -f "${target}" ]] && verify "${target}"; then
    echo "[ok] ${name} (이미 있음)"
    continue
  fi

  echo "[download] ${name}"
  tmp="$(mktemp "${DATA_DIR}/.${name}.XXXXXX")"
  trap 'rm -f "${tmp}"' EXIT
  curl --fail --location --proto '=https' --tlsv1.2 --retry 3 \
    --output "${tmp}" "${BASE_URL}/${name}"

  if ! verify "${tmp}"; then
    echo "[error] 체크섬 불일치: ${name}. 파일을 삭제합니다." >&2
    rm -f "${tmp}"
    exit 1
  fi
  mv "${tmp}" "${target}"
  trap - EXIT
  echo "[ok] ${name}"
done

echo "완료: ${DATA_DIR}"
