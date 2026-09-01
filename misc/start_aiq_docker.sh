#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
compose_dir="${script_dir}/../deploy/compose"

if (( $# == 0 )); then
    services=(aiq-agent frontend)
else
    services=("$@")
fi

for service in "${services[@]}"; do
    case "${service}" in
        aiq-agent | frontend) ;;
        *)
            echo "Usage: $0 [aiq-agent] [frontend]" >&2
            exit 2
            ;;
    esac
done

(
    cd "${compose_dir}"
    BUILD_TARGET=release docker compose --env-file ../.env -f docker-compose.yaml up -d --build "${services[@]}"
)

if [[ " ${services[*]} " == *" aiq-agent "* ]]; then
    docker network inspect nvidia-rag >/dev/null
    if ! docker network inspect nvidia-rag --format '{{json .Containers}}' | grep -q '"aiq-agent"'; then
        docker network connect nvidia-rag aiq-agent
    fi
fi
