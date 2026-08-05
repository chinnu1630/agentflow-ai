#!/usr/bin/env bash

set -Eeuo pipefail

readonly EXPECTED_OS_ID="ubuntu"
readonly EXPECTED_OS_CODENAME="noble"
readonly -a SUPPORTED_ARCHITECTURES=("amd64" "arm64")
readonly DOCKER_KEY_URL="https://download.docker.com/linux/ubuntu/gpg"
readonly DOCKER_KEY_PATH="/etc/apt/keyrings/docker.asc"
readonly DOCKER_REPOSITORY_PATH="/etc/apt/sources.list.d/docker.list"
readonly DOWNLOAD_MAX_ATTEMPTS=5
readonly DOWNLOAD_INITIAL_BACKOFF_SECONDS=2

export DEBIAN_FRONTEND="noninteractive"

log_event() {
  local level="$1"
  local event="$2"
  shift 2

  printf 'timestamp=%s level=%s event=%s' \
    "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
    "$level" \
    "$event"

  for field in "$@"; do
    printf ' %s' "$field"
  done

  printf '\n'
}

download_file_with_retry() {
  local url="$1"
  local destination="$2"
  local attempt=1
  local backoff_seconds="$DOWNLOAD_INITIAL_BACKOFF_SECONDS"
  local temporary_destination="${destination}.partial"

  rm -f "$temporary_destination"

  while ((attempt <= DOWNLOAD_MAX_ATTEMPTS)); do
    if curl \
      --fail \
      --silent \
      --show-error \
      --location \
      --connect-timeout 10 \
      --max-time 60 \
      "$url" \
      --output "$temporary_destination"; then
      mv "$temporary_destination" "$destination"

      log_event \
        "info" \
        "external_download_completed" \
        "attempt=${attempt}"

      return 0
    fi

    rm -f "$temporary_destination"

    if ((attempt == DOWNLOAD_MAX_ATTEMPTS)); then
      log_event \
        "error" \
        "external_download_failed" \
        "attempts=${attempt}"

      return 1
    fi

    log_event \
      "warning" \
      "external_download_retry_scheduled" \
      "attempt=${attempt}" \
      "backoff_seconds=${backoff_seconds}"

    sleep "$backoff_seconds"
    backoff_seconds=$((backoff_seconds * 2))
    attempt=$((attempt + 1))
  done
}

handle_error() {
  local line_number="$1"
  local exit_code="$2"

  log_event \
    "error" \
    "agentflow_ec2_bootstrap_failed" \
    "line=${line_number}" \
    "exit_code=${exit_code}"

  exit "$exit_code"
}

trap 'handle_error "$LINENO" "$?"' ERR

require_root() {
  if [[ "$EUID" -ne 0 ]]; then
    log_event \
      "error" \
      "root_privileges_required" \
      "command=sudo_scripts_bootstrap_ec2"
    exit 1
  fi
}

is_supported_architecture() {
  local candidate_architecture="$1"
  local supported_architecture

  for supported_architecture in "${SUPPORTED_ARCHITECTURES[@]}"; do
    if [[ "$candidate_architecture" == "$supported_architecture" ]]; then
      return 0
    fi
  done

  return 1
}

validate_host() {
  # /etc/os-release exists on the target Ubuntu EC2 host.
  # shellcheck disable=SC1091
  source /etc/os-release

  local architecture
  architecture="$(dpkg --print-architecture)"

  if [[ "${ID:-}" != "$EXPECTED_OS_ID" ]]; then
    log_event \
      "error" \
      "unsupported_operating_system" \
      "expected=${EXPECTED_OS_ID}" \
      "actual=${ID:-unknown}"
    exit 1
  fi

  if [[ "${VERSION_CODENAME:-}" != "$EXPECTED_OS_CODENAME" ]]; then
    log_event \
      "error" \
      "unsupported_ubuntu_release" \
      "expected=${EXPECTED_OS_CODENAME}" \
      "actual=${VERSION_CODENAME:-unknown}"
    exit 1
  fi

  if ! is_supported_architecture "$architecture"; then
    log_event \
      "error" \
      "unsupported_cpu_architecture" \
      "supported=$(IFS=,; printf '%s' "${SUPPORTED_ARCHITECTURES[*]}")" \
      "actual=${architecture}"
    exit 1
  fi

  log_event \
    "info" \
    "ec2_host_validated" \
    "os=${ID}" \
    "codename=${VERSION_CODENAME}" \
    "architecture=${architecture}"
}

remove_conflicting_packages() {
  local package
  local -a conflicting_packages=(
    docker.io
    docker-compose
    docker-compose-v2
    docker-doc
    podman-docker
    containerd
    runc
  )
  local -a installed_conflicts=()

  for package in "${conflicting_packages[@]}"; do
    if dpkg-query -W -f='${Status}' "$package" 2>/dev/null \
      | grep -q 'ok installed'; then
      installed_conflicts+=("$package")
    fi
  done

  if [[ "${#installed_conflicts[@]}" -gt 0 ]]; then
    log_event \
      "info" \
      "removing_conflicting_container_packages" \
      "count=${#installed_conflicts[@]}"

    apt-get remove --yes "${installed_conflicts[@]}"
  fi
}

install_base_packages() {
  apt-get update

  apt-get install \
    --yes \
    --no-install-recommends \
    ca-certificates \
    curl \
    git
}

configure_docker_repository() {
  # /etc/os-release exists on the target Ubuntu EC2 host.
  # shellcheck disable=SC1091
  source /etc/os-release

  local architecture
  architecture="$(dpkg --print-architecture)"

  install -m 0755 -d /etc/apt/keyrings

  download_file_with_retry \
    "$DOCKER_KEY_URL" \
    /tmp/docker.asc

  install -m 0644 /tmp/docker.asc "$DOCKER_KEY_PATH"
  rm -f /tmp/docker.asc

  printf \
    'deb [arch=%s signed-by=%s] https://download.docker.com/linux/%s %s stable\n' \
    "$architecture" \
    "$DOCKER_KEY_PATH" \
    "$ID" \
    "$VERSION_CODENAME" \
    > "$DOCKER_REPOSITORY_PATH"

  chmod 0644 "$DOCKER_REPOSITORY_PATH"
}

install_docker() {
  apt-get update

  apt-get install \
    --yes \
    --no-install-recommends \
    docker-ce \
    docker-ce-cli \
    containerd.io \
    docker-buildx-plugin \
    docker-compose-plugin

  systemctl enable --now docker
}

configure_deployment_user() {
  local deployment_user="${SUDO_USER:-ubuntu}"

  if ! id "$deployment_user" >/dev/null 2>&1; then
    log_event \
      "error" \
      "deployment_user_not_found" \
      "user=${deployment_user}"
    exit 1
  fi

  if [[ "$deployment_user" != "root" ]]; then
    usermod --append --groups docker "$deployment_user"

    log_event \
      "info" \
      "deployment_user_added_to_docker_group" \
      "user=${deployment_user}" \
      "session_restart_required=true"
  fi
}

verify_installation() {
  systemctl is-active --quiet docker
  docker info >/dev/null
  docker compose version >/dev/null

  log_event \
    "info" \
    "agentflow_ec2_bootstrap_completed" \
    "docker_version=$(docker version --format '{{.Server.Version}}')" \
    "compose_version=$(docker compose version --short)"
}

main() {
  require_root
  validate_host
  remove_conflicting_packages
  install_base_packages
  configure_docker_repository
  install_docker
  configure_deployment_user
  verify_installation
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
