#!/usr/bin/env bash

set -Eeuo pipefail

ROLLBACK_SCRIPT_DIRECTORY="$(
  cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1
  pwd
)"
readonly ROLLBACK_SCRIPT_DIRECTORY

# Reuse the deployment script's validated Compose, retry, capacity,
# environment-file, and health-check functions.
# shellcheck source=scripts/deploy_ec2.sh
source "${ROLLBACK_SCRIPT_DIRECTORY}/deploy_ec2.sh"

readonly DEPLOYMENT_BRANCH="${AGENTFLOW_DEPLOY_BRANCH:-main}"

ROLLBACK_ORIGINAL_COMMIT=""
ROLLBACK_TARGET_COMMIT=""
ROLLBACK_CONFIRMED="false"

print_usage() {
  cat <<'USAGE'
Usage:
  scripts/rollback_ec2.sh --target <commit> --confirm

Options:
  --target <commit>  Older Git commit to restore.
  --confirm          Explicitly authorize the rollback.
  --help             Show this message.

The rollback preserves Docker volumes and does not downgrade the database.
USAGE
}

handle_rollback_error() {
  local line_number="$1"
  local exit_code="$2"

  trap - ERR
  set +e

  log_event \
    "error" \
    "agentflow_ec2_rollback_failed" \
    "line=${line_number}" \
    "exit_code=${exit_code}" \
    "original_commit=${ROLLBACK_ORIGINAL_COMMIT:-unknown}" \
    "target_commit=${ROLLBACK_TARGET_COMMIT:-unknown}"

  if command -v docker >/dev/null 2>&1 \
    && [[ -f "$ENV_FILE" ]] \
    && [[ -f "$BASE_COMPOSE_FILE" ]] \
    && [[ -f "$EC2_COMPOSE_FILE" ]]; then
    compose_command ps 2>/dev/null || true
    compose_command logs \
      --tail 200 \
      backend \
      frontend \
      migrate \
      2>/dev/null || true
  fi

  exit "$exit_code"
}

trap 'handle_rollback_error "$LINENO" "$?"' ERR

parse_arguments() {
  while (($# > 0)); do
    case "$1" in
      --target)
        if (($# < 2)) || [[ -z "$2" ]]; then
          log_event \
            "error" \
            "rollback_target_value_missing"
          print_usage >&2
          exit 2
        fi

        ROLLBACK_TARGET_COMMIT="$2"
        shift 2
        ;;
      --confirm)
        ROLLBACK_CONFIRMED="true"
        shift
        ;;
      --help|-h)
        print_usage
        exit 0
        ;;
      *)
        log_event \
          "error" \
          "rollback_argument_rejected" \
          "argument=$1"
        print_usage >&2
        exit 2
        ;;
    esac
  done

  if [[ -z "$ROLLBACK_TARGET_COMMIT" ]]; then
    log_event \
      "error" \
      "rollback_target_required"
    print_usage >&2
    exit 2
  fi

  if [[ "$ROLLBACK_CONFIRMED" != "true" ]]; then
    log_event \
      "error" \
      "rollback_confirmation_required"
    print_usage >&2
    exit 2
  fi
}

validate_deployment_branch() {
  local current_branch

  current_branch="$(
    git -C "$REPOSITORY_ROOT" branch --show-current
  )"

  if [[ "$current_branch" != "$DEPLOYMENT_BRANCH" ]]; then
    log_event \
      "error" \
      "rollback_branch_rejected" \
      "current_branch=${current_branch:-detached}" \
      "required_branch=${DEPLOYMENT_BRANCH}"
    exit 1
  fi

  log_event \
    "info" \
    "rollback_branch_validated" \
    "branch=${current_branch}"
}

validate_rollback_target() {
  local requested_commit="$ROLLBACK_TARGET_COMMIT"
  local resolved_commit
  local current_commit
  local required_path

  if ! resolved_commit="$(
    git -C "$REPOSITORY_ROOT" \
      rev-parse \
      --verify \
      "${requested_commit}^{commit}" \
      2>/dev/null
  )"; then
    log_event \
      "error" \
      "rollback_target_commit_invalid" \
      "requested_commit=${requested_commit}"
    exit 1
  fi

  current_commit="$(
    git -C "$REPOSITORY_ROOT" rev-parse HEAD
  )"

  if [[ "$resolved_commit" == "$current_commit" ]]; then
    log_event \
      "error" \
      "rollback_target_is_current_commit" \
      "commit=${resolved_commit}"
    exit 1
  fi

  if ! git -C "$REPOSITORY_ROOT" \
    merge-base \
    --is-ancestor \
    "$resolved_commit" \
    "$current_commit"; then
    log_event \
      "error" \
      "rollback_non_ancestor_target_rejected" \
      "target_commit=${resolved_commit}" \
      "current_commit=${current_commit}"
    exit 1
  fi

  for required_path in \
    docker-compose.yml \
    docker-compose.ec2.yml \
    backend/Dockerfile \
    frontend/Dockerfile; do
    if ! git -C "$REPOSITORY_ROOT" \
      cat-file \
      -e \
      "${resolved_commit}:${required_path}" \
      2>/dev/null; then
      log_event \
        "error" \
        "rollback_target_file_missing" \
        "target_commit=${resolved_commit}" \
        "path=${required_path}"
      exit 1
    fi
  done

  ROLLBACK_ORIGINAL_COMMIT="$current_commit"
  ROLLBACK_TARGET_COMMIT="$resolved_commit"

  log_event \
    "info" \
    "rollback_target_validated" \
    "original_commit=${ROLLBACK_ORIGINAL_COMMIT}" \
    "target_commit=${ROLLBACK_TARGET_COMMIT}"
}

capture_pre_rollback_state() {
  log_event \
    "info" \
    "rollback_preparation_started" \
    "original_commit=${ROLLBACK_ORIGINAL_COMMIT}" \
    "target_commit=${ROLLBACK_TARGET_COMMIT}" \
    "database_downgrade=not_performed" \
    "docker_volumes=preserved"

  compose_command ps || true
}

stop_application_preserving_data() {
  compose_command down --remove-orphans

  log_event \
    "info" \
    "rollback_application_stopped" \
    "docker_volumes=preserved"
}

restore_target_commit() {
  git -C "$REPOSITORY_ROOT" \
    reset \
    --hard \
    "$ROLLBACK_TARGET_COMMIT"

  local restored_commit
  restored_commit="$(
    git -C "$REPOSITORY_ROOT" rev-parse HEAD
  )"

  if [[ "$restored_commit" != "$ROLLBACK_TARGET_COMMIT" ]]; then
    log_event \
      "error" \
      "rollback_commit_restore_mismatch" \
      "expected_commit=${ROLLBACK_TARGET_COMMIT}" \
      "actual_commit=${restored_commit}"
    exit 1
  fi

  log_event \
    "info" \
    "rollback_commit_restored" \
    "commit=${restored_commit}"
}

verify_backend_health() {
  compose_command exec -T backend \
    python -c \
    'import urllib.request; response = urllib.request.urlopen(
        "http://127.0.0.1:8000/api/v1/health",
        timeout=10,
    ); raise SystemExit(0 if response.status == 200 else 1)'

  log_event \
    "info" \
    "rollback_backend_health_verified"
}

perform_rollback() {
  capture_pre_rollback_state
  stop_application_preserving_data
  restore_target_commit

  validate_compose_configuration
  pull_runtime_images
  build_application_images
  start_application
  verify_backend_health
  verify_frontend_health

  log_event \
    "info" \
    "agentflow_ec2_rollback_completed" \
    "original_commit=${ROLLBACK_ORIGINAL_COMMIT}" \
    "restored_commit=${ROLLBACK_TARGET_COMMIT}" \
    "database_downgrade=not_performed" \
    "docker_volumes=preserved"
}

main() {
  parse_arguments "$@"
  validate_required_commands
  validate_repository
  validate_environment_file
  validate_host_capacity
  validate_deployment_branch
  validate_rollback_target
  validate_compose_configuration
  perform_rollback
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
