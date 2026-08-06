#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIRECTORY="$(
  cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1
  pwd
)"
readonly SCRIPT_DIRECTORY

REPOSITORY_ROOT="$(
  cd "${SCRIPT_DIRECTORY}/.." >/dev/null 2>&1
  pwd
)"
readonly REPOSITORY_ROOT

readonly ENV_FILE="${AGENTFLOW_ENV_FILE:-/etc/agentflow/agentflow.env}"
readonly COMPOSE_PROJECT_NAME="${AGENTFLOW_COMPOSE_PROJECT_NAME:-agentflow-ai}"
readonly BASE_COMPOSE_FILE="${REPOSITORY_ROOT}/docker-compose.yml"
readonly EC2_COMPOSE_FILE="${REPOSITORY_ROOT}/docker-compose.ec2.yml"
readonly RETRY_MAX_ATTEMPTS=4
readonly RETRY_INITIAL_BACKOFF_SECONDS=2
readonly MINIMUM_MEMORY_KIB=$((3 * 1024 * 1024))
readonly MINIMUM_DISK_KIB=$((12 * 1024 * 1024))

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

compose_command() (
  local -a compose_environment_variables=(
    "AGENTFLOW_FRONTEND_DEMO_ACCESS_TOKEN"
    "AGENTFLOW_FRONTEND_REQUEST_TIMEOUT_SECONDS"
    "AUTH_ENABLED"
    "AUTH_JWT_AUDIENCE"
    "AUTH_JWT_ISSUER"
    "AUTH_JWT_PUBLIC_KEY"
    "BACKEND_PORT"
    "FRONTEND_PORT"
    "GITHUB_DEFAULT_BRANCH"
    "GITHUB_REPOSITORY_NAME"
    "GITHUB_REPOSITORY_OWNER"
    "GITHUB_TOKEN"
    "JIRA_API_TOKEN"
    "JIRA_BASE_URL"
    "JIRA_EMAIL"
    "JIRA_PROJECT_KEY"
    "OTEL_ENABLED"
    "OTEL_EXPORTER_OTLP_ENDPOINT"
    "OTEL_METRICS_ENABLED"
    "OTEL_METRICS_EXPORTER_OTLP_ENDPOINT"
    "OTEL_METRICS_EXPORT_INTERVAL_MILLISECONDS"
    "OTEL_SAMPLE_RATIO"
    "OTEL_SERVICE_NAME"
    "POSTGRES_DB"
    "POSTGRES_PASSWORD"
    "POSTGRES_PORT"
    "POSTGRES_USER"
    "RATE_LIMIT_EXPENSIVE_CAPACITY"
    "RATE_LIMIT_EXPENSIVE_REFILL_RATE_PER_SECOND"
    "RATE_LIMIT_KEY_HMAC_SECRET"
    "RATE_LIMIT_STANDARD_CAPACITY"
    "RATE_LIMIT_STANDARD_REFILL_RATE_PER_SECOND"
    "REDIS_CONNECT_TIMEOUT_SECONDS"
    "REDIS_MAX_CONNECTIONS"
    "REDIS_PASSWORD"
    "REDIS_PORT"
    "REDIS_SOCKET_TIMEOUT_SECONDS"
    "TRUSTED_HOSTS"
  )

  unset "${compose_environment_variables[@]}"

  docker compose \
    --project-name "$COMPOSE_PROJECT_NAME" \
    --env-file "$ENV_FILE" \
    -f "$BASE_COMPOSE_FILE" \
    -f "$EC2_COMPOSE_FILE" \
    "$@"
)

handle_error() {
  local line_number="$1"
  local exit_code="$2"

  trap - ERR
  set +e

  log_event \
    "error" \
    "agentflow_ec2_deployment_failed" \
    "line=${line_number}" \
    "exit_code=${exit_code}"

  if command -v docker >/dev/null 2>&1 \
    && [[ -f "$ENV_FILE" ]] \
    && [[ -f "$BASE_COMPOSE_FILE" ]] \
    && [[ -f "$EC2_COMPOSE_FILE" ]]; then
    compose_command ps 2>/dev/null || true
  fi

  exit "$exit_code"
}

trap 'handle_error "$LINENO" "$?"' ERR

require_command() {
  local command_name="$1"

  if ! command -v "$command_name" >/dev/null 2>&1; then
    log_event \
      "error" \
      "required_command_missing" \
      "command=${command_name}"
    exit 1
  fi
}

retry_command() {
  local operation="$1"
  shift

  local attempt=1
  local backoff_seconds="$RETRY_INITIAL_BACKOFF_SECONDS"

  while ((attempt <= RETRY_MAX_ATTEMPTS)); do
    if "$@"; then
      log_event \
        "info" \
        "deployment_operation_completed" \
        "operation=${operation}" \
        "attempt=${attempt}"
      return 0
    fi

    if ((attempt == RETRY_MAX_ATTEMPTS)); then
      log_event \
        "error" \
        "deployment_operation_failed" \
        "operation=${operation}" \
        "attempts=${attempt}"
      return 1
    fi

    log_event \
      "warning" \
      "deployment_operation_retry_scheduled" \
      "operation=${operation}" \
      "attempt=${attempt}" \
      "backoff_seconds=${backoff_seconds}"

    sleep "$backoff_seconds"
    backoff_seconds=$((backoff_seconds * 2))
    attempt=$((attempt + 1))
  done
}

validate_required_commands() {
  require_command curl
  require_command docker
  require_command git
  require_command stat

  docker compose version >/dev/null
  docker info >/dev/null

  log_event \
    "info" \
    "deployment_dependencies_validated"
}

validate_repository() {
  if [[ ! -f "$BASE_COMPOSE_FILE" || ! -f "$EC2_COMPOSE_FILE" ]]; then
    log_event \
      "error" \
      "compose_configuration_missing"
    exit 1
  fi

  if [[ ! -d "${REPOSITORY_ROOT}/.git" ]]; then
    log_event \
      "error" \
      "git_repository_missing" \
      "repository=${REPOSITORY_ROOT}"
    exit 1
  fi

  if [[ -n "$(git -C "$REPOSITORY_ROOT" status --porcelain)" ]]; then
    log_event \
      "error" \
      "deployment_repository_not_clean"
    exit 1
  fi

  log_event \
    "info" \
    "deployment_repository_validated" \
    "commit=$(git -C "$REPOSITORY_ROOT" rev-parse --short HEAD)"
}

get_file_mode() {
  local file_path="$1"

  if stat -c '%a' "$file_path" >/dev/null 2>&1; then
    stat -c '%a' "$file_path"
    return
  fi

  stat -f '%Lp' "$file_path"
}

get_file_owner_uid() {
  local file_path="$1"

  if stat -c '%u' "$file_path" >/dev/null 2>&1; then
    stat -c '%u' "$file_path"
    return
  fi

  stat -f '%u' "$file_path"
}

validate_environment_file() {
  if [[ ! -f "$ENV_FILE" ]]; then
    log_event \
      "error" \
      "deployment_environment_file_missing" \
      "path=${ENV_FILE}"
    exit 1
  fi

  if [[ -L "$ENV_FILE" ]]; then
    log_event \
      "error" \
      "deployment_environment_symlink_rejected" \
      "path=${ENV_FILE}"
    exit 1
  fi

  local file_mode
  local permission_value
  local owner_uid
  local current_uid

  file_mode="$(get_file_mode "$ENV_FILE")"
  permission_value=$((8#$file_mode))
  owner_uid="$(get_file_owner_uid "$ENV_FILE")"
  current_uid="$(id -u)"

  if ((permission_value & 0007)); then
    log_event \
      "error" \
      "deployment_environment_world_access_rejected" \
      "mode=${file_mode}"
    exit 1
  fi

  if ((permission_value & 0020)); then
    log_event \
      "error" \
      "deployment_environment_group_write_rejected" \
      "mode=${file_mode}"
    exit 1
  fi

  if [[ "$owner_uid" != "0" && "$owner_uid" != "$current_uid" ]]; then
    log_event \
      "error" \
      "deployment_environment_owner_rejected" \
      "owner_uid=${owner_uid}"
    exit 1
  fi

  log_event \
    "info" \
    "deployment_environment_file_validated" \
    "mode=${file_mode}"
}


count_environment_value_occurrences() {
  local variable_name="$1"

  awk -v target="$variable_name" '
    {
      line = $0
      sub(/\r$/, "", line)

      if (line ~ /^[[:space:]]*#/ || line ~ /^[[:space:]]*$/) {
        next
      }

      separator = index(line, "=")

      if (separator == 0) {
        next
      }

      name = substr(line, 1, separator - 1)
      sub(/^[[:space:]]*export[[:space:]]+/, "", name)
      gsub(/^[[:space:]]+/, "", name)
      gsub(/[[:space:]]+$/, "", name)

      if (name == target) {
        occurrences += 1
      }
    }

    END {
      print occurrences + 0
    }
  ' "$ENV_FILE"
}

read_environment_value() {
  local variable_name="$1"

  awk -v target="$variable_name" '
    {
      line = $0
      sub(/\r$/, "", line)

      if (line ~ /^[[:space:]]*#/ || line ~ /^[[:space:]]*$/) {
        next
      }

      separator = index(line, "=")

      if (separator == 0) {
        next
      }

      name = substr(line, 1, separator - 1)
      sub(/^[[:space:]]*export[[:space:]]+/, "", name)
      gsub(/^[[:space:]]+/, "", name)
      gsub(/[[:space:]]+$/, "", name)

      if (name == target) {
        value = substr(line, separator + 1)
        gsub(/^[[:space:]]+/, "", value)
        gsub(/[[:space:]]+$/, "", value)
        print value
      }
    }
  ' "$ENV_FILE"
}

strip_matching_quotes() {
  local value="$1"
  local value_length="${#value}"

  if ((value_length >= 2)); then
    local first_character="${value:0:1}"
    local last_character="${value:value_length-1:1}"

    if [[ "$first_character" == "$last_character" ]] \
      && [[ "$first_character" == '"' || "$first_character" == "'" ]]; then
      value="${value:1:value_length-2}"
    fi
  fi

  printf '%s' "$value"
}

load_required_environment_value() {
  local variable_name="$1"
  local occurrence_count
  local raw_value

  occurrence_count="$(
    count_environment_value_occurrences "$variable_name"
  )"

  if [[ "$occurrence_count" == "0" ]]; then
    log_event \
      "error" \
      "deployment_environment_required_value_missing" \
      "variable=${variable_name}"
    return 1
  fi

  if [[ "$occurrence_count" != "1" ]]; then
    log_event \
      "error" \
      "deployment_environment_duplicate_value_rejected" \
      "variable=${variable_name}"
    return 1
  fi

  raw_value="$(read_environment_value "$variable_name")"
  REQUIRED_ENVIRONMENT_VALUE="$(strip_matching_quotes "$raw_value")"

  if [[ -z "$REQUIRED_ENVIRONMENT_VALUE" ]]; then
    log_event \
      "error" \
      "deployment_environment_required_value_blank" \
      "variable=${variable_name}"
    return 1
  fi
}

is_unsafe_secret_placeholder() {
  local secret_value="$1"
  local normalized_value

  normalized_value="$(
    printf '%s' "$secret_value" |
      tr '[:upper:]' '[:lower:]'
  )"

  case "$normalized_value" in
    changeme|change-me|change_me|password|secret|default|example|test|\
test-secret|your-password|replace-me|replace_me|agentflow-ci-password)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

validate_deployment_secret() {
  local variable_name="$1"
  local secret_value="$2"

  if is_unsafe_secret_placeholder "$secret_value"; then
    log_event \
      "error" \
      "deployment_environment_secret_rejected" \
      "variable=${variable_name}" \
      "reason=placeholder"
    return 1
  fi

  if ((${#secret_value} < 16)); then
    log_event \
      "error" \
      "deployment_environment_secret_rejected" \
      "variable=${variable_name}" \
      "reason=minimum_length"
    return 1
  fi
}

validate_trusted_hosts_value() {
  local trusted_hosts="$1"
  local normalized_hosts
  local host
  local external_host_found="false"
  local -a host_entries=()

  if [[ ! "$trusted_hosts" =~ ^\[[[:space:]]*\"[^\"]+\"([[:space:]]*,[[:space:]]*\"[^\"]+\")*[[:space:]]*\]$ ]]; then
    log_event \
      "error" \
      "deployment_environment_trusted_hosts_rejected" \
      "reason=json_array_required"
    return 1
  fi

  normalized_hosts="$(
    printf '%s' "$trusted_hosts" |
      tr '[:upper:]' '[:lower:]' |
      tr -d '[]"' |
      tr -d '[:space:]'
  )"

  if [[ "$normalized_hosts" == *"*"* ]]; then
    log_event \
      "error" \
      "deployment_environment_trusted_hosts_rejected" \
      "reason=wildcard"
    return 1
  fi

  IFS=',' read -r -a host_entries <<< "$normalized_hosts"

  for host in "${host_entries[@]}"; do
    case "$host" in
      ""|localhost|127.0.0.1|::1|testserver|backend)
        ;;
      *)
        external_host_found="true"
        ;;
    esac
  done

  if [[ "$external_host_found" != "true" ]]; then
    log_event \
      "error" \
      "deployment_environment_trusted_hosts_rejected" \
      "reason=local_only"
    return 1
  fi
}

validate_environment_values() {
  local auth_enabled
  local auth_jwt_audience
  local auth_jwt_issuer
  local auth_jwt_public_key
  local postgres_password
  local rate_limit_hmac_secret
  local redis_password
  local trusted_hosts

  load_required_environment_value "AUTH_ENABLED" || return 1
  auth_enabled="$REQUIRED_ENVIRONMENT_VALUE"

  load_required_environment_value "AUTH_JWT_AUDIENCE" || return 1
  auth_jwt_audience="$REQUIRED_ENVIRONMENT_VALUE"

  load_required_environment_value "AUTH_JWT_ISSUER" || return 1
  auth_jwt_issuer="$REQUIRED_ENVIRONMENT_VALUE"

  load_required_environment_value "AUTH_JWT_PUBLIC_KEY" || return 1
  auth_jwt_public_key="$REQUIRED_ENVIRONMENT_VALUE"

  load_required_environment_value "POSTGRES_PASSWORD" || return 1
  postgres_password="$REQUIRED_ENVIRONMENT_VALUE"

  load_required_environment_value "RATE_LIMIT_KEY_HMAC_SECRET" || return 1
  rate_limit_hmac_secret="$REQUIRED_ENVIRONMENT_VALUE"

  load_required_environment_value "REDIS_PASSWORD" || return 1
  redis_password="$REQUIRED_ENVIRONMENT_VALUE"

  load_required_environment_value "TRUSTED_HOSTS" || return 1
  trusted_hosts="$REQUIRED_ENVIRONMENT_VALUE"

  if [[ "$(
    printf '%s' "$auth_enabled" |
      tr '[:upper:]' '[:lower:]'
  )" != "true" ]]; then
    log_event \
      "error" \
      "deployment_environment_authentication_rejected" \
      "variable=AUTH_ENABLED" \
      "reason=must_be_true"
    return 1
  fi

  if [[ "$auth_jwt_issuer" != http://* \
    && "$auth_jwt_issuer" != https://* ]]; then
    log_event \
      "error" \
      "deployment_environment_value_rejected" \
      "variable=AUTH_JWT_ISSUER" \
      "reason=http_scheme_required"
    return 1
  fi

  if [[ "$auth_jwt_public_key" != *"-----BEGIN PUBLIC KEY-----"* \
    || "$auth_jwt_public_key" != *"-----END PUBLIC KEY-----"* ]]; then
    log_event \
      "error" \
      "deployment_environment_value_rejected" \
      "variable=AUTH_JWT_PUBLIC_KEY" \
      "reason=pem_public_key_required"
    return 1
  fi

  if [[ -z "$auth_jwt_audience" ]]; then
    log_event \
      "error" \
      "deployment_environment_required_value_blank" \
      "variable=AUTH_JWT_AUDIENCE"
    return 1
  fi

  validate_deployment_secret \
    "POSTGRES_PASSWORD" \
    "$postgres_password" || return 1

  validate_deployment_secret \
    "RATE_LIMIT_KEY_HMAC_SECRET" \
    "$rate_limit_hmac_secret" || return 1

  validate_deployment_secret \
    "REDIS_PASSWORD" \
    "$redis_password" || return 1

  validate_trusted_hosts_value "$trusted_hosts" || return 1

  log_event \
    "info" \
    "deployment_environment_values_validated"
}


validate_host_capacity() {
  local memory_kib
  local available_disk_kib

  memory_kib="$(
    awk '/^MemTotal:/ {print $2}' /proc/meminfo
  )"

  available_disk_kib="$(
    df -Pk "$REPOSITORY_ROOT" |
      awk 'NR == 2 {print $4}'
  )"

  if ((memory_kib < MINIMUM_MEMORY_KIB)); then
    log_event \
      "error" \
      "insufficient_host_memory" \
      "available_kib=${memory_kib}" \
      "required_kib=${MINIMUM_MEMORY_KIB}"
    exit 1
  fi

  if ((available_disk_kib < MINIMUM_DISK_KIB)); then
    log_event \
      "error" \
      "insufficient_host_disk" \
      "available_kib=${available_disk_kib}" \
      "required_kib=${MINIMUM_DISK_KIB}"
    exit 1
  fi

  log_event \
    "info" \
    "deployment_host_capacity_validated" \
    "memory_kib=${memory_kib}" \
    "available_disk_kib=${available_disk_kib}"
}

validate_compose_configuration() {
  compose_command config --quiet

  log_event \
    "info" \
    "deployment_compose_configuration_validated"
}

pull_runtime_images() {
  retry_command \
    "pull_runtime_images" \
    compose_command pull postgres redis
}

build_application_images() {
  retry_command \
    "build_application_images" \
    compose_command build --pull backend frontend migrate
}

start_application() {
  retry_command \
    "start_application" \
    compose_command up \
      --detach \
      --remove-orphans \
      --wait \
      --wait-timeout 300
}

verify_frontend_health() {
  local frontend_binding
  local health_response

  frontend_binding="$(
    compose_command port frontend 8501 |
      tail -n 1
  )"

  if [[ -z "$frontend_binding" ]]; then
    log_event \
      "error" \
      "frontend_binding_not_found"
    exit 1
  fi

  health_response="$(
    curl \
      --fail \
      --silent \
      --show-error \
      --connect-timeout 5 \
      --max-time 10 \
      "http://${frontend_binding}/_stcore/health"
  )"

  if [[ "$health_response" != "ok" ]]; then
    log_event \
      "error" \
      "frontend_health_check_failed"
    exit 1
  fi

  log_event \
    "info" \
    "agentflow_ec2_deployment_completed" \
    "commit=$(git -C "$REPOSITORY_ROOT" rev-parse --short HEAD)" \
    "frontend_binding=${frontend_binding}"
}

main() {
  validate_required_commands
  validate_repository
  validate_environment_file
  validate_environment_values
  validate_host_capacity
  validate_compose_configuration
  pull_runtime_images
  build_application_images
  start_application
  verify_frontend_health
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
