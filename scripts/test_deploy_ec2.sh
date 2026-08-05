#!/usr/bin/env bash

set -Eeuo pipefail

TEST_TEMP_DIRECTORY="$(mktemp -d)"
readonly TEST_TEMP_DIRECTORY

export AGENTFLOW_ENV_FILE="${TEST_TEMP_DIRECTORY}/agentflow.env"

TEST_SCRIPT_DIRECTORY="$(
  cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1
  pwd
)"
readonly TEST_SCRIPT_DIRECTORY

# Loading the deployment script defines its functions without running main().
# shellcheck source=scripts/deploy_ec2.sh
source "${TEST_SCRIPT_DIRECTORY}/deploy_ec2.sh"

# Tests handle expected failures explicitly.
trap - ERR

cleanup() {
  rm -rf "$TEST_TEMP_DIRECTORY"
}

trap cleanup EXIT

fail_test() {
  local message="$1"

  printf 'TEST FAILED: %s\n' "$message" >&2
  exit 1
}

assert_equal() {
  local expected="$1"
  local actual="$2"
  local description="$3"

  if [[ "$actual" != "$expected" ]]; then
    fail_test \
      "${description}: expected=${expected}, actual=${actual}"
  fi
}

test_retry_succeeds_after_temporary_failures() {
  local attempts=0
  local -a observed_backoffs=()

  temporary_operation() {
    attempts=$((attempts + 1))
    ((attempts >= 3))
  }

  sleep() {
    observed_backoffs+=("$1")
  }

  log_event() {
    return 0
  }

  retry_command \
    "temporary_operation" \
    temporary_operation

  assert_equal \
    "3" \
    "$attempts" \
    "successful retry attempt count"

  assert_equal \
    "2 4" \
    "${observed_backoffs[*]}" \
    "successful retry backoff sequence"
}

test_retry_stops_after_maximum_attempts() {
  local attempts=0
  local -a observed_backoffs=()

  failing_operation() {
    attempts=$((attempts + 1))
    return 1
  }

  sleep() {
    observed_backoffs+=("$1")
  }

  log_event() {
    return 0
  }

  if retry_command \
    "failing_operation" \
    failing_operation; then
    fail_test "permanent operation failure unexpectedly succeeded"
  fi

  assert_equal \
    "4" \
    "$attempts" \
    "failed retry attempt count"

  assert_equal \
    "2 4 8" \
    "${observed_backoffs[*]}" \
    "failed retry backoff sequence"
}

test_secure_environment_file_is_accepted() {
  printf 'POSTGRES_PASSWORD=test-secret\n' > "$ENV_FILE"
  chmod 0600 "$ENV_FILE"

  log_event() {
    return 0
  }

  validate_environment_file
}

test_world_accessible_environment_file_is_rejected() {
  printf 'POSTGRES_PASSWORD=test-secret\n' > "$ENV_FILE"
  chmod 0604 "$ENV_FILE"

  if (validate_environment_file >/dev/null 2>&1); then
    fail_test "world-accessible environment file was accepted"
  fi
}

test_environment_symlink_is_rejected() {
  local target_file="${TEST_TEMP_DIRECTORY}/target.env"

  printf 'POSTGRES_PASSWORD=test-secret\n' > "$target_file"
  chmod 0600 "$target_file"

  rm -f "$ENV_FILE"
  ln -s "$target_file" "$ENV_FILE"

  if (validate_environment_file >/dev/null 2>&1); then
    fail_test "environment-file symlink was accepted"
  fi
}

write_valid_deployment_environment() {
  rm -f "$ENV_FILE"

  cat > "$ENV_FILE" <<'EOF'
AUTH_ENABLED=true
AUTH_JWT_AUDIENCE=agentflow-api
AUTH_JWT_ISSUER=https://identity.example.com/
AUTH_JWT_PUBLIC_KEY=-----BEGIN PUBLIC KEY-----\ntest-key\n-----END PUBLIC KEY-----
GITHUB_REPOSITORY_OWNER=example-owner
GITHUB_REPOSITORY_NAME=example-repository
GITHUB_DEFAULT_BRANCH=main
GITHUB_TOKEN=example-github-token
JIRA_BASE_URL=https://example.atlassian.net
JIRA_EMAIL=release.manager@example.com
JIRA_API_TOKEN=example-jira-api-token
JIRA_PROJECT_KEY=AGENT
POSTGRES_PASSWORD=production-postgres-secret
RATE_LIMIT_KEY_HMAC_SECRET=production-rate-limit-secret
REDIS_PASSWORD=production-redis-secret
TRUSTED_HOSTS=["backend","api.agentflow.example.com"]
EOF

  chmod 0600 "$ENV_FILE"
}

test_valid_deployment_environment_values_are_accepted() {
  write_valid_deployment_environment
  validate_environment_values >/dev/null
}

test_missing_required_environment_value_is_rejected() {
  write_valid_deployment_environment
  sed -i.bak '/^REDIS_PASSWORD=/d' "$ENV_FILE"
  rm -f "${ENV_FILE}.bak"

  if (validate_environment_values >/dev/null 2>&1); then
    fail_test "missing required deployment value was accepted"
  fi
}

test_placeholder_database_password_is_rejected() {
  write_valid_deployment_environment
  sed -i.bak \
    's/^POSTGRES_PASSWORD=.*/POSTGRES_PASSWORD=changeme/' \
    "$ENV_FILE"
  rm -f "${ENV_FILE}.bak"

  if (validate_environment_values >/dev/null 2>&1); then
    fail_test "placeholder PostgreSQL password was accepted"
  fi
}

test_disabled_production_authentication_is_rejected() {
  write_valid_deployment_environment
  sed -i.bak \
    's/^AUTH_ENABLED=.*/AUTH_ENABLED=false/' \
    "$ENV_FILE"
  rm -f "${ENV_FILE}.bak"

  if (validate_environment_values >/dev/null 2>&1); then
    fail_test "disabled production authentication was accepted"
  fi
}

test_local_only_trusted_hosts_are_rejected() {
  write_valid_deployment_environment
  sed -i.bak \
    's|^TRUSTED_HOSTS=.*|TRUSTED_HOSTS=["localhost","127.0.0.1"]|' \
    "$ENV_FILE"
  rm -f "${ENV_FILE}.bak"

  if (validate_environment_values >/dev/null 2>&1); then
    fail_test "local-only production trusted hosts were accepted"
  fi
}

test_non_json_trusted_hosts_are_rejected() {
  write_valid_deployment_environment
  sed -i.bak \
    's/^TRUSTED_HOSTS=.*/TRUSTED_HOSTS=api.agentflow.example.com/' \
    "$ENV_FILE"
  rm -f "${ENV_FILE}.bak"

  if (validate_environment_values >/dev/null 2>&1); then
    fail_test "non-JSON TRUSTED_HOSTS configuration was accepted"
  fi
}

test_rejected_secret_is_not_logged() {
  local rejected_secret="changeme"
  local validation_output

  write_valid_deployment_environment
  sed -i.bak \
    "s/^REDIS_PASSWORD=.*/REDIS_PASSWORD=${rejected_secret}/" \
    "$ENV_FILE"
  rm -f "${ENV_FILE}.bak"

  validation_output="$(
    validate_environment_values 2>&1 || true
  )"

  if [[ "$validation_output" == *"$rejected_secret"* ]]; then
    fail_test "rejected secret value was exposed in validation output"
  fi
}

test_compose_command_rejects_process_environment_overrides() (
  local observed_values_file
  local observed_postgres_password
  local observed_frontend_port

  observed_values_file="$(
    mktemp "${TEST_TEMP_DIRECTORY}/compose-environment.XXXXXX"
  )"

  docker() {
    printf 'POSTGRES_PASSWORD=%s\n' \
      "${POSTGRES_PASSWORD-<unset>}" \
      > "$observed_values_file"

    printf 'FRONTEND_PORT=%s\n' \
      "${FRONTEND_PORT-<unset>}" \
      >> "$observed_values_file"
  }

  export POSTGRES_PASSWORD="unsafe-process-override"
  export FRONTEND_PORT="9999"

  compose_command config --quiet

  unset POSTGRES_PASSWORD
  unset FRONTEND_PORT

  observed_postgres_password="$(
    awk -F= '
      $1 == "POSTGRES_PASSWORD" {
        print substr($0, index($0, "=") + 1)
      }
    ' "$observed_values_file"
  )"

  observed_frontend_port="$(
    awk -F= '
      $1 == "FRONTEND_PORT" {
        print substr($0, index($0, "=") + 1)
      }
    ' "$observed_values_file"
  )"

  assert_equal \
    "<unset>" \
    "$observed_postgres_password" \
    "Compose PostgreSQL process-environment override"

  assert_equal \
    "<unset>" \
    "$observed_frontend_port" \
    "Compose frontend-port process-environment override"
)


test_compose_command_rejects_collector_process_environment_overrides() (
  local observed_values_file
  local observed_github_owner
  local observed_jira_project_key

  observed_values_file="$(
    mktemp "${TEST_TEMP_DIRECTORY}/collector-environment.XXXXXX"
  )"

  docker() {
    printf 'GITHUB_REPOSITORY_OWNER=%s\n' \
      "${GITHUB_REPOSITORY_OWNER-<unset>}" \
      > "$observed_values_file"

    printf 'JIRA_PROJECT_KEY=%s\n' \
      "${JIRA_PROJECT_KEY-<unset>}" \
      >> "$observed_values_file"
  }

  export GITHUB_REPOSITORY_OWNER="unsafe-process-owner"
  export JIRA_PROJECT_KEY="UNSAFE"

  compose_command config --quiet

  unset GITHUB_REPOSITORY_OWNER
  unset JIRA_PROJECT_KEY

  observed_github_owner="$(
    awk -F= '
      $1 == "GITHUB_REPOSITORY_OWNER" {
        print substr($0, index($0, "=") + 1)
      }
    ' "$observed_values_file"
  )"

  observed_jira_project_key="$(
    awk -F= '
      $1 == "JIRA_PROJECT_KEY" {
        print substr($0, index($0, "=") + 1)
      }
    ' "$observed_values_file"
  )"

  assert_equal \
    "<unset>" \
    "$observed_github_owner" \
    "Compose GitHub-owner process-environment override"

  assert_equal \
    "<unset>" \
    "$observed_jira_project_key" \
    "Compose Jira-project process-environment override"
)

test_backend_collector_environment_is_wired() {
  local variable_name
  local -a required_variables=(
    GITHUB_REPOSITORY_OWNER
    GITHUB_REPOSITORY_NAME
    GITHUB_DEFAULT_BRANCH
    GITHUB_TOKEN
    JIRA_BASE_URL
    JIRA_EMAIL
    JIRA_API_TOKEN
    JIRA_PROJECT_KEY
  )

  for variable_name in "${required_variables[@]}"; do
    if ! grep -Eq \
      "^[[:space:]]+${variable_name}:" \
      "$BASE_COMPOSE_FILE"; then
      fail_test \
        "backend collector environment mapping missing: ${variable_name}"
    fi
  done
}

main() {
  test_retry_succeeds_after_temporary_failures
  test_retry_stops_after_maximum_attempts
  test_secure_environment_file_is_accepted
  test_world_accessible_environment_file_is_rejected
  test_environment_symlink_is_rejected
  test_compose_command_rejects_process_environment_overrides
  test_compose_command_rejects_collector_process_environment_overrides
  test_backend_collector_environment_is_wired
  test_valid_deployment_environment_values_are_accepted
  test_missing_required_environment_value_is_rejected
  test_placeholder_database_password_is_rejected
  test_disabled_production_authentication_is_rejected
  test_local_only_trusted_hosts_are_rejected
  test_non_json_trusted_hosts_are_rejected
  test_rejected_secret_is_not_logged

  printf 'EC2 deployment-script tests passed.\n'
}

main "$@"
