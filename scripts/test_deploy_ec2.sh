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

main() {
  test_retry_succeeds_after_temporary_failures
  test_retry_stops_after_maximum_attempts
  test_secure_environment_file_is_accepted
  test_world_accessible_environment_file_is_rejected
  test_environment_symlink_is_rejected

  printf 'EC2 deployment-script tests passed.\n'
}

main "$@"
