#!/usr/bin/env bash

set -Eeuo pipefail

TEST_ROLLBACK_SCRIPT_DIRECTORY="$(
  cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1
  pwd
)"
readonly TEST_ROLLBACK_SCRIPT_DIRECTORY

# Loading the rollback script defines its functions without running main().
# shellcheck source=scripts/rollback_ec2.sh
source "${TEST_ROLLBACK_SCRIPT_DIRECTORY}/rollback_ec2.sh"

# Expected validation failures are handled explicitly by individual tests.
trap - ERR

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

assert_command_fails() {
  local description="$1"
  shift

  if ("$@" >/dev/null 2>&1); then
    fail_test "${description}: command unexpectedly succeeded"
  fi
}

reset_argument_state() {
  ROLLBACK_TARGET_COMMIT=""
  ROLLBACK_CONFIRMED="false"
}

test_target_and_confirmation_are_accepted() {
  reset_argument_state

  parse_arguments \
    --target \
    HEAD~1 \
    --confirm

  assert_equal \
    "HEAD~1" \
    "$ROLLBACK_TARGET_COMMIT" \
    "parsed rollback target"

  assert_equal \
    "true" \
    "$ROLLBACK_CONFIRMED" \
    "parsed rollback confirmation"
}

test_missing_target_is_rejected() {
  reset_argument_state

  assert_command_fails \
    "missing rollback target" \
    parse_arguments \
    --confirm
}

test_missing_confirmation_is_rejected() {
  reset_argument_state

  assert_command_fails \
    "missing rollback confirmation" \
    parse_arguments \
    --target \
    HEAD~1
}

test_unknown_argument_is_rejected() {
  reset_argument_state

  assert_command_fails \
    "unknown rollback argument" \
    parse_arguments \
    --force
}

test_invalid_commit_is_rejected() {
  ROLLBACK_TARGET_COMMIT="not-a-real-agentflow-commit"

  assert_command_fails \
    "invalid rollback commit" \
    validate_rollback_target
}

test_current_commit_is_rejected() {
  ROLLBACK_TARGET_COMMIT="$(
    git -C "$REPOSITORY_ROOT" rev-parse HEAD
  )"

  assert_command_fails \
    "current rollback commit" \
    validate_rollback_target
}

test_older_ancestor_commit_is_accepted() {
  local expected_target

  expected_target="$(
    git -C "$REPOSITORY_ROOT" rev-parse HEAD~1
  )"
  ROLLBACK_TARGET_COMMIT="$expected_target"

  log_event() {
    return 0
  }

  validate_rollback_target

  assert_equal \
    "$expected_target" \
    "$ROLLBACK_TARGET_COMMIT" \
    "resolved ancestor rollback target"
}

test_application_stop_preserves_volumes() {
  local -a compose_arguments=()

  compose_command() {
    compose_arguments=("$@")
  }

  log_event() {
    return 0
  }

  stop_application_preserving_data

  assert_equal \
    "down --remove-orphans" \
    "${compose_arguments[*]}" \
    "volume-preserving Compose shutdown"

  for argument in "${compose_arguments[@]}"; do
    if [[ "$argument" == "--volumes" || "$argument" == "-v" ]]; then
      fail_test "rollback shutdown unexpectedly removes Docker volumes"
    fi
  done
}

main() {
  test_target_and_confirmation_are_accepted
  test_missing_target_is_rejected
  test_missing_confirmation_is_rejected
  test_unknown_argument_is_rejected
  test_invalid_commit_is_rejected
  test_current_commit_is_rejected
  test_older_ancestor_commit_is_accepted
  test_application_stop_preserves_volumes

  printf 'EC2 rollback-script tests passed.\n'
}

main "$@"
