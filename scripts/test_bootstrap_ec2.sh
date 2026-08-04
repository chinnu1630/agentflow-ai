#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIRECTORY="$(
  cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1
  pwd
)"
readonly SCRIPT_DIRECTORY

# Loading the script defines its functions but does not execute main().
# shellcheck source=scripts/bootstrap_ec2.sh
source "${SCRIPT_DIRECTORY}/bootstrap_ec2.sh"

# The production script installs an ERR trap. Unit tests handle failures
# explicitly, so remove that trap for predictable assertions.
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

assert_file_absent() {
  local file_path="$1"
  local description="$2"

  if [[ -e "$file_path" ]]; then
    fail_test "${description}: unexpected file=${file_path}"
  fi
}

test_download_succeeds_after_retries() {
  local temporary_directory
  temporary_directory="$(mktemp -d)"

  local destination="${temporary_directory}/docker.asc"
  local attempts=0
  local -a observed_backoffs=()

  curl() {
    local output_path=""

    attempts=$((attempts + 1))

    while (($# > 0)); do
      if [[ "$1" == "--output" ]]; then
        output_path="$2"
        break
      fi

      shift
    done

    if [[ -z "$output_path" ]]; then
      return 2
    fi

    if ((attempts < 3)); then
      return 22
    fi

    printf 'trusted-signing-key\n' > "$output_path"
  }

  sleep() {
    observed_backoffs+=("$1")
  }

  log_event() {
    return 0
  }

  download_file_with_retry \
    "https://example.invalid/docker.asc" \
    "$destination"

  assert_equal \
    "3" \
    "$attempts" \
    "successful download attempt count"

  assert_equal \
    "2 4" \
    "${observed_backoffs[*]}" \
    "successful download backoff sequence"

  assert_equal \
    "trusted-signing-key" \
    "$(tr -d '\n' < "$destination")" \
    "downloaded file content"

  assert_file_absent \
    "${destination}.partial" \
    "successful download partial-file cleanup"

  rm -rf "$temporary_directory"
}

test_download_stops_after_maximum_attempts() {
  local temporary_directory
  temporary_directory="$(mktemp -d)"

  local destination="${temporary_directory}/docker.asc"
  local attempts=0
  local -a observed_backoffs=()

  curl() {
    attempts=$((attempts + 1))
    return 22
  }

  sleep() {
    observed_backoffs+=("$1")
  }

  log_event() {
    return 0
  }

  if download_file_with_retry \
    "https://example.invalid/docker.asc" \
    "$destination"; then
    fail_test "permanent download failure unexpectedly succeeded"
  fi

  assert_equal \
    "5" \
    "$attempts" \
    "failed download attempt count"

  assert_equal \
    "2 4 8 16" \
    "${observed_backoffs[*]}" \
    "failed download backoff sequence"

  assert_file_absent \
    "$destination" \
    "failed download destination cleanup"

  assert_file_absent \
    "${destination}.partial" \
    "failed download partial-file cleanup"

  rm -rf "$temporary_directory"
}

main() {
  test_download_succeeds_after_retries
  test_download_stops_after_maximum_attempts

  printf 'EC2 bootstrap retry tests passed.\n'
}

main "$@"
