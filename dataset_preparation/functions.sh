#!/usr/bin/env bash

urlencode_() {
  local length="${#1}"
  for ((i = 0; i < length; i++)); do
    local c="${1:i:1}"
    case $c in
    [a-zA-Z0-9.~_-]) printf '%s' "$c" ;;
    *) printf '%%%02X' "'$c" ;;
    esac
  done
}

urlencode() {
  LC_COLLATE=C urlencode_ "$1"
}

check_data_root() {
  if [[ -z ${DATA_ROOT+x} ]]; then
    echo "Set the environment variable DATA_ROOT to some path." \
      "All datasets will be downloaded under \$DATA_ROOT"
    exit 1
  fi

  if [[ ! -d $DATA_ROOT ]]; then
    echo "$DATA_ROOT is not a directory!"
    exit 1
  fi

  if [[ ! -w $DATA_ROOT ]]; then
    echo "$DATA_ROOT is not writable!"
    exit 1
  fi
}

ask() {
  while true; do
    read -rp "$1" yn; case $yn in [Yy]*) echo y; break;; [Nn]*) echo n; break;;
    *) echo "Please answer yes or no.";;
    esac
  done
}

check_md5() {
  echo "$2 $1" | md5sum --status -c
}

get_script_dir() {
  dirname "$(readlink -f "${BASH_SOURCE[0]}")"
}
