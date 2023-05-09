#!/usr/bin/env bash
set -euxo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
pushd $SCRIPT_DIR

for dir in */
do
    dir=${dir%*/}
    echo "${dir##*/}"
    pushd $dir
    git pull
    popd
done

popd
