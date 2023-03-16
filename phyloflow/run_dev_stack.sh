#/usr/bin/env bash

set -euox pipefail

pushd flask
./run.sh &

popd
pushd nodemapper
./run.sh &
popd

