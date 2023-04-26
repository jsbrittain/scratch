podman build --rm -f Dockerfile -t ubuntu:jsb dev/pybamm/.
podman run --rm -it --userns=keep-id -e "TERM=xterm-256color" -e "HOST=$HOSTNAME:podman" --mount src=/home/jsb,target=/home/jsb,type=bind ubuntu:jsb
