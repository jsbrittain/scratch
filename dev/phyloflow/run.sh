podman build --rm -f Dockerfile -t ubuntu:phyloflow dev/phyloflow/.
podman run --rm -it --userns=keep-id -e "TERM=xterm-256color" -e DISPLAY -e "HOST=$HOSTNAME:podman" --mount src=/home/jsb,target=/home/jsb,type=bind --net=host ubuntu:phyloflow
