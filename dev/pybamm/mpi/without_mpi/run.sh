docker build --rm -f Dockerfile -t ubuntu:pybamm_nompi .
docker run --rm -it -e "TERM=xterm-256color" --name pybamm ubuntu:pybamm_nompi
