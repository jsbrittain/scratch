docker build --rm -f Dockerfile -t ubuntu:pybamm_noopenmp .
docker run --rm -it -e "TERM=xterm-256color" --name pybamm_noopenmp ubuntu:pybamm_noopenmp
