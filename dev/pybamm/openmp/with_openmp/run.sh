docker build --rm -f Dockerfile -t ubuntu:pybamm_openmp .
docker run --rm -it -e "TERM=xterm-256color" --name pybamm_openmp ubuntu:pybamm_openmp
