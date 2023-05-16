docker build --rm -f Dockerfile -t ubuntu:pybamm_mpi .
docker run --rm -it -e "TERM=xterm-256color" --name pybamm_mpi ubuntu:pybamm_mpi
