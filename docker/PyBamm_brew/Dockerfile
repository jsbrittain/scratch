FROM homebrew/brew:latest

WORKDIR /repos

RUN brew update
RUN brew install python openblas gcc gfortran git
RUN brew install sundials

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install --user "tox<4"

WORKDIR /home/linuxbrew/
RUN git clone https://github.com/pybamm-team/PyBaMM.git
WORKDIR /home/linuxbrew/PyBaMM
RUN git clone https://github.com/pybind/pybind11.git

RUN brew install vim

#RUN python3 -m tox -e dev
