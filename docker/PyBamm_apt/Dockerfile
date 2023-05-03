FROM ubuntu:latest

WORKDIR /repos

RUN apt-get -y update && \
    apt-get -y upgrade && \
    apt-get -y update && \
    apt-get install -y software-properties-common
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install python3 python3-dev libopenblas-dev gcc gfortran git vim
RUN apt-get install -y curl sudo build-essential

RUN useradd -rm -d /home/jsb -s /bin/zsh -G sudo jsb -u 1005
RUN echo 'jsb:jsb' | chpasswd
RUN echo 'root:root' | chpasswd
RUN adduser jsb sudo

USER jsb
WORKDIR /tmp
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3 get-pip.py

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install --user "tox<4"

WORKDIR /home/jsb
RUN git clone https://github.com/pybamm-team/PyBaMM.git
WORKDIR /home/jsb/PyBaMM

#RUN python3 -m tox -e pybamm-requires
#RUN python3 -m tox -e dev
