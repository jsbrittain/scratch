FROM ubuntu:latest

RUN apt-get -y update && \
    apt-get -y upgrade && \
    apt-get -y update && \
    apt-get install -y software-properties-common
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install zsh tmux vim python3 python3-dev python3-distutils libopenblas-dev gcc gfortran curl make cmake git build-essential texlive-latex-recommended texlive-latex-extra
RUN apt-get -y install dvipng graphviz
RUN apt-get -y install tree git

RUN useradd -rm -d /home/jsb -s /bin/zsh -G sudo jsb
RUN echo 'jsb:jsb' | chpasswd
RUN echo 'root:root' | chpasswd

USER jsb
WORKDIR /home/jsb

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3 get-pip.py
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install "tox<4"

RUN git clone https://github.com/jsbrittain/PyBaMM.git
WORKDIR /home/jsb/PyBaMM

# Install from pre-openmp commit
#RUN git pull
#RUN git checkout 7c9bcfb
#RUN python3 -m tox -e pybamm-requires
#RUN python3 -m tox -e dev

# Update source code (without reinstall)
#RUN git checkout develop
#COPY --chown=jsb:jsb ./PyBaMM/./ .
