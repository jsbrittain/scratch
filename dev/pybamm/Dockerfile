FROM ubuntu:latest

RUN apt-get -y update && \
    apt-get -y upgrade && \
    apt-get -y update && \
    apt-get install -y software-properties-common
RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt-get -y update && \
    apt-get -y upgrade
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install zsh tmux vim python3.9 python3.9-dev python3.9-distutils libopenblas-dev gcc gfortran curl make cmake git build-essential texlive-latex-recommended texlive-latex-extra
RUN apt-get -y install dvipng graphviz
RUN apt-get -y install vim-nox mono-complete golang nodejs openjdk-17-jdk openjdk-17-jre npm
RUN apt-get -y install tree

RUN useradd -rm -d /home/jsb -s /bin/zsh -G sudo jsb -u 1005
RUN echo 'jsb:jsb' | chpasswd
RUN echo 'root:root' | chpasswd

USER jsb
WORKDIR /tmp
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.9 get-pip.py

WORKDIR /home/jsb

CMD ["zsh", "-c", "tmux"]
