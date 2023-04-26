FROM ubuntu:latest

RUN apt-get -y update && \
    apt-get -y upgrade && \
    apt-get -y update && \
    apt-get install -y software-properties-common
RUN apt-get -y update && \
    apt-get -y upgrade
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install zsh tmux vim python3 python3-dev python3-distutils libopenblas-dev gcc gfortran curl make cmake git build-essential
RUN apt-get -y install dvipng graphviz imagemagick
RUN apt-get -y install yarn npm

RUN useradd -rm -d /home/jsb -s /bin/zsh -G sudo jsb -u 1005
RUN echo 'jsb:jsb' | chpasswd
RUN echo 'root:root' | chpasswd

USER jsb
WORKDIR /tmp
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3 get-pip.py

WORKDIR /home/jsb

CMD ["zsh", "-c", "tmux"]
