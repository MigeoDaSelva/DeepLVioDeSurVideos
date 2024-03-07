FROM tensorflow/tensorflow:latest-gpu-jupyter

ARG USERNAME
ARG USER_UID
ARG USER_GID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

RUN DEBIAN_FRONTEND=noninteractive apt install -y tzdata
RUN apt install -y software-properties-common

COPY ../requirements.txt ./requirements.txt

RUN pip install -r requirements.txt

RUN rm -f ./requirements.txt

USER $USERNAME