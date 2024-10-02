FROM python:3.10.14-bookworm
COPY --from=ghcr.io/astral-sh/uv:0.4.5 /uv /bin/uv

ARG DEBIAN_FRONTEND=noninteractive
ARG TZ=Etc/UTC

# Install apt packages.
RUN apt-get update \
    && apt-get -y upgrade \
    && apt-get -y install --no-install-recommends \
        libosmesa-dev \
        vim \
        wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install MuJoCo 2.1.0 for D4RL.
ARG MUJOCO_BASE_PATH=/root/.mujoco
ENV MUJOCO_PY_MUJOCO_PATH=$MUJOCO_BASE_PATH/mujoco210
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MUJOCO_PY_MUJOCO_PATH/bin
RUN mkdir -p $MUJOCO_PY_MUJOCO_PATH \
    && wget -q -P /tmp https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz \
    && tar --no-same-owner -xf /tmp/mujoco210-linux-x86_64.tar.gz -C $MUJOCO_BASE_PATH

# Since uv's project system uses virtual environment, and DevContainer mounts the
# project directory, including the virtual environment, we don't need to run `uv sync`
# to install Python packages in the Dockerfile. Instead, we can install after
# the container is created.