FROM python:3.8-slim-bookworm

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        wget \
        git \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Create madsci user matching host UID/GID
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd -g ${GROUP_ID} madsci && \
    useradd -m -u ${USER_ID} -g ${GROUP_ID} -s /bin/bash madsci

# Install Miniconda (architecture-aware)
ARG TARGETARCH
RUN CONDA_ARCH=$([ "$TARGETARCH" = "amd64" ] && echo "x86_64" || echo "aarch64") && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-${CONDA_ARCH}.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh

ENV PATH="/opt/conda/bin:$PATH"

# Create conda environment with Python 3.9
RUN conda create -n rosenv python=3.9 -c conda-forge --override-channels -y

SHELL ["conda", "run", "-n", "rosenv", "/bin/bash", "-c"]

# Install ONLY communication infrastructure from RoboStack (no message packages)
# Message packages will be mounted from host for MD5 compatibility
RUN conda install -y \
    ros-noetic-rospy \
    ros-noetic-actionlib \
    -c robostack \
    -c conda-forge \
    --override-channels

# Fix pyassimp for MoveIt mesh support
RUN pip install pyassimp==4.1.3

# Clone MADSci repository at v0.5.0 release tag
WORKDIR /home/madsci
RUN git clone --depth 1 --branch v0.5.0 https://github.com/AD-SDL/MADSci.git && \
    chown -R madsci:madsci /home/madsci/MADSci

# Install MADSci as editable packages in conda environment
RUN pip install -e /home/madsci/MADSci/src/madsci_common && \
    pip install -e /home/madsci/MADSci/src/madsci_client && \
    pip install -e /home/madsci/MADSci/src/madsci_node_module

# Clean up to reduce image size
RUN conda clean -afy && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Switch to madsci user
USER madsci
WORKDIR /home/madsci

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "rosenv"]
CMD ["/bin/bash"]
