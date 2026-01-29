FROM ubuntu:22.04

# Set environment variables to prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV APT_LISTCHANGES_FRONTEND=none

# Ensure package lists are clean and sources are valid
RUN set -e && \
    apt-get update && \
    apt-get upgrade -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        apt-transport-https \
        ca-certificates \
        curl \
        nodejs \
        python3 \
        python3-pip \
        python3-dev \
        build-essential \
        gdebi-core \
        wget \
        unzip \
        less \
        adb && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Android Platform Tools for ADB
RUN mkdir -p /usr/local/android-sdk && \
    cd /usr/local/android-sdk/ && \
    wget --progress=dot:giga -O platform-tools-latest-linux.zip https://dl.google.com/android/repository/platform-tools-latest-linux.zip && \
    unzip -qo platform-tools-latest-linux.zip && \
    rm -f platform-tools-latest-linux.zip && \
    rm -f /usr/bin/adb && \
    ln -s /usr/local/android-sdk/platform-tools/adb /usr/bin/adb

# Define environment variables
ENV QNN_SDK_VERSION=2.37.1.250807
ENV QNN_INSTALLATION_DIR="/opt/qnn/"
ENV QNN_SDK_ROOT="${QNN_INSTALLATION_DIR}/${QNN_SDK_VERSION}"

# Download and install QNN SDK
RUN QNN_SDK_VERSION=2.37.1.250807 && \
    QNN_SDK_ZIP="qnn2.37.1.250807.zip" && \
    QNN_SDK_URL="https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/2.37.1.250807/v2.37.1.250807.zip" && \
    echo "Downloading QNN SDK..." && \
    wget --progress=dot:giga -O "/tmp/${QNN_SDK_ZIP}" "${QNN_SDK_URL}" && \
    mkdir -p "${QNN_INSTALLATION_DIR}" && \
    unzip -qo "/tmp/${QNN_SDK_ZIP}" -d /tmp/qnn_sdk && \
    if [ -d /tmp/qnn_sdk/qairt ]; then \
        mv /tmp/qnn_sdk/qairt/* "${QNN_INSTALLATION_DIR}"; \
    else \
        mv /tmp/qnn_sdk/* "${QNN_INSTALLATION_DIR}"; \
    fi && \
    echo "QNN installed in ${QNN_INSTALLATION_DIR}" && \
    ls -lah "${QNN_INSTALLATION_DIR}" && \
    rm -rf /tmp/qnn_sdk /tmp/${QNN_SDK_ZIP}

# Install Linux dependencies for QNN SDK
RUN bash -c "${QNN_SDK_ROOT}/bin/check-linux-dependency.sh"

# Install Python 3.10 and dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-distutils \
    python3.10-venv \
    libpython3.10 \
    && rm -rf /var/lib/apt/lists/*

ENV MY_ENV_NAME=qnn237_env
# Set up Python virtual environment
WORKDIR $QNN_SDK_ROOT
RUN python3.10 -m venv ${MY_ENV_NAME} && \
    bash -c ". ${MY_ENV_NAME}/bin/activate && python3 -m ensurepip"

# Ensure pip3 is available within the virtual environment
RUN bash -c ". ${QNN_SDK_ROOT}/${MY_ENV_NAME}/bin/activate && which pip3"

# Preinstall a matplotlib build that supports Python 3.10 to avoid source build failures in the QNN checker
RUN bash -c ". ${QNN_SDK_ROOT}/${MY_ENV_NAME}/bin/activate && \
    pip install --no-cache-dir matplotlib==3.8.4"

# Update Python dependencies for QNN SDK
RUN bash -c ". ${QNN_SDK_ROOT}/${MY_ENV_NAME}/bin/activate && \
    python ${QNN_SDK_ROOT}/bin/check-python-dependency"

# Install model frameworks inside the virtual environment
RUN bash -c ". ${QNN_SDK_ROOT}/${MY_ENV_NAME}/bin/activate && \
    python -m pip install --no-cache-dir \
    torch==1.13.1 \
    torchvision==0.14.1 \
    tflite==2.10 \
    onnx==1.12.0 \
    onnxruntime==1.17.1 \
    onnxsim==0.4.36 \
    typer==0.15.1 \
    adbutils \
    robotframework \
    numpy \
    opencv-python \
    pure-python-adb \
    pre-commit==2.13.0 \
    pre-commit-hooks==4.0.1 \
    isort==5.6.4 \
    flake8==3.9.2 \
    flake8-bugbear==21.4.3 \
    flake8-comprehensions==3.6.0 \
    autoflake==1.4 \
    black==22.3.0 \
    pydocstyle==6.1.1 \
    pep8-naming==0.12.1 \
    mypy==0.910"

# Create symbolic link for tflite_runtime inside the virtual environment
RUN bash -c ". ${QNN_SDK_ROOT}/${MY_ENV_NAME}/bin/activate && \
    ln -s \$(python -c \"import tflite_runtime; print(tflite_runtime.__path__[0])\") \
    ${QNN_SDK_ROOT}/${MY_ENV_NAME}/lib/python3.10/site-packages/tflite"

# Install Android NDK r26c
ENV ANDROID_NDK_VERSION=r26c
ENV ANDROID_NDK_ROOT=/opt/android-ndk-${ANDROID_NDK_VERSION}

RUN ANDROID_NDK_ZIP="android-ndk-${ANDROID_NDK_VERSION}-linux.zip" && \
    ANDROID_NDK_URL="https://dl.google.com/android/repository/${ANDROID_NDK_ZIP}" && \
    echo "Downloading Android NDK..." && \
    wget --progress=dot:giga -O "/tmp/${ANDROID_NDK_ZIP}" "${ANDROID_NDK_URL}" && \
    unzip -qo "/tmp/${ANDROID_NDK_ZIP}" -d /opt && \
    rm -rf "/tmp/${ANDROID_NDK_ZIP}"

# Copy scripts and make them executable
COPY scripts/ /opt/scripts/
RUN chmod +x /opt/scripts/*.sh

# Copy and extract pySecureMR
COPY pySecureMR-dev.tar.gz /tmp/
RUN cd /tmp && \
    mkdir -p /tmp/pySecureMR-dev && \
    tar xzf pySecureMR-dev.tar.gz -C /tmp/pySecureMR-dev && \
    mkdir -p /pySecureMR && \
    if [ -d /tmp/pySecureMR-dev/pySecureMR-dev ]; then \
        mv /tmp/pySecureMR-dev/pySecureMR-dev/* /pySecureMR/; \
    elif [ -d /tmp/pySecureMR-dev/pySecureMR ]; then \
        mv /tmp/pySecureMR-dev/pySecureMR/* /pySecureMR/; \
    else \
        mv /tmp/pySecureMR-dev/* /pySecureMR/; \
    fi && \
    cd /pySecureMR && \
    pip3 install --no-build-isolation --no-deps . && \
    rm -rf /tmp/pySecureMR-dev.tar.gz /tmp/pySecureMR-dev

# Add QNN SDK binaries to PATH for global access
ENV PATH=$QNN_SDK_ROOT/bin:$QNN_SDK_ROOT/bin/x86_64-linux-clang:${QNN_SDK_ROOT}/${MY_ENV_NAME}/bin:$ANDROID_NDK_ROOT:/usr/local/android-sdk/platform-tools:/opt/scripts:$PATH

# Add environment setup to bashrc
RUN echo "source ${QNN_SDK_ROOT}/bin/envsetup.sh" >> /root/.bashrc && \
    echo "source ${QNN_SDK_ROOT}/${MY_ENV_NAME}/bin/activate" >> /root/.bashrc

# Set working directory inside the container
WORKDIR /app

# Copy app config files
RUN mkdir -p /opt/config
COPY HtpConfigFile.json /opt/config/HtpConfigFile.json
COPY PerfSetting.conf /opt/config/PerfSetting.conf
