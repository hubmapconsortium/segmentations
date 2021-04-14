FROM nvidia/cuda:10.2-cudnn8-runtime-ubuntu18.04


RUN apt-get -qq update \
    && apt-get -qq install --no-install-recommends --yes \
    build-essential \
    wget \
    bzip2 \
    ca-certificates \
    curl \
    unzip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && /bin/bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh
ENV PATH /opt/conda/bin:$PATH

# update base environment from yaml file
COPY environment.yml /tmp/
RUN conda env update -f /tmp/environment.yml \
    && echo "source activate base" > ~/.bashrc \
    && conda clean --index-cache --tarballs --yes \
    && rm /tmp/environment.yml

CMD ["/bin/bash"]
