FROM ubuntu:bionic

RUN apt-get -qq update \
    && apt-get -qq install --no-install-recommends --yes \
    build-essential \
    wget \
    bzip2 \
    tar \
    ca-certificates \
    curl \
    unzip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && /bin/bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh
ENV PATH /opt/conda/bin:$PATH

RUN mkdir /output && chmod -R a+rwx /output

# update base environment from yaml file
COPY environment.yml /tmp/
RUN conda env update -f /tmp/environment.yml \
    && echo "source activate base" > ~/.bashrc \
    && conda clean --index-cache --tarballs --yes \
    && rm /tmp/environment.yml

# download cellpose models
ENV CELLPOSE_LOCAL_MODELS_PATH=/.cellpose/models
RUN mkdir -p /.cellpose/models \
    && cd /.cellpose/models \
    && wget http://www.cellpose.org/models/nuclei_0 \
    && wget http://www.cellpose.org/models/nuclei_1 \
    && wget http://www.cellpose.org/models/nuclei_2 \
    && wget http://www.cellpose.org/models/nuclei_3 \
    && wget http://www.cellpose.org/models/size_nuclei_0.npy \
    && wget https://www.cellpose.org/models/cyto_0 \
    && wget https://www.cellpose.org/models/cyto_1 \
    && wget https://www.cellpose.org/models/cyto_2 \
    && wget https://www.cellpose.org/models/cyto_3 \
    && wget https://www.cellpose.org/models/size_cyto_0.npy \
    && wget https://www.cellpose.org/models/cytotorch_0 \
    && wget https://www.cellpose.org/models/cytotorch_1 \
    && wget https://www.cellpose.org/models/cytotorch_2 \
    && wget https://www.cellpose.org/models/cytotorch_3 \
    && wget https://www.cellpose.org/models/size_cytotorch_0.npy \
    && wget https://www.cellpose.org/models/nucleitorch_0 \
    && wget https://www.cellpose.org/models/nucleitorch_1 \
    && wget https://www.cellpose.org/models/nucleitorch_2 \
    && wget https://www.cellpose.org/models/nucleitorch_3 \
    && wget https://www.cellpose.org/models/size_nucleitorch_0.npy

# download deepcell models
RUN mkdir -p /.keras/models \
    && cd /.keras/models \
    && wget https://deepcell-data.s3-us-west-1.amazonaws.com/saved-models/MultiplexSegmentation-7.tar.gz \
    && tar -xvzf MultiplexSegmentation-7.tar.gz \
    && rm MultiplexSegmentation-7.tar.gz \
    && ln -s /.keras /root

COPY keras.json /.keras/.
COPY bin /opt


CMD ["/bin/bash"]
