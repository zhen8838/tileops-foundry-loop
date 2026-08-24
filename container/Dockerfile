ARG BASE_IMAGE
FROM ${BASE_IMAGE}

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends openssh-server \
    && rm -rf /var/lib/apt/lists/*

# The TileOPs runner already carries the libisl build required by isl-python.
# Keep it ahead of Ubuntu's older ABI-compatible soname.
ENV LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH}

COPY sshd_config /etc/ssh/sshd_config
COPY entrypoint.sh /usr/local/bin/tileops-ssh-entrypoint
RUN chmod 0755 /usr/local/bin/tileops-ssh-entrypoint

CMD ["/usr/local/bin/tileops-ssh-entrypoint"]
