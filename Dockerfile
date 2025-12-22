FROM ubuntu:latest
LABEL authors="danka"

ENTRYPOINT ["top", "-b"]