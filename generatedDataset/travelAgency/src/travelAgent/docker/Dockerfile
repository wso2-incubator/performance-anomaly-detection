FROM openjdk:8

LABEL maintainer="dev@ballerina.io"

WORKDIR /mnt
COPY target/bin/travelAgent.jar /mnt
COPY ballerina.conf /mnt

EXPOSE  9298
CMD java -jar /mnt/travelAgent.jar