FROM openjdk:8

LABEL maintainer="dev@ballerina.io"

WORKDIR /mnt
COPY target/bin/mockingServiceAirline.jar /mnt
COPY ballerina.conf /mnt

EXPOSE 7278
CMD java -jar /mnt/mockingServiceAirline.jar