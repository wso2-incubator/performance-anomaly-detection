FROM openjdk:8

LABEL maintainer="dev@ballerina.io"

WORKDIR /mnt
COPY target/bin/mockingServiceHotel.jar /mnt
COPY ballerina.conf /mnt

EXPOSE 6268
CMD java -jar /mnt/mockingServiceHotel.jar