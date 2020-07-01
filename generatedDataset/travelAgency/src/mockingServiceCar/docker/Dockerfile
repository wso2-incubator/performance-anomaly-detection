FROM openjdk:8

LABEL maintainer="dev@ballerina.io"

WORKDIR /mnt
COPY target/bin/mockingServiceCar.jar /mnt
COPY ballerina.conf /mnt

EXPOSE 5258
CMD java -jar /mnt/mockingServiceCar.jar
