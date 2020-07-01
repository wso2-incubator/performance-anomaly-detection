password=test@123
docker container stop $(docker ps -a -q)
docker container prune -f
docker network prune -f
docker network create --subnet=192.168.1.0/24 myNet;

#Populating the databases
docker run -dit --cpus=1 --net myNet --ip 192.168.1.6 --name=mysqlcontainerairline -e MYSQL_ROOT_PASSWORD=test@123 -d mysql/mysql-server:latest
docker run -dit --cpus=1 --net myNet --ip 192.168.1.7 --name=mysqlcontainerhotel -e MYSQL_ROOT_PASSWORD=test@123 -d mysql/mysql-server:latest
docker run -dit --cpus=1 --net myNet --ip 192.168.1.8 --name=mysqlcontainercar -e MYSQL_ROOT_PASSWORD=test@123 -d mysql/mysql-server:latest

sleep 20
docker exec -i mysqlcontainerairline mysql -u root -p$password < mySQLairlineCommandsFile.txt
docker exec -i mysqlcontainerhotel mysql -u root -p$password < mySQLhotelCommandsFile.txt
docker exec -i mysqlcontainercar mysql -u root -p$password < mySQLcarCommandsFile.txt

#Starting the service containers
# docker container run --rm -it -v $(pwd):/home/ballerina -u $(id -u):$(id -u) -e JAVA_OPTS="-Duser.home=/home/ballerina" choreoipaas/choreo-ballerina:with-fix-for-perf-issue ballerina build --skip-tests travelAgent
ballerina build --skip-tests travelAgent
docker build -t travelobs -f src/travelAgent/docker/Dockerfile ${PWD}
docker run -dit --cpus=1 --net myNet --ip 192.168.1.2 --name=ta1 -p 9298:9298 travelobs:latest;

# docker container run --rm -it -v $(pwd):/home/ballerina -u $(id -u):$(id -u) -e JAVA_OPTS="-Duser.home=/home/ballerina" choreoipaas/choreo-ballerina:with-fix-for-perf-issue ballerina build --skip-tests mockingServiceAirline
ballerina build --skip-tests mockingServiceAirline
docker build -t airlineobs -f src/mockingServiceAirline/docker/Dockerfile ${PWD}
docker run -dit --cpus=1 --net myNet --ip 192.168.1.3 --name=mk1 airlineobs:latest;

# docker container run --rm -it -v $(pwd):/home/ballerina -u $(id -u):$(id -u) -e JAVA_OPTS="-Duser.home=/home/ballerina" choreoipaas/choreo-ballerina:with-fix-for-perf-issue ballerina build --skip-tests mockingServiceHotel
ballerina build --skip-tests mockingServiceHotel
docker build -t hotelobs -f src/mockingServiceHotel/docker/Dockerfile ${PWD}
docker run -dit --cpus=1 --net myNet --ip 192.168.1.4 --name=mk2 hotelobs:latest;

# docker container run --rm -it -v $(pwd):/home/ballerina -u $(id -u):$(id -u) -e JAVA_OPTS="-Duser.home=/home/ballerina" choreoipaas/choreo-ballerina:with-fix-for-perf-issue ballerina build --skip-tests mockingServiceCar
ballerina build --skip-tests mockingServiceCar
docker build -t carobs -f src/mockingServiceCar/docker/Dockerfile ${PWD}
docker run -dit --cpus=1 --net myNet --ip 192.168.1.5 --name=mk3 carobs:latest;