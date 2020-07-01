password=test@123
docker container stop $(docker ps -a -q)
docker container prune -f
# docker network prune -f
# docker network create --subnet=192.168.1.0/24 myNet;

#Populating the databases
docker run -dit --cpus=1 --name=mysqlcontainerairline -p 3300:3306 -e MYSQL_ROOT_PASSWORD=test@123 -d mysql/mysql-server:latest
docker run -dit --cpus=1 --name=mysqlcontainerhotel -p 3301:3306 -e MYSQL_ROOT_PASSWORD=test@123 -d mysql/mysql-server:latest
docker run -dit --cpus=1 --name=mysqlcontainercar -p 3302:3306 -e MYSQL_ROOT_PASSWORD=test@123 -d mysql/mysql-server:latest

sleep 20
docker exec -i mysqlcontainerairline mysql -u root -p$password < mySQLairlineCommandsFile.txt
docker exec -i mysqlcontainerhotel mysql -u root -p$password < mySQLhotelCommandsFile.txt
docker exec -i mysqlcontainercar mysql -u root -p$password < mySQLcarCommandsFile.txt

#Starting the service containers
ballerina build --skip-tests travelAgent
docker build -t travelobs -f src/travelAgent/docker/Dockerfile ${PWD}
docker run -dit --cpus=1 --name=ta1 -p 9298:9298 travelobs:latest;

ballerina build --skip-tests mockingServiceAirline
docker build -t airlineobs -f src/mockingServiceAirline/docker/Dockerfile ${PWD}
docker run -dit --cpus=1 --name=mk1 -p 7278:7278 airlineobs:latest;

ballerina build --skip-tests mockingServiceHotel
docker build -t hotelobs -f src/mockingServiceHotel/docker/Dockerfile ${PWD}
docker run -dit --cpus=1 --name=mk2 -p 6268:6268 hotelobs:latest;

ballerina build --skip-tests mockingServiceCar
docker build -t carobs -f src/mockingServiceCar/docker/Dockerfile ${PWD}
docker run -dit --cpus=1 --name=mk3 -p 5258:5258 carobs:latest;