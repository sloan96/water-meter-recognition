## Build Server
$ docker build -t watermeter-server-predict.cpu -f Dockerfile.server.cpu .

## Run Server
$ docker run -p 21327:21327 -it --rm watermeter-server-predict.cpu

## Build Client
$ docker build -t watermeter-client-app -f Dockerfile.client .

## Run Client
|服务器   |命令|
|--------|---|
|阿里云   | $ docker run -e WATER_METER_SERVER=watermeter.houghlines.cn -it --rm --name watermeter-client-app watermeter-client-app |
|本地     | $ docker run -e WATER_METER_SERVER={HostPC_IPaddress} -it --rm --name watermeter-client-app watermeter-client-app |

以上。
