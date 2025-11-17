# predict

## docker
```
docker build -t myfastapi .
docker run --rm -p 9696:9696 myfastapi
```
after spinning up the docker container,   
open another terminal, run    
`python test.py`.  

or go to 'localhost:9696/docs'.  
try it out. copy paste test sample.    

## fly.io
```
curl -L https://fly.io/install.sh | sh
export FLYCTL_INSTALL="$HOME/.fly"
export PATH="$FLYCTL_INSTALL/bin:$PATH"
flyctl auth login

flyctl launch
```
after auth login, copy url into browser to login.  

to destroy app.  
```
flyctl apps list
flyctl apps destroy <app-name>
```
in the list, find the app going to destroyed. copy the name, paste after destroy.   

![test_deployment](test_deployment.png)

![fastapi_in_cloud](fastapi_in_cloud.png)

![flyio_logs](flyio_logs.png)