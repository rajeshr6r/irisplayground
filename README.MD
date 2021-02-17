# Produtionizing ML models with single page web apps

## Technologies used 
<br>

>Docker - For containerizing all necessary requirements for the app

>GitHub Docker Integration - For automatic image build with repository commit

> Python Flask - For rendering a single page app with the models loaded on the background 

> Azure Container Instanes - To demonstrate how to consume a docker container on internet.

> HTML - For the single web app

<br>

## Disclaimer 

This repository is not a Machine Learning Tutorial . But a tutorial to demonstrate how quickly one can productionize Machine Learing Models with minimal effort.

Also for dockers I'm running a limited base UBUNTU image . Note that it is not always the right choice . For example if you are running libraries like H2O , Tensorflow or PyTorch then the image can get heavier. And slimming up your docker image is an art by itself . Happy Hunting . 

Please drop a note if you find anything simpler.

Update : 14Feb2021 : I used a python3 image that under the hoods use a debian buster image . However this script might make it helpful for someone who is from a non-linux side. It simply uses the python and its dependencies .




<br>


## Shameless Marketing :)

> If you find this useful you can consider following me on [Medium](https://rajesh-r6r.medium.com/) 

<br>

## Helpful Reference URLs

[IRIS Dataset gist](https://gist.github.com/curran/a08a1080b88344b0c8a7)

[IRIS Dataset UCI Machine Learning Reposiroty](https://archive.ics.uci.edu/ml/datasets/iris)

And ofcourse [Wikipedia](https://en.wikipedia.org/wiki/Iris_flower_data_set) :)


## Wish to cut the english and jump into action ? 

> You can do that by simply pulling the docker image with the following command and then running it as follows

~~~
docker pull rajeshr6routlook/irisplayground:latest

docker run -itd -p 8000:8000 rajeshr6routlook/irisplayground:latest
~~~

> or if you'd like the python only ( which is not true as underneath the python image uses debian buster )

~~~ 
docker pull rajeshr6routlook/pythonirisplayground:latest

docker run -itd -p 8000:8000 rajeshr6routlook/pythonirisplayground:latest

~~~

