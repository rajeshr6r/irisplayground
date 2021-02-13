# Produtionizing ML models with single page web apps

## Technologies used 
<br>

>Docker - For containerizing all necessary requirements for the app

>GitHub Docker Integration - For automatic image build with repository commit

> Python Flask - For rendering a single page app with the models loaded on the background 

> AWS ECS - To demonstrate how to consume a docker container on internet.

> HTML - For the single web app

<br>

## Disclaimer 

This repository is not a Machine Learning Tutorial . But a tutorial to demonstrate how quickly one can productionize Machine Learing Models with minimal effort.

Also for dockers I'm running a limited Raspbian Stretch image to keep it light . Note that it is not always the right choice . For example if you are running libraries like H2O , Tensorflow or PyTorch then the Raspbian image can make life very difficult. Try at your own risk. 


<br>


## Shameless Marketing :)

> If you find this useful you can consider following me on [Medium](https://rajesh-r6r.medium.com/) 

<br>

## Helpful Reference URLs

[IRIS Dataset gist](https://gist.github.com/curran/a08a1080b88344b0c8a7)

[IRIS Dataset UCI Machine Learning Reposiroty](https://archive.ics.uci.edu/ml/datasets/iris)

And ofcourse [Wikipedia](https://en.wikipedia.org/wiki/Iris_flower_data_set) :)


## Wish to cut the english and jump into action ? 

> You can do that by simply pulling the docker image with the following command 

> docker pull rajeshr6r/irisplayground:latest
