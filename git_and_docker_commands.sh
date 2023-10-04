git init
git commit -am "Initial commit"
git remote add origin
git push -u origin master
git pull origin master

docker login -u nilooy5
docker build -t restaurant-modelling .
docker tag restaurant-modelling nilooy5/sydney-restaurant-modelling
docker push nilooy5/sydney-restaurant-modelling
docker run restaurant-modelling