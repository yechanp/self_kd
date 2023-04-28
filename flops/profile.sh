

# nvprof -f -o net.sql --profile-from-start off sh train.sh

# sh train.sh

nsys profile -f true -o net --export sqlite bash ./train.sh


python -m pyprof.parse net.sqlite > net.dict
python -m pyprof.prof --csv -c idx,dir,op,kernel,params,sil,flops,bytes net.dict > results.csv
rm net.dict net.sqlite net.qdrep
