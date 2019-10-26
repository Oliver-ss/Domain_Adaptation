### README 
Every diretory represents a trial, now the baseline model is **xh.deeplab.mobilenet.&lt;city name&gt;**, city name means the source domain


To train the model, run ```make train```


To watch the results of every epoch on the validation dataset and delete the bad models, run ```make watch```


To test the model, run ```make test MODEL=<path where the model is saved>```

For the details, please directly check the python scripts. 
