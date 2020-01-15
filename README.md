# Multi-Temporal-Crop-Classification (work-in-progress)

* Recreating code for the paper "Deep Learning Based Multi-Temporal Crop Classification"

* There are a few minor differences - Zhong et al. use 46 element vector as time series, I use MODIS 16 day composite NDVI data (23 entries per year).

* Paper reference: "Zhong, Liheng, Lina Hu, and Hang Zhou. "Deep learning based multi-temporal crop classification." Remote sensing of environment 221 (2019): 430-443."

* ```cnn-network.py``` and ````lstm-network.py``` contain code for cnn and lstm described in the paper

* Platform: Python3, Keras, Tensorflow

* Use Hufkens et al. to download MODIS data.


References:
* Hufkens K. (2017) A Google Earth Engine time series subset script & library. DOI: 10.5281/zenodo.833789
* Zhong, Liheng, Lina Hu, and Hang Zhou. "Deep learning based multi-temporal crop classification." Remote sensing of environment 221 (2019): 430-443.

