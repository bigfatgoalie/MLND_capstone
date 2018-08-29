# Appliance Energy Prediction

Capstone project for Udacity Machine Learning Nanodegree

I came across the following dataset during a online coding challenge.
* [Appliance Energy Data](http://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction)

This led me to explore it further and led to the follwing:
* [Research Paper](http://dx.doi.org/10.1016/j.enbuild.2017.01.083)           <br>
* [GitHub repo](https://github.com/LuisM78/Appliances-energy-prediction-data)

<img src='https://github.com/bigfatgoalie/MLND_capstone/blob/master/Corr_heatmap.png'><br>

Author's best R2 scores (benchmark):
* Training Data : 0.97
* Testing Data  : 0.58 

My model's scores :
* Training Data : 1.00
* Testing Data  : 0.60

### Software Requirements

Python version: 3.6

The software requirements can be met by installing the Anaconda distribution of python. 
* [Instructions](https://conda.io/docs/user-guide/install/index.html)

The main libraries used are:
* numpy
* pandas
* matplotlib
* seaborn
* scikit-learn
* scipy

To run the notebook, navigate to the directory and type

```jupyter notebook predict_energy_consumption.ipynb```
