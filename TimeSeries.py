import numpy as np
import pandas as pd

class TimeSeries:
    def __init__(self, rv_name, rv,time=None, covariate_name=None, covariate=None):
        self.time = time
        self.covariate = covariate
        self.covariate_name = covariate_name
        self.rv = rv
        self.rv_name = rv_name


    def split(self, split = True):
        if split and self.covariate_name is not None:
            index = np.array(self.rv) > 0
            if max(self.rv) > 0 > min(self.rv):
                return [self.splitter(index), self.splitter(~index, flip=True)]
            elif min(self.rv) > 0:
                return [self.splitter(index)]
            else:
                return [self.splitter(~index, flip=True)]
        else:
            ts = TimeSeries(time=self.time, rv_name= self.rv_name, rv =self.rv*(-1))
            return ts

    def transform(self, transform_type):
        if transform_type == "log_diff":
            self.log_difference()
        else :
            return

    def log_difference(self):
        if self.covariate is not None:
            self.covariate = np.diff(np.log(self.covariate.astype(float)))
        log_rv = np.log(self.rv.astype(float))    
        self.rv = np.diff(np.squeeze(log_rv)) 
        self.time = self.time[1:]


    def splitter(self,index, flip = False):
        rv_name = self.rv_name
        rv_name += "_negative" if flip else "_positive"
        #Should one also flip the covariate in case of Markov setting? 
        return TimeSeries(time=self.time[index], covariate_name=self.covariate_name,  covariate=self.covariate[index], rv_name= rv_name, rv =self.rv[index]*(1-2*flip))


    def to_rv_series(self):
        returns_pd = pd.Series(
            index= self.time,
            data= self.rv)
        return returns_pd