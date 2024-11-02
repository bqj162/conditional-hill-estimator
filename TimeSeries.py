import numpy as np
class TimeSeries:
    def __init__(self,time,  covariate_name, covariate, rv_name, rv):
        self.time = time
        self.covariate = covariate
        self.covariate_name = covariate_name
        self.rv = rv
        self.rv_name = rv_name


    def split(self):
        index = np.array(self.rv) > 0
        if max(self.rv) > 0 > min(self.rv):
            return [self.splitter(index), self.splitter(~index, flip=True)]
        elif min(self.rv) > 0:
            return [self.splitter(index)]
        else:
            return [self.splitter(~index, flip=True)]

    def transform(self, transform_type):
        if transform_type == "log_diff":
            self.log_difference()
        else :
            return

    def log_difference(self):
        self.covariate = np.diff(np.log(self.covariate))
        self.rv = np.diff(np.log(self.rv))
        self.time = self.time[1:]


    def splitter(self,index, flip = False):
        rv_name = self.rv_name
        rv_name += "_negative" if flip else "_positive"
        return TimeSeries(time=self.time[index], covariate_name=self.covariate_name,  covariate=self.covariate[index], rv_name= rv_name, rv =self.rv[index]*(1-2*flip))

