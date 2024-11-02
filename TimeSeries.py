import numpy as np
class TimeSeries:
    def __init__(self, time, covariate, rv):
        self.time = time
        self.covariate = covariate
        self.rv = rv

    def split(self):
        index = np.array(self.rv) > 0
        if max(self.rv) > 0 > min(self.rv):
            return [self.splitter(index), self.splitter(~index)]
        elif min(self.rv) > 0:
            return [self.splitter(index)]
        else:
            return [self.splitter(~index)]

    def transform(self, transform_type):
        if transform_type == "log_diff":
            self.log_difference()
        else :
            return

    def log_difference(self):
        self.covariate = np.diff(np.log(self.covariate))
        self.rv = np.diff(np.log(self.rv))
        self.time = self.time[1:]


    def splitter(self,index):
       return TimeSeries(time=self.time[index],covariate=self.covariate[index], rv =self.rv[index])

