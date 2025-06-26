import pandas as pd

class ResidualSeries:
    def __init__(self, time, values, name):
        self.time = time
        self.values = values
        self.name = name
        self.series = None

    def to_series(self):
        returns_pd = pd.Series(
            index= self.time,
            data= self.values
        )
        self.series = returns_pd
        return self.series
