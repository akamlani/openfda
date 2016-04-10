
import pandas as pd
#import plots as pl
import utils


import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import seaborn as sns
sns.set()


class Reactions(object):
    def __init__(self):
        self.df = pd.DataFrame()
    def in_format(self):
        labels = {"1": "Recovered/resolved", "2": "Recovering/resolving", "3": "Not recovered/Not resolved",
                  "4": "Recovered/resolved with sequelae", "5": "Fatal", "6": "unknown"}
        outcome = self.df["reactionoutcome"]
        outcome = outcome[outcome.notnull()].astype('int')
        self.df_outcome = outcome.map(lambda x: labels[str(x)])

    def transform_data(self, ds):
        for (k,v) in iter(ds):
            size = len(v)
            di = pd.DataFrame(v)
            di = di.set_index([[k]*size, range(size)])
            self.df = self.df.append(di)
        self.df.index.names = ['safetyreportid','reactionnum']
        # create single reaction matrix  per patient id (safetyreportid)
        reaction_occurences = self.df.reactionmeddrapt
        self.df_rm = pd.get_dummies(reaction_occurences)
        self.df_rm = self.df_rm.groupby(level=0).sum()
    def explore(self):
        gs4 = gs.GridSpec(1,2)
        fig = plt.figure(figsize=(12,5))
        # horizontal bar plot: Top 10 overall reactions occuring for indication
        ax1 = fig.add_subplot(gs4[0:1,0])
        topreactions = self.df_rm.sum()
        topreactions.sort(ascending=False)
        topreactions = topreactions[:10]
        topreactions.sort(ascending=True)
        topreactions.plot(kind='barh', ax=ax1)
        ax1.set_title("Top 10 Reactions Occured")
        ax1.set_xlabel("# Reactions Reported")
        # horizontal bar plot: Reaction Outcomes Reported
        ax2 = fig.add_subplot(gs4[0:1,1])
        outcome = self.df_outcome.value_counts(sort=True, ascending=True)
        outcome.plot(kind='barh', ax=ax2)
        ax2.set_title("Top Reaction Outcomes")
        ax2.set_xlabel("# Reaction Outcomes Reported")
        plt.tight_layout()



