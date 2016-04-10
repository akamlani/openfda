import numpy as np
import pandas as pd
#import plots as pl

import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import seaborn as sns
sns.set()


class Reporter(object):
    def __init__(self, df):
        # Fixed: {'sender.sendertype = 2, sender.senderorganization = 'FDA-Public Use'}
        # Fixed: {'receiver.receiverype = 6, receiver.receiverorganization = 'FDA'}
        r_labels =  ['safetyreportid']
        r_labels += ['transmissiondate', 'receiptdate', 'receivedate']
        r_labels += ['duplicate', "reportduplicate.duplicatesource", "reportduplicate.duplicatenumb"]
        r_labels += ['occurcountry']
        r_labels += ['primarysourcecountry', 'primarysource.qualification', 'primarysource.reportercountry', 'companynumb']
        self.df = df[r_labels].copy()

    def in_format(self):
        # normalize the delta date units to days
        self.df[['transmissiondate', 'receiptdate', 'receivedate']] = \
        self.df[['transmissiondate', 'receiptdate', 'receivedate']].apply(lambda x: pd.to_datetime(x))
        self.df["receipt.duration"] = abs(self.df['receiptdate'] - self.df['transmissiondate']) / np.timedelta64(1, 'D')
        self.df["receive.duration"] = abs(self.df['receivedate'] - self.df['transmissiondate']) / np.timedelta64(1, 'D')
        # set the correct report duration (always pick the larger period in days)
        self.df["report.duration"] = self.df.apply(lambda x: x["receipt.duration"]
                                                   if x["receipt.duration"] > x["receive.duration"]
                                                   else x["receive.duration"], axis=1)

    def prime_data(self):
        self.df["duplicate"] = self.df["duplicate"].convert_objects(convert_numeric=True)
        self.df["primarysource.qualification"] = self.df["primarysource.qualification"].convert_objects(convert_numeric=True)


    def explore(self):
        gs1 = gs.GridSpec(2,2)
        fig = plt.figure(figsize=(15,6))
        # histogram: report duration
        ax1 = fig.add_subplot(gs1[0:1,0])
        sns.distplot(self.df['report.duration'], bins=15, ax=ax1)
        ax1.set_title("patient report maximum delay reception")
        ax1.set_xlabel("Days")
        # bar horizontal: duplicates
        ax2 = fig.add_subplot(gs1[0:1,1])
        duplicates = self.df.duplicate.isnull().value_counts()
        duplicates.index=["UnReported", "Duplicates"]
        duplicates.plot(kind='barh', ax=ax2)
        ax2.set_title("Number of Duplicates Reported")
        # bar horizontal: country occurence
        ax3 = fig.add_subplot(gs1[1:2,0])
        countries = self.df["occurcountry"].value_counts(sort=True, ascending=True)
        countries.plot(kind='barh', ax=ax3)
        ax3.set_title("Countries where reported event occured")
        # bar horizontal: reporting types
        ax4 = fig.add_subplot(gs1[1:2,1])
        reportings = self.df["primarysource.qualification"]
        reportings = reportings[reportings.notnull()].astype('int').value_counts(sort=True, ascending=True)
        labels = {"1": "Physician", "2": "Pharamacist", "3": "Professional", "4": "Lawyer", "5": "Consumer"}
        reportings.index = reportings.index.map(lambda x: labels[str(x)])
        reportings.plot(kind='barh', ax=ax4)
        ax4.set_title("Distribution of Reporting Types")
        plt.tight_layout()

    def diagnostics(self):
        print "# of Duplicate Reports: ", len(self.df[self.df["duplicate"].notnull()])
        print self.df.info()
        self.df.head()




