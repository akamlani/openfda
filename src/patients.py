from filters import *
import numpy as np
import utils
#import plots as pl

import numericalunits as nu
nu.reset_units('SI')


import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import seaborn as sns
sns.set()

class Patients(object):
    def __init__(self, df):
        v = report_id + serious_id + serious_id_type + patient_id
        self.df = df.ix[:, v]
        self.age_conv()

    def in_format(self):
        col_ids = {"format": "patient.patientonsetageunit", "param": "patient.patientonsetage", "func": self.age_conversion}
        format_cl = ["800", "801", "802", "803", "804", "805"]
        self.df = utils.format_conversion(self.df, format_cl, col_ids)

    def prime_data(self):
        """primate the data for the model"""
        # change select types to float
        v = serious_id + serious_id_type
        self.df.loc[:, v] = self.df[v].convert_objects(convert_numeric=True)
        self.df.loc[:, "patient.patientweight"] = self.df["patient.patientweight"].convert_objects(convert_numeric=True)

    def imputate(self):
        # treat sex as binary
        sex_col = pd.get_dummies(df["patient.patientsex"])
        sex_col = sex_col.rename(columns = {'1':'patient.male', '2': 'patient.female'})
        self.df = self.df.join(sex_col)
        self.df = self.df.drop(["patient.patientsex"], axis=1)
        # weight averaging
        self.df["patient.patientweight"].fillna(self.df["patient.patientweight"].mean(), inplace=True)
        # serious types
        self.df[serious_id + serious_id_type] = self.df[serious_id + serious_id_type].fillna(0)

    def explore(self):
        # Histogram: Patient Age Distribution
        gs2 = gs.GridSpec(2,2)
        fig = plt.figure(figsize=(15,8))
        ax1 = fig.add_subplot(gs2[0:1,0])
        sns.distplot(self.df['patient.patientonsetage.modified'].dropna(), bins=15, ax=ax1)
        ax1.set_title("Patient Age Distribution")
        ax1.set_xlabel("Age")
        # Histogram: Patient Weight Distribution
        ax2 = fig.add_subplot(gs2[0:1,1])
        sns.distplot(self.df['patient.patientweight'].dropna(), ax=ax2)
        ax2.set_title("Patient Weight Distribution")
        ax2.set_xlabel("Weight (kg)")
        # Horizontal Bar Plot: Male/Female Distribution
        ax3 = fig.add_subplot(gs2[1:2,0])
        labels = {"1": "male", "2": "female"}
        sex = self.df["patient.patientsex"].value_counts(sort=True, ascending=True)
        sex.index = sex.index.map(lambda x: labels[str(x)])
        sex.plot(kind='barh', ax=ax3)
        ax3.set_title("Gender Count Distribution")
        ax3.set_xlabel("# Reports Reported")
        # Horizontal Bar Plot: Seriousiness Conditions
        ax4 = fig.add_subplot(gs2[1:2,1])
        seriouscnt = self.df[serious_id + serious_id_type].describe().T["count"].copy()
        seriouscnt.sort(ascending=True)
        seriouscnt.plot(kind='barh', ax=ax4)
        ax4.set_title("Distribution of Serious Type Conditions")
        ax4.set_xlabel("# Events Reported")
        plt.tight_layout()
    def age_conv(self):
        self.age_conversion = {
            "800": lambda x: ((x*10) * nu.year) /nu.year,   # decade -> years
            "801": lambda x: (x * nu.year) / nu.year,       # year   -> years
            "802": lambda x: (x * nu.m) / nu.year,          # month  -> years
            "803": lambda x: (x * nu.week) / nu.year,       # week   -> years
            "804": lambda x: (x * nu.day) / nu.year,        # days   -> years
            "805": lambda x: (x * nu.hour) / nu.year        # hour   -> years
        }

    def diagnostics(self):
        print self.df["patient.patientonsetage"].describe()
        print self.df["patient.patientonsetage"].value_counts().head()

        print self.df["patient.patientweight"].describe()
        print self.df["patient.patientweight"].value_counts().head()
        print "mean: \n", self.df["patient.patientweight"].mean()

        self.df.describe(); print



# bin the results for patient weight and age
# decades = list(range(0,120,10))
# self.df["age_range"] = pd.cut(self.df["patient.patientonsetage"], decades)

#col = self.df["patient.patientweight"].notnull()
#df.loc[df_patients[col == True].index, "patient.patientweight"] = df[col]["patient.patientweight"].astype('float')
#df = df.loc[df_patients[col == True].index, "patient.patientweight"].values

#print self.df[["patient.male", "patient.female"]].describe()
#print "Male Count: ", len(df[df["patient.male"] == 1])
#print "Female Count: ", len(df[df["patient.female"] == 1])
