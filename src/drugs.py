import numpy as np
import pandas as pd
#import plots as pl
import utils

import numericalunits as nu
nu.reset_units('SI')

import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import seaborn as sns
sns.set()


class Drugs(object):
    def __init__(self):
        s_labels =  ["drugindication", "medicinalproduct", "drugcharacterization"]
        s_labels += ["drugadministrationroute", "drugauthorizationnumb", "drugbatchnumb"]
        s_labels += ["actiondrug", "drugrecurreadministration", "drugseparatedosagenumb"]
        s_labels += ["drugstartdate", "drugenddate"]
        s_labels += ["drugtreatmentduration", "drugtreatmentdurationunit"]
        s_labels += ["drugcumulativedosagenumb", "drugcumulativedosageunit"]
        s_labels += ["drugintervaldosageunitnumb", "drugintervaldosagedefinition"]
        s_labels += ["drugstructuredosagenumb", "drugstructuredosageunit"]
        s_labels += ['drugdosageform', 'drugindication', 'drugdosagetext', 'drugadditional']
        self.labels = s_labels

        # initialize data series information
        self.df = pd.DataFrame()

        # initialize conversion format functions
        self.duration_conv()
        self.dosage_conv()
        self.interval_conv()

    def transform_data(self, ds):
        # Create Hierarchial Dataframe
        for (k,v) in iter(ds):
            size = len(v)
            di = pd.DataFrame(v)
            di = di.set_index([ [k]*size, range(size) ])
            self.df = self.df.append(di)
            # TODO: Regular Expressions to include openfda fields (each field includes list of string arrays)
        self.df.index.names = ['safetyreportid','medicinalnum']

        """
        for idx, s_id in df_report['safetyreportid'].iteritems():
            d_n = self.ds[idx]["medicinalproduct"].size
            d_il = self.ds[idx]["medicinalproduct"].values
            print idx, d_n, d_il
            s_il = [s_id for i in range(d_n) ]
            missing_col = set(self.labels).difference(self.ds[idx].columns)
            col_header = list(set(self.labels).difference(missing_col))
            di = pd.DataFrame(self.ds[idx][col_header].values, index=[s_il,range(d_n)], columns=col_header )
            self.df = self.df.append(di)
            # print idx, s_id, len(d_il); print d_il
            # print df_drug_i.columns; print

        self.df.index.names = ['safetyreportid','medicinalnum']
        """

    def in_format(self):
        # normalize the units
        col_ids = {"format": "drugtreatmentdurationunit", "param": "drugtreatmentduration", "func": self.duration_conversion}
        format_cl = ["801", "802", "803", "804", "805", "806"]
        self.df = utils.format_conversion(self.df, format_cl, col_ids)


        col_ids = {"format": "drugcumulativedosageunit", "param": "drugcumulativedosagenumb", "func": self.dosage_conversion}
        format_cl = ["001", "002", "003", "004"]
        self.df = utils.format_conversion(self.df, format_cl, col_ids)

        col_ids = {"format": "drugstructuredosageunit", "param": "drugstructuredosagenumb", "func": self.dosage_conversion}
        format_cl = ["001", "002", "003", "004"]
        self.df = utils.format_conversion(self.df, format_cl, col_ids)

        col_ids = {"format": "drugintervaldosagedefinition", "param": "drugintervaldosageunitnumb", "func": self.interval_conversion}
        format_cl = ["801","802","803","804","805","806","807","810","811","812","813"]
        self.df = utils.format_conversion(self.df, format_cl, col_ids)

        #  map to categorical data
        interval_d = {"801":"days","802":"days","803":"days","804":"days","805":"days","806":"days",
                      "807":"trimester","810":"cyclical","811":"trimester","812":"as necessary","813":"total"}
        df_c = self.df[self.df["drugintervaldosagedefinition"].isin(interval_d.keys())]
        df_c.loc[:, "drugintervaldosageunitnumb"+"category.modified"] = df_c["drugintervaldosagedefinition"].map(interval_d)
        self.df["drugintervaldosageunitnumb"+"category.modified"] = np.nan
        self.df.update(df_c)


    def prime_data(self):
        # TODO: Dates - if either start or end date has NaN, output will be Nan
        # Using 'coerce', as some dates are not complete (e.g. Year,Month only)
        self.df[['drugstartdate', 'drugenddate']] = \
        self.df[['drugstartdate', 'drugenddate']].apply(lambda x: pd.to_datetime(x, coerce=True))
        self.df['drug.duration.modified'] = (self.df['drugenddate'] - self.df['drugstartdate'])/np.timedelta64(1, 'D')

        # numeric conversions
        numeric_labels =  ["actiondrug"]
        numeric_labels += ["drugcharacterization", "drugauthorizationnumb", "drugbatchnumb"]
        numeric_labels += ["drugadministrationroute", "drugrecurreadministration"]
        numeric_labels += ["drugseparatedosagenumb"]
        self.df[numeric_labels] = self.df[numeric_labels].convert_objects(convert_numeric=True)

        # Some values have 'None' associated with it rather than NaN
        # self.df["drugtreatmentduration" + ".modified"] = self.df['drugtreatmentdurationunit'].fillna(value=np.nan)
        # TODO: for patients that have death dates, can we identify what drugs that were taking or closest to taking

    def explore(self):
        gs3 = gs.GridSpec(3,3)
        fig = plt.figure(figsize=(15,12))
        # horizontal bar graph: Top 10 Drugs taken for this indication
        ax1 = fig.add_subplot(gs3[0:1,0])
        top10drugs = self.df["medicinalproduct"].value_counts(sort=True, ascending=False)[:10]
        top10drugs.plot(kind='barh', ax=ax1)
        ax1.set_title("Top 10 Drugs Taken for Condition")
        ax1.set_xlabel("# Times Taken")
        # horizontal bar graph: Reported Role of the drug
        # TBD: correlation with a corresponding drug
        ax2 = fig.add_subplot(gs3[0:1,1])
        labels = {"1": "Suspect", "2": "Concomitant", "3": "Interacting"}
        characterization = self.df["drugcharacterization"].value_counts(sort=True, ascending=True)
        characterization.index = characterization.index.map(lambda x: labels[str(x)])
        characterization.plot(kind='barh', ax=ax2)
        ax2.set_title("Characterization of Drug")
        ax2.set_xlabel("# Occurences")
        # horizontal bar graph: Actions take by the drug
        # TBD: correlation with a corresponding drug
        ax3 = fig.add_subplot(gs3[0:1,2])
        labels = {"1": "Drug Withdrawn", "2": "Dose Reduced", "3": "Dose Increased", "4": "Dose Not Changed", "5": "Unknown", "6": "N/A"}
        actions = self.df["actiondrug"]
        actions = actions[actions.notnull()].astype('int').value_counts(sort=True, ascending=True)
        actions.index = actions.index.map(lambda x: labels[str(x)])
        actions.plot(kind='barh', ax=ax3)
        ax3.set_title("Actions taken with Drug")
        ax3.set_xlabel("# Occurences")
        # histogram: Drug Treatment Duration (Based off of drugtreatmentduration field)
        ax4 = fig.add_subplot(gs3[1:2,0])
        drugtreatmentduration= self.df["drugtreatmentduration" + ".modified"].value_counts(sort=True, ascending=True)
        sns.distplot(drugtreatmentduration.dropna(), ax=ax4)
        ax4.set_title("Patient Drug Maximum Treatment Duration")
        ax4.set_xlabel("Days")
        # histogram: Patient Drug Maximum Duration (Should be based off Start Date/End Date when available)
        ax5 = fig.add_subplot(gs3[1:2,1])
        drugduration = self.df["drug.duration" + ".modified"].value_counts(sort=True, ascending=True)
        sns.distplot(drugduration.dropna(), ax=ax5)
        ax5.set_title("Patient Drug Maximum Duration")
        ax5.set_xlabel("Days")
        plt.tight_layout()

    def imputate(self):
        "print imputate"

    def duration_conv(self):
        self.duration_conversion = {
            "801": lambda x: (x * nu.year)   / nu.day,   # year      -> days
            "802": lambda x: (x * nu.m)      / nu.day,   # month     -> days
            "803": lambda x: (x * nu.week)   / nu.day,   # week      -> days
            "804": lambda x: (x * nu.day)    / nu.day,   # day       -> days
            "805": lambda x: (x * nu.hour)   / nu.day,   # hour      -> days
            "806": lambda x: (x * nu.minute) / nu.day    # minute    -> days
        }

    def dosage_conv(self):
        self.dosage_conversion = {
            "001": lambda x: (x * nu.kG) /nu.mG,        # kg        -> mg
            "002": lambda x: (x * nu.G)  /nu.mG,        # g         -> mg
            "003": lambda x: (x * nu.mG) /nu.mG,        # mg        -> mg
            "004": lambda x: (x * nu.uG) /nu.mG         # ug        -> ug
        }

    def interval_conv(self):
        self.interval_conversion = {
            "801": lambda x: (x * nu.year) / nu.day,    # year      -> days
            "802": lambda x: (x * nu.m) / nu.day,       # month     -> days
            "803": lambda x: (x * nu.week) / nu.day,    # week      -> days
            "804": lambda x: (x * nu.day)  / nu.day,    # days      -> days
            "805": lambda x: (x * nu.hour) / nu.day,    # hour      -> days
            "806": lambda x: (x * nu.minute) / nu.day   # minute    -> days
        }

    def diagnostics(self):
        print self.df.keys()
        print self.df.index.levels
        print self.df.index.names
        print self.df.info()
        self.df.head()





# drug characterization
# df_role = pd.get_dummies(df_drugs_i['drugcharacterization'])
# df_role = df_role.rename(columns = {'1':'drug.suspect', '2': 'drug.concomitant', '3': 'drug.interacting'})
# df_drugs_i = df_drugs_i.join(df_role)
# df_drugs_i = df_drugs_i.drop('drugcharacterization', axis=1)
# actiondrug -> (values 1-6): perform as factorization; very few existent
# drugadministrationroute -> (values: 001-067): factorization with so many values? categorical?

# openfda fields (not currently included): FOR LATER
# Pharamacological Claas (EPC): pharm_class_epc -> what the drug is supposed to fix
# Mecanism of Action (MOA): pharm_class_moa -> how the drug works
# Physiologic Effect (PE) - pharm_class_pe -> what the drug affects
# Chemical Structure (CS) - pharm_class_cs -> what is in the drug

# NLP Types: FOR LATER
# patient.drug.drugdosageform (e.g. tablet)
# patient.drug.drugdosagetext
# patient.drug.drugindication (drug indicated for)
# patient.drug.drugadditional (circumstances for taking the drug - not commonly used)


