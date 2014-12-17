import numpy as np
import pandas as pd
from pandas.io.json import json_normalize

from sklearn import preprocessing



def scale(X, eps = 0.001):
	# scale the data points s.t the columns of the feature space
	# (i.e the predictors) are within the range [0, 1]
	return (X - np.min(X, axis = 0)) / (np.max(X, axis = 0) + eps)
	
def encode_labels(df, col=""):
	df_numerical = df.copy()
	le = preprocessing.LabelEncoder()
	if col == "":
		le.fit(df)
		df_numerical = le.transform(df)
	else:
		le.fit(df[col])
		df_numerical[col] = le.transform(df[col])
	#print le.classes_
	return le, df_numerical


def convert_date_units(df):
	duration_units = {'801': 'Y', '802': 'M', '803': 'W', '804': 'D', '805': 'h', '806': 'm'}
	unit = df['drugtreatmentdurationunit']
	duration = df['drugtreatmentduration']
	if unit == '804':  df['drugtreatmentduration_units'] = np.timedelta64(duration, 'D') / np.timedelta64(1, 's') 
	elif unit == '805':  df['drugtreatmentduration_units'] = np.timedelta64(duration, 'h') / np.timedelta64(1, 's') 
	elif unit == '806':  df['drugtreatmentduration_units'] = np.timedelta64(duration, 'm') / np.timedelta64(1, 's') 
	#elif unit == '801':  df['drugtreatmentduration_units'] = np.timedelta64(duration, 'Y') / np.timedelta64(1, 's') 
	#elif unit == '802':  df['drugtreatmentduration_units'] = np.timedelta64(duration, 'M') / np.timedelta64(1, 's') 
	elif unit == '803':  df['drugtreatmentduration_units'] = np.timedelta64(duration, 'W') / np.timedelta64(1, 's') 
	return df

def normalize_date_units(df):
	df['drugenddate'] = pd.to_datetime(df['drugenddate'])
	df['drugstartdate'] = pd.to_datetime(df['drugstartdate'])
	df['drugdates_duration'] = ( (pd.to_datetime(df['drugenddate']) - \
								  pd.to_datetime(df['drugstartdate']) ) + 
								  np.timedelta64(1, 'D') ) / np.timedelta64(1, 's')
	return df

def convert_to_hiearchial_fmt(df, df_id, column_names, query=[]):
	data   = []
	data_i = []
	for records, id in zip(df, df_id): 
	    count = 0
	    for record in records:
	    	v = [id] 
	    	[v.append(record[q]) for q in query if q in record.keys()]
	    	data.append(v)
	    	data_i.append([id, count + 1])
	    	count = count + 1

	data_i =  [list(t) for t in zip(*data_i)]
	return pd.DataFrame(data, columns=column_names, index=[data_i[0], data_i[1]])


def convert_to_multicol_hiearchial_fmt(df, df_id, labels):
	data   = []
	data_i = []
	for records, id in zip(df, df_id): 
	    count = 0
	    for record in records: 
			n_record = {}
			record_dict = json_normalize(record).to_dict()
			# create a dictionary of all record data
			label_list = [label for label in labels if label in record_dict.keys()]
			n_record['safetyreportid'] = id
			for label in labels:
				if label in label_list:
					if isinstance(record_dict[label], dict):
						if(len(record_dict[label])  == 1): 
							if isinstance(record_dict[label][0], unicode): 
								if label in label_list: n_record[label] = record_dict[label][0]
							elif record_dict[label][0] is None:
								if label in label_list: n_record[label] = float('NaN')
							elif isinstance(record_dict[label][0], list):
								if label in label_list: 
									v = json_normalize(record_dict[label])
									print "List Entries:{0},{1}".format(len(v[0][0]), v[0][0])
									for i in v: n_record[label] = v[0][0][i]
						else: print "Bad Label:{0}".format(label)
				elif label != 'safetyreportid': n_record[label] = float('NaN') 

			# walk through the dictionary, parse and create new columns
			n_record_l = []
			for label in labels: n_record_l.append(n_record[label])
			data.append(n_record_l)	        
			data_i.append([id, count+1])
			count = count +1

	data_i =  [list(t) for t in zip(*data_i)]
	return pd.DataFrame(data, columns=labels, index=[data_i[0], data_i[1]])


