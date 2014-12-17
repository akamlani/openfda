import urllib2 as ulib
import json

import numpy as np
import pandas as pd
from pandas.io.json import json_normalize

import yaml
from pprint import pprint

import filters as filters
import utilities as utl

from sklearn import preprocessing



def read_credentials():
	"""read in the configuration API Key"""
	# openfda is rate limited without api key
	# No API Key: 40 requests per minute (per IP Address): 1000 requests per day (per IP Address)
	# API Key:    240 requests per minute (per key):       120000 requests per day (per key)
	credentials = yaml.load(open('../config/api_cred.yml'))
	api_key = credentials['openfda_key']
	return api_key


def openfda_request(query, count, api_key, limit=100, skip=0):
	"""openfda Restful API Request"""
	request_string ='https://api.fda.gov/drug/event.json?api_key={0}&search={1}&count={2}&limit={3}&skip={4}'
	request = request_string.format(api_key, query, count, limit, skip)
	print "Send Query as:\n{0}".format(request)
	response=ulib.urlopen(request)
	fda_data=json.load(response)
	records_received = len(fda_data["results"]) 
	print "Result Count: {0}".format(records_received)
	return fda_data, records_received 


def load_from_file():
	results=[]
	with open("./../data/openfda_data.json", 'r') as f: results = json.load(f)
	#this returns a string, not dictionary, use for print only: results = json.dumps(results, indent=4)
	print "file version", type(results)
	return results


def load_from_api():
	results = []
	max_records = 1000
	record_count = 100; offset=0
	api_key = read_credentials()
	count, search = get_search_query()	
	# limited by OpenFDA, maximum records at a time is 100 via search filters, use pagination
	for i in range(max_records/record_count):
		records, records_received = openfda_request(search, count, api_key, record_count, offset)
		offset += records_received
		results += records['results']
		print "Total Count: {0}".format(len(results))

	with open("./../data/openfda_data.json", 'w') as f: json.dump(results, f, indent=4, ensure_ascii=False)
	print "api version", type(results)
	return results

def get_search_query():
	count = ''
	search = 'patient.patientonsetage:[65+TO+99]'
	search+= '+AND+_exists_:serious+AND+serious:1'
	search+= '+AND+_exists_:patient.drug.drugcharacterization'
	search+= '+AND+_exists_:patient.drug.drugindication+AND+patient.drug.drugindication:hypertension'

	"""
	search+= '+AND+_exists_:occurcountry'
	search+= '+AND+_exists_:patient.drug.actiondrug'
	search+= '+AND+_exists_:patient.drug.drugstartdate'
	search+= '+AND+_exists_:patient.drug.drugenddate'
	search+= '+AND+_exists_:patient.drug.drugtreatmentduration'
	search+= '+AND+_exists_:patient.drug.drugbatchnumb'
	search+= '+AND+_exists_:patient.drug.drugadministrationroute'
	search+= '+AND+_exists_:patient.drug.drugrecurreadministration'
	search+= '+AND+_exists_:patient.drug.drugdosageform'
	search+= '+AND+_exists_:patient.drug.drugcumulativedosagenumb+AND+_exists_:patient.drug.drugcumulativedosageunit'
	search+= '+AND+_exists_:patient.drug.drugintervaldosagedefinition+AND+_exists_:patient.drug.drugintervaldosageunitnumb'
	search+= '+AND+_exists_:patient.drug.drugseparatedosagenumb+AND+_exists_:patient.drug.drugstructuredosagenumb'
	search+= '+AND+_exists_:patient.drug.drugstructuredosageunit'

	search+= '+AND+_exists_:patient.reaction.reactionoutcome+AND+patient.reaction.reactionoutcome:3'
	search+= '+AND+_exists_:patient.reaction.reactionoutcome'

	search+= '+AND+_exists_:patient.patientonsetage+AND+_exists_:patient.patientsex+AND+_exists_:patient.patientweight'
	"""
	return count, search




def extract_embedded_extensions_df(df):
	"""extract condition embedded list of extensions"""
	drug_list = []
	for drug in df:
		d = {};      
		for label in drug.keys(): 
			for f_id in [filters.drug_id, filters.drug_span_id, 
						 filters.drug_dosage_id, filters.drug_openfda]:
				if label in f_id:	d[label] = drug[label]
		drug_list.append(d)
	return drug_list


def extract_extensions_df(df, features, d, extension):
	"""extract condtional extensions"""
	for label in extension: 
	    if label in features: d[label] = df[label]
	return d    


def extract_df(df, features, obj):
	"""deal with finding embedded dictionary information"""
	# determine if these labels exit in the row and pull out neceaary informationoo
	# no embedded lists are pulled out at this point
	d = {}		
	for label in features: 
		for f_id in [filters.report_id, filters.incident_id, filters.patient_id, filters.reactions_id]:
			if label in f_id:  d[label] = df[label]
	# some values are dependent upon conditional values: futher filtering
	cond = ('serious' in features) and (int(df['serious']) == 1) 
	if(cond): extract_extensions_df(df, features, d, filters.incident_id_ext)
	else: 
		for i in filters.incident_id_ext: d[i] = float('NaN')  
	# serious -> serious deaths -> patient deathdate
	cond = (('seriousnessdeath' in features) and df['seriousnessdeath'] == str(1))
	if (cond): extract_extensions_df(df, features, d, filters.patient_id_ext)
	else:  
		for i in filters.patient_id_ext: d[i] = float('NaN')
	obj.append(d)
   

def parse_response(json_response):
	"""parse the json dictionaries response data"""
	obj = []
	df = json_normalize(json_response)

	df.apply(lambda x: extract_df(x, df.columns, obj), axis=1)	
	df_report = pd.DataFrame(obj, columns=obj[0].keys())
	df_reactions = df_report['patient.reaction']
	patient_drugs = df['patient.drug'].apply(extract_embedded_extensions_df)
	df_patient_drugs = pd.DataFrame(patient_drugs)
	return df_report, df_reactions, df_patient_drugs


def build_relational_tables(df_report, df_reactions, df_patient_drugs):
	"several tables each sharing unique identifier of safetyreportid"
	# (1) report table (report_id, incidents, incidents extensions)
	missing_col = list(set(filters.report_id).difference(df_report.columns))
	col_header = list(set(filters.report_id).difference(missing_col))
	reporting_table = df_report[col_header].reindex_axis(filters.report_id, axis=1)
	missing_col = list(set(filters.incident_id).difference(df_report.columns))
	col_header = list(set(filters.incident_id).difference(missing_col))
	reporting_table[col_header] = df_report[col_header]
	missing_col = list(set(filters.incident_id_ext).difference(df_report.columns))
	col_header = list(set(filters.incident_id_ext).difference(missing_col))
	reporting_table[col_header] = df_report[col_header]
	reporting_table.set_index('safetyreportid')
	# (2) patient table
	patient_table = df_report[['safetyreportid'] + filters.patient_id + filters.patient_id_ext]
	patient_table.set_index('safetyreportid')
	# (3) patient reactions table
	reaction_table = utl.convert_to_hiearchial_fmt(df_reactions, patient_table['safetyreportid'], 
												   ['safetyreportid', 'reactionmeddrapt'],  
												   ['reactionmeddrapt'])#'reactionoutcome', 'reactionoutcome'
	# (4) drug prescriptions table
	prescriptions_table = utl.convert_to_multicol_hiearchial_fmt(df_patient_drugs['patient.drug'], 
																 patient_table['safetyreportid'], 
																 ['safetyreportid'] + 
																 filters.drug_id + filters.drug_span_id + 
																 filters.drug_dosage_id) # +
																 #filters.open_drug_harmon_id +
																 #filters.open_pharm_drug_class_id +
																 #filters.open_ingredient_id +
																 #filters.open_ingredient_labels_id)

	return patient_table, reaction_table, reporting_table, prescriptions_table



def imputate_relational_tables(patient_table, reaction_table, reporting_table, prescriptions_table):
	"""imputate data in relational tables"""
	# patient information	
	patient_table_ = patient_table.copy()
	patient_table_['patient.patientdeath.patientdeathdate'] = pd.to_datetime(patient_table_['patient.patientdeath.patientdeathdate']) 
	numeric_features = ['patient.patientsex', 'patient.patientonsetage', 'patient.patientweight']
	for n in numeric_features: patient_table_[n] = patient_table_[n].convert_objects(convert_numeric=True,convert_dates=True)
 	for col in ['patient.patientsex', 'patient.patientonsetage', 'patient.patientweight']:
		avg = round(patient_table_[col][patient_table_[col].notnull()].mean())
		patient_table_[col].fillna(avg, inplace=True)

	# reporting table
	reporting_table['receivedate'] = pd.to_datetime(reporting_table['receivedate'])
	reporting_table['receiptdate'] = pd.to_datetime(reporting_table['receiptdate'])
	reporting_table['reportage'] = (reporting_table['receiptdate'] - reporting_table['receivedate'])/ np.timedelta64(1, 's') 
	reporting_table['serious'] = reporting_table['serious'].apply(lambda x: float(x))
	#le_rep, reporting_table = utl.encode_labels(reporting_table, 'occurcountry')	
	report_filters = ['primarysource.qualification'] + filters.incident_id + filters.incident_id_ext
	for col in report_filters: 
		if col in reporting_table.columns: reporting_table[col] = reporting_table[col].convert_objects(convert_numeric=True)
	reporting_table['primarysource.qualification'] = reporting_table['primarysource.qualification'].fillna(0)	
	for label in filters.incident_id_ext:
		if label in reporting_table.columns: reporting_table[label] = reporting_table[label].fillna(0)
	reporting_table.drop(['companynumb', 'duplicate'], axis=1, inplace=True)
	reporting_table.drop(['receivedate', 'receiptdate'], axis=1, inplace=True)

	# prescriptions table
	col = ['safetyreportid', 'medicinalproduct', 'drugindication', 'drugcharacterization', 'actiondrug']
	drug_table = prescriptions_table.copy()
	drug_table = drug_table[col]
	drug_count = len(drug_table['medicinalproduct'].unique())
	drug_table['drugindication'] = drug_table['drugindication'].fillna('')
	le_di, drug_table = utl.encode_labels(drug_table, 'drugindication')	
	le_med, drug_table = utl.encode_labels(drug_table, 'medicinalproduct')	
	drug_table['actiondrug'] = drug_table['actiondrug'].fillna(0)
	drug_table['actiondrug'] = drug_table['actiondrug'].convert_objects(convert_numeric=True)
	drug_table['drugcharacterization'] = drug_table['drugcharacterization'].convert_objects(convert_numeric=True)

	# reaction_table
	# reaction_outcome_labels = np.sort(reaction_table['reactionoutcome'].unique())
	# reaction_table['reactionoutcome'] = reaction_table['reactionoutcome'].convert_objects(convert_numeric=True)	
	le_react, reaction_table = utl.encode_labels(reaction_table, 'reactionmeddrapt')

	return reporting_table, patient_table_, drug_table, reaction_table, [le_di, le_med, le_react]


def prepare_data(samples):	
	# Parse Information
	df_report, df_reactions, df_patient_drugs = parse_response(samples) 
	# Build Relational Tables
	patient_table, reaction_table, reporting_table, prescriptions_table = \
	build_relational_tables(df_report, df_reactions, df_patient_drugs)
	# Imputate data (not used much currently)
	reporting_table, patient_table_, drug_table, reaction_table, le_l = \
	imputate_relational_tables(patient_table, reaction_table, reporting_table, prescriptions_table)
	# Reverse per Label Datapoints and unique labels
	reaction_labels =  le_l[2].inverse_transform(reaction_table['reactionmeddrapt'])
	drug_labels = le_l[1].inverse_transform(drug_table['medicinalproduct'])		
	# Create Patient/Drug Matrix
	names_l, drugs_l, dg_v = drug_binary_vec(drug_table)
	name_drug_df = pd.DataFrame(dg_v, index=names_l, columns=drugs_l)
	print "parse binary vector shape: {0}".format(name_drug_df.shape)
	return drug_table, reaction_table, name_drug_df, le_l


def group_drugs(drug_table):
	"""group drugs into a list to prepare for ML algorithms"""
	v = []
	drug_table_v = drug_table.copy()
	#drug_table_v = drug_table_v.reset_index()
	#drug_table_v.drop(['level_0', 'level_1'], axis=1, inplace=True)
	drug_table_v = drug_table_v.groupby('medicinalproduct', sort=True, as_index=False)
	for name, group in drug_table_v: v.append(group.to_dict())
	
	d_v = []
	for de in v: 
		d = {}; length = len(de['medicinalproduct']);
		for key,value in de.iteritems():
			d_l = []
			for i in range(length):
				for k, v in value.iteritems(): d[key] =  v
				d_l.append(d)
		d_v.append(d_l)
	
	return d_v
	

def drug_binary_vec(drug_table):
	"""group report id drugs and binarize the data"""
	dg_v = []; names_l = []; drugs_l = []
	drug_p = pd.get_dummies(drug_table['medicinalproduct']).reset_index()
	drug_p.drop('level_1', axis=1, inplace=True)
	dg =  drug_p.groupby('level_0')
	for name, group in dg:
		col = group.columns - ['level_0']
		df = group[col]; t = [0] * len(col)
		for index, row in df.iterrows(): t += row[col]
		dg_v.append(t)
		names_l.append(name)
		drugs_l = col

	dg_v = np.array(dg_v)
	dg_v = preprocessing.binarize(dg_v)
	return names_l, drugs_l, dg_v

def drug_reaction_binary_vec(drug_reaction):
	dr_v = []; drugs_l = []; react_l = [] 
	drug_reaction = pd.get_dummies(drug_reaction['reactionmeddrapt']).reset_index()
	drug_reaction = drug_reaction.groupby('medicinalproduct')
	for name, group in drug_reaction: 
		col = group.columns - ['safetyreportid', 'medicinalproduct']
		df = group[col]; t = [0] * len(col)
		for index, row in df.iterrows(): t += row[col]
		dr_v.append(t)
		drugs_l.append(name)
		react_l = col

	dr_v = np.array(dr_v)
	dr_v = preprocessing.binarize(dr_v)
	return drugs_l, react_l, dr_v
