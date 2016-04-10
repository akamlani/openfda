import urllib2 as ulib
import pandas as pd
import numpy as np

from pandas.io.json import json_normalize
import json

from filters import *
import reports as rp


def read_credentials():
    """read in the configuration API Key"""
    # openfda is rate limited without api key
    # No API Key: 40 requests per minute (per IP Address): 1000 requests per day (per IP Address)
    # API Key:    240 requests per minute (per key):       120000 requests per day (per key)
    credentials = yaml.load(open('./../config/api_cred.yml'))
    api_key = credentials['openfda_key']
    return api_key

def load_from_file():
    "load cached openfda request data from a file"
    results=[]
    with open("./../data/openfda_data_500.json", 'r') as f:
        results = json.load(f)
    # this returns a string, not dictionary
    # for logging purposes usage: results = json.dumps(results, indent=4)
    print "file version", type(results)
    return results


def openfda_request(query, count, api_key, limit=100, skip=0):
    """perform an OpenFDA Restful API Request"""
    request_string ='https://api.fda.gov/drug/event.json?api_key={0}&search={1}&count={2}&limit={3}&skip={4}'
    request = request_string.format(api_key, query, count, limit, skip)
    print "Send Query as:\n{0}".format(request)
    response=ulib.urlopen(request)
    fda_data=json.load(response)
    records_received = len(fda_data["results"])
    print "Result Count: {0}".format(records_received)
    return fda_data, records_received


def load_from_api(search, count=""):
    """request the data from the OpenFDA API, requests are limited in 100 records at a time"""
    results = []
    max_records = 15000
    record_count = 100; offset=0
    api_key = read_credentials()
    # limited by OpenFDA, maximum records at a time is 100 via search filters, use pagination
    for i in range(max_records/record_count):
        records, records_received = openfda_request(search, count, api_key, record_count, offset)
        offset += records_received
        results += records['results']
        print "Records Retrieved Count: {0}".format(len(results))

    with open("./../data/openfda_data.json", 'w') as f:
        print "file api: results type", type(results)
        json.dump(results, f, indent=4, ensure_ascii=False)

    return results


def get_search_query():
    """define the default query"""
    search = 'patient.patientonsetage:[65+TO+99]'
    search+= '+AND+_exists_:serious+AND+serious:1'
    search+= '+AND+_exists_:patient.drug.drugcharacterization'
    search+= '+AND+_exists_:patient.drug.drugindication+AND+patient.drug.drugindication:hypertension'
    return search

def find_data(df):
    """parse the labels in terms of default columns"""
    d = {}
    cols = [report_id, duplicate_id, serious_id, serious_id_type, patient_id]
    d = dict([ (label, df[label]) for f_id in cols for label in df.columns if label in f_id])
    return pd.DataFrame(d)

def parse_response(json_response):
    """parse the json dictionaries response data"""
    obj = []
    df = json_normalize(json_response)
    df_patients = find_data(df)
    # place in form of tuples to track safetyreportid
    df_v_reactions = df.apply(lambda x:(x.safetyreportid, x["patient.reaction"]),axis=1)
    df_v_drugs = df.apply(lambda x:(x.safetyreportid, x["patient.drug"]),axis=1)
    return df, df_patients, df_v_drugs, df_v_reactions


