#primarysource.qualification {1-5}
report_id = ["safetyreportid", "receivedate", "receiptdate", "duplicate", "companynumb", "primarysource.qualification"]

#"serious" {1-2} = if 1 -> incident_ext
incident_id = ["occurcountry", "serious"]
incident_id_ext = ["seriousnesscongenitalanomali", "seriousnessdeath", "seriousnessdisabling", 
				   "seriousnesshospitalization", "seriousnesslifethreatening", "seriousnessother"] 

#if incident_ext["seriousnessdeath" == str(1)] --> patient_id_ext
patient_id = ['patient.patientsex', 'patient.patientonsetage', 'patient.patientweight']
patient_id_ext = ["patient.patientdeath.patientdeathdate"]

# list of reactions of type ('patient.reaction.reactionmeddrapt'); few of types 'reactionoutcome', as voluntary report
reactions_id = ['patient.reaction']
reactions_id_ext = ['reactionoutcome']  

# 'drugcharacterization': {1-3} = report the role of drug in adverse event
# 'actiondrug' : {1-6} = actions taken with drug
drug_id = ['medicinalproduct', 'drugindication', 'drugcharacterization', 'actiondrug', 'drugadditional']
drug_span_id = ['drugstartdate', 'drugenddate', 'drugtreatmentduration', 'drugtreatmentdurationunit']
drug_dosage_id = ['drugcumulativedosagenumb', 'drugcumulativedosageunit', 'drugdoseagetext',
                  'drugdosageform', 'drugintervaldosagedefinition', 'drugintervaldosageunitnumb',
                  'drugseparatedosagenumb', 'drugstructuredosagenumb', 'drugstructuredosageunit',  
                  'drugadministrationroute', 'drugrecurreadministration', 'drugbatchnumb' ]

# openfda harmonization
# Drug Harmonization - Roughly 86% of adverse event records have at least one openfda section
# Harmonization as different datasets use different drug identifiers 
# Harmonization process requires an exact match, so some drug products cannot be harmonized
drug_openfda = ['openfda']
# RxNorm is a normalized naming system for generic and branded drugs
# UNII stands for Unique Ingredient Identifier
open_ingredient_id = ['openfda.rxcui', 'openfda.unii', 'openfda.nui'] 
#ingredient_labels_id = ['pharm_class_cs', 'pharm_class_moa']
open_ingredient_labels_id = ['openfda.pharm_class_cs', 'openfda.pharm_class_moa']

# Structured Productuct Labeling
#pharm_drug_class_id = ['pharm_class_epc', 'pharm_class_pe']
open_pharm_drug_class_id = ['openfda.pharm_class_epc', 'openfda.pharm_class_pe']

# NDC (National Drug Code)
# Drug products identified and reported using a unique, 3 segment number
#drug_harmon_id = ['brand_name', 'generic_name', 'substance_name', 'manufacturer_name', 'product_type'] 
open_drug_harmon_id = ['openfda.brand_name', 'openfda.generic_name', 'openfda.substance_name', 
					   'openfda.manufacturer_name', 'openfda.product_type'] 

#'product_ndc'
#drug_rec_id = ['route']


