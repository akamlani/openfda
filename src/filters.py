report_id = ["safetyreportid"]
report_id_info = ["receivedate", "receiptdate", 
                  "receiver.receivertype", "receiver.receiverorganization", 
                  "companynumb", "occurcountry", "transmissiondate", 
                  "sender.sendertype", "sender.senderorganization",           
                  "primarysourcecountry","primarysource.qualification", "primarysource.reportercountry"]

duplicate_id = ["duplicate"]
duplicate_id_info = ["reportduplicate.duplicatesource", "reportduplicate.duplicatenumb"]

serious_id = ["serious"]
serious_id_type = ["seriousnesscongenitalanomali", "seriousnessdeath", "seriousnessdisabling", 
                   "seriousnesshospitalization", "seriousnesslifethreatening", "seriousnessother"] 

patient_id = ["patient.patientonsetage", "patient.patientonsetageunit", 
              "patient.patientsex", "patient.patientweight",
              "patient.patientdeath.patientdeathdate"] 

reaction_id = ["patient.reaction.reactionmeddraversionpt",
               "patient.reaction.reactionmeddrapt",
               "patient.reaction.reactionoutcome"]


