openfda
=======

Correlating the OpenFDA Pharmacy Drugs and Reactions based on Adverse Events

### Overview
Adverse event reports submitted to FDA do not undergo extensive validation or verification.  Therefore a casual relationsip cannot be established between product and reactions listed in a report.



### Case Study
Focused on Elderly Patients that have reported serious conditions with hypertension.  Hypertension was chosen due to the large number of adverse reports available in contrast to other conditions.  At some point in time, while taking multiple prescribed drugs, they have indicated one of the drugs was for hypertension.
* serious:1
* patient.patientonsetage:[65+TO+99]
* patient.drug.drugindication:hypertension
    
### Datasets
* [openFDA](https://open.fda.gov/)
    * Adverse Events 
    * Recalls (Certain Recalls of FDA Regulated Products)
    * Labeling

### ML Aspects
* Data Munging, Scaling
* RBM (Restricted Boltzmann Machine) -  Neural Networks (Single Layer) 
* Cosine Similarity 

### Resources
#### Neural Networks
* [Deep Learning](http://www.deeplearning.net)
* [Theano](http://deeplearning.net/software/theano)
* http://www.cs.toronto.edu/~hinton/absps/guideTR.pdf
* http://www.cs.utoronto.ca/~hinton/absps/netflixICML.pdf
* http://blog.echen.me/2011/07/18/introduction-to-restricted-boltzmann-machines/

#### Reference Sites
* [ResearchAE](http://www.researchae.com/)

### External Source
* [RBM](https://github.com/echen/restricted-boltzmann-machines)

### Usage
* File Access: ipython test_vectors.py
* Restful OpenFDA API: ipython test_vectors.py -- -f 0