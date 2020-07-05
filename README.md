# A Dicision-Based Dynamic Ensemble Selection Method for Concept Drift

paper from: https://ieeexplore.ieee.org/document/8995320  
R. A. S. Albuquerque, A. F. J. Costa, E. Miranda dos Santos, R. Sabourin and R. Giusti, "A Decision-Based Dynamic Ensemble Selection Method for Concept Drift," 2019 IEEE 31st International Conference on Tools with Artificial Intelligence (ICTAI), Portland, OR, USA, 2019, pp. 1132-1139, doi: 10.1109/ICTAI.2019.00158.

## Abstract
The main task of this paper is concept drift. Concept drift will occur when data are continuously generated in streams, data and target concepts may change over time. For this problem `drift detector` is a commom solution, so the author proposed an online method which monitoring the stabilization of class distribution over time which named `Dynamic Ensemble Selection for Drift Detection(DESDD)`.  

Online learning ensambles are used to tackle the drift problem. these methods update their knowledge base by adding, removing or updating classifiers. The drift detection mechanism mainly relies on an auxiliary drift detection such as leveraging bagging(LB) or ADWIN. These drift detectors have two common characteristic, 1) they incourage diversity only emplicitly by using algorithm such as the online versions of bagging and boosting; and 2) the generate only **one ensemble of classifiers**.  

According to the idea of the author, the model should estimate the class for each unknown instance, in order to raise the possibility of making a correct classification, the author then proposed an `ensemble-based method` which include **diverse population** of ensambles with different member's diversity called `dynamic ensemble selection(DES)`, which elect a single ensemble that is probabily the best qualified to predict the class for given sample.  

## Proposed Method
DESDD is divided into four step:  
1) **Diverse Ensemble Population Generation**
