# A Dicision-Based Dynamic Ensemble Selection Method for Concept Drift

cited paper: https://ieeexplore.ieee.org/document/8995320  
R. A. S. Albuquerque, A. F. J. Costa, E. Miranda dos Santos, R. Sabourin and R. Giusti, "A Decision-Based Dynamic Ensemble Selection Method for Concept Drift," 2019 IEEE 31st International Conference on Tools with Artificial Intelligence (ICTAI), Portland, OR, USA, 2019, pp. 1132-1139, doi: 10.1109/ICTAI.2019.00158.

## Abstract
The main task of this paper is concept drift. Concept drift will occur when data are continuously generated in streams, data and target concepts may change over time. For this problem `drift detector` is a commom solution, so the author proposed an online method which monitoring the stabilization of class distribution over time which named `Dynamic Ensemble Selection for Drift Detection(DESDD)`.  

According to the idea of the author, the model should be able to estimate the class for each unknown instance, in order to raise the possibility of making a correct classification, the author then proposed an `ensemble-based method` which include **diverse population** of ensambles with different **member's diversity**, called `dynamic ensemble selection(DES)`, which elect a single ensemble that is probabily the best qualified to predict the class for given sample.  

## Proposed Method
DESDD is divided into four step:  
1. Diverse Ensemble Population Generation  
This phrase is to generate population for ensembles for decision-based dynamic selection. There are three parameters for this step, size of population , number of components of ensemble, and types of the components, the ensemble can be homogeneous or heterogeneous.
2. Dynamic Ensemble Selection  
This phrase aimed at choosing an expert ensemble to assign labels to unknown instance. The ensemble with highest prequential (predictive and sequential) accuracy until instance (x_t) will be choose to assign a label (y_t+1) for unknown instance (x_t+1).
3. Drift Detection  
Here the predicted label is compared to the true label to detect whether the prediction is correct or not, this information is used in error monitoring progress from drift detector to indicate the occurrence of drift. The author uses a tradition drift detector DDM and ADWIN with Leveraging Bagging as drift detector in this paper.
4. Drift Reaction  
In the last phrase, there are different possible scenarios. When a drift is indicated, a new population of ensembles will be created, if not, then the ensemble will be updated by training with new instance (x_t+1) and new label (y_t+1).

## How to run
### Installation


    install anaconda https://www.anaconda.com/products/individual


### Run


    run "pip install -r requirements.txt"

There are multiple scripts, each script name denots which dataset it loads and run model on. For example, `spam.py` works with spam.csv dataset. The datasets are inside dataset folders.

To run a script use the following command:

    python -W ignore script_name.py   

For example: python -W ignore spam.py

### Output

After running, there will be a graph, and the terminal will print the accuracy, drift count for model.

The results are in results folder.
