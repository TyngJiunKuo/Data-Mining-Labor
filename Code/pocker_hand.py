import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.naive_bayes import GaussianNB
from classifiers.detector_classifier import DetectorClassifier
from drift_detector.adwin import Adwin
from drift_detector.DDM import DDM
from evluation.metrics import Exact_match
from evluation.prequential import prequential_evaluation, get_errors

np.random.seed(0)

print('Loading data')


df = pd.read_csv("real_data/pocker_hand.csv").dropna()

df = df.drop(columns = ['idx'])
# extra pre-processing


L = 10
N_train = 500000

labels = df.columns.values.tolist()[L:]
data = df.values
data = data[:, :]
T = len(data)
Y = data[:, L:]
print(set(Y.reshape(-1,).tolist()))
X = data[:, 0:L]

print(Y.shape)
# print(Y)


print("Experimentation")

h = [DetectorClassifier(GaussianNB(), DDM())]
     #GaussianNB()]
E_pred, E_time, E_usage = prequential_evaluation(X, Y, h, N_train)

print("Evaluation")

E = np.zeros((len(h), T - N_train))
for m in range(len(h)):
    E[m] = get_errors(Y[N_train:], E_pred[m], J=Exact_match)

print("Plot Results")
print("---------------------------------------")
w = 200
fig, axes = plt.subplots(nrows=2, ncols=1)
fig.tight_layout()
for m in range(len(h)):
    acc = np.mean(E[m, :])
    time = np.mean(E_time[m, :])
    usage = np.mean(E_usage[m, :])
    if h[m].__class__.__name__ == 'DetectorClassifier':
        print(h[m].__class__.__name__)
        print(h[m].get_detector_name())
    else:
        print(h[m].__class__.__name__)
    print("Exact Match %3.2f" % np.mean(acc))
    # print("Running Time  %3.2f" % np.mean(time))
    if h[m].__class__.__name__ == 'DetectorClassifier':
        print("Number of detected drifts: %d" % h[m].num_change_detected)
    print("---------------------------------------")
    acc_run = np.convolve(E[m, :], np.ones((w,)) / w, 'same')
    acc_time = np.convolve(E_time[m, :], np.ones((w,)) / w, 'same')
    if h[m].__class__.__name__ != 'DetectorClassifier':
        plt.subplot(2, 1, 1)
        plt.plot(np.arange(len(acc_run)), acc_run, '-', label=h[m].__class__.__name__)
        plt.subplot(2, 1, 2)
        plt.plot(np.arange(len(acc_time)), acc_time, '-', label=h[m].__class__.__name__)
    else:
        plt.subplot(2, 1, 1)
        plt.plot(np.arange(len(acc_run)), acc_run, 'x', label=h[m].get_detector_name())
        plt.subplot(2, 1, 2)
        plt.plot(np.arange(len(acc_time)), acc_time, 'x', label=h[m].get_detector_name())


plt.subplot(2, 1, 1)
plt.xlabel('Instance(samples)')
plt.ylabel('Accuracy(exact match)')
plt.title('Performance(acc)')
plt.legend(loc='best')
plt.subplot(2, 1, 2)
plt.xlabel('Instance(samples)')
plt.ylabel('Running time(ms)')
plt.title('Performance(Running time)')
plt.legend(loc='best')
plt.show()
