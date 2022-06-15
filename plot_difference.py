import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


import os 


trained_results = "/Jarvis/workspace/gwangjin/dehazing/cvf-results/reside_unetbl_weighted_sl1/weighted/1/quantitative_results.txt"
dcp_results = "/Jarvis/workspace/gwangjin/dehazing/cvf-results/reside_standard_dcp.txt"
dataset_results = "/Jarvis/workspace/gwangjin/dehazing/cvf-results/reside_standard_hazy_clear_comparison.txt"


train_psnr = []
train_ssmi = []

for line in open(trained_results, 'r').readlines():
    if not line.startswith('batch'):
        continue
    line_split = line.split(',')
    psnr = line_split[1].split(' ')[3]
    ssmi = line_split[2].split(' ')[3]

    train_psnr.append(float(psnr))
    train_ssmi.append(float(ssmi))


dcp_psnr = []
dcp_ssmi = []

for line in open(dcp_results, 'r').readlines():
    if not line.startswith('batch'):
        continue
    line_split = line.split(',')
    psnr = line_split[1].split(' ')[3]
    ssmi = line_split[2].split(' ')[3]

    dcp_psnr.append(float(psnr))
    dcp_ssmi.append(float(ssmi))
    

dataset_psnr = []
dataset_ssmi = []

for line in open(dataset_results, 'r').readlines():
    if not line.startswith('batch'):
        continue
    line_split = line.split(',')
    psnr = line_split[1].split(' ')[3]
    ssmi = line_split[2].split(' ')[3]

    dataset_psnr.append(float(psnr))
    dataset_ssmi.append(float(ssmi))

def argsort(seq):
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(seq)), key=seq.__getitem__)


import numpy as np

dataset_psnr_argsort = argsort(dataset_psnr)

print(dataset_psnr_argsort)

plt.clf()

plt.plot(np.array(train_psnr)[dataset_psnr_argsort], label='trained', alpha=0.5)
plt.plot(np.array(dcp_psnr)[dataset_psnr_argsort], label='dcp', alpha=0.5)
plt.plot(np.array(dataset_psnr)[dataset_psnr_argsort], label='lowerbound', alpha=0.5)

plt.legend()

plt.ylabel('psnr')
plt.xlabel('batch')

plt.savefig("/Jarvis/workspace/gwangjin/dehazing/cvf-results/reside_unetbl_weighted_sl1/weighted/1/quantitative_result_psnr_comparision.png")



dataset_ssim_argsort = argsort(dataset_ssmi)


plt.clf()

plt.plot(np.array(train_ssmi)[dataset_ssim_argsort], label='trained', alpha=0.5)
plt.plot(np.array(dcp_ssmi)[dataset_ssim_argsort], label='dcp', alpha=0.5)
plt.plot(np.array(dataset_ssmi)[dataset_ssim_argsort], label='lowerbound', alpha=0.5)

plt.legend()

plt.ylabel('ssmi')
plt.xlabel('batch')

plt.savefig("/Jarvis/workspace/gwangjin/dehazing/cvf-results/reside_unetbl_weighted_sl1/weighted/1/quantitative_result_ssmi_comparision.png")