import matplotlib.pyplot as plt
import matplotlib

plt.rc('text', usetex=True)
plt.rcParams['axes.unicode_minus']=False    # works and adds minus signs!!!!!!
plt.rcParams["font.family"] = "cmr10"     # Change globally!!!

matplotlib.rcParams.update({'font.size': 15})
plt.rc('xtick', labelsize='small')
plt.rc('ytick', labelsize='small')
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'