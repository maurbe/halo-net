"""
    Purpose:    Compute various metrics for simulation box A
    Metrics:    Full Dice, Full L2, Full R2
    Comment:    The validation boxes already have the correct orientation
"""
import numpy as np, os
from sklearn.metrics import mean_absolute_error, r2_score

def dice_coef(y_true, y_pred, smooth=1.0):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

homedir = os.path.dirname(os.getcwd()) + '/'
gt    = np.load(homedir + 'boxesT/gt_distancemap_normT.npy')
pred  = np.load(homedir + 'boxesT/predictionT.npy')

print('Training simulation (T):')
print('Dice:\t', dice_coef(gt > 0.0, pred > 0.0))
print('L1:\t\t', mean_absolute_error(gt.flatten(), pred.flatten()))
print('R2:\t\t', r2_score(gt.flatten(), pred.flatten()), '\n')