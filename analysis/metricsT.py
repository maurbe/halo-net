"""
    Compute various metrics for all boxes: validation (T)
    Metrics: Full Dice, Full L2, Full R2
    Comment: the validation boxes already have the correct orientation
"""
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def dice_coef(y_true, y_pred, smooth=1.0):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

gt    = np.load('boxesT/gtT.npy')
pred  = np.load('boxesT/predictionT.npy')

print('Validation T')
print('Dice:\t', dice_coef(gt > 0.0, pred > 0.0))
print('L2:\t\t', mean_squared_error(gt.flatten(), pred.flatten()))
print('R2:\t\t', r2_score(gt.flatten(), pred.flatten()), '\n')