# coding: utf-8
'''
TODO:
    Data import & training & test for the UNet
    Dependencies:
    Keras 2.0.8
    Tensorflow
    UNetPACT
'''

__author__ = 'ACM'

from Models.UNetPACT import UNet_PA, dice_coef_loss, dice_coef
from keras.optimizers import Adam
import Models.UNetConfig as uc


# model construction
model = UNet_PA(dropout_rate=uc.DROPOUT_RATE, batch_norm=uc.BATCH_NORM_FLAG)
# training setup
optimizer = Adam() # training optimizer
loss = dice_coef_loss # training loss function
metrics = [dice_coef] # training evaluation metrics
# model configuration
model.complile(optimizer=optimizer, loss=loss, metrics=metrics)

# model training
# model.fit()

