'''
TODO:
    Configurations for UNet
    Hyper parameters
'''

__author__ = 'ACM'

# inputs
# input data
INPUT_SIZE = 128
INPUT_CHANNEL = 1   # 1-grayscale, 3-RGB scale
OUTPUT_MASK_CHANNEL = 1
# network structure
FILTER_NUM = 32 # number of basic filters for the first layer
FILTER_SIZE = 3 # size of the convolutional filter
DOWN_SAMP_SIZE = 2 # size of pooling filters
UP_SAMP_SIZE = 3 # size of upsampling filters

# network hyper-parameter
DROPOUT_RATE = 0.3
BATCH_NORM_FLAG = True