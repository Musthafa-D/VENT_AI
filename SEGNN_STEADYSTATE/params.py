
import os
import torch

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

### MARK: CONSTANTS

NFLUID_PROBES = 2000  # fluid nodes
NIO_PROBES = 100  # inlet nodes (input and output probes, the ones at the top and bottom(s) )

INLET_ID = 0  # inlet id

NSTEP = 10500
NSTEP = 7000
SPACING = 0.0008
NFRAME = 6 # depends on the saving frequency of probes, usually 1

NPOS = 6  # pos, mis, sdf, size (2 if fluid_node, 1 if outlet_node, 0 if inlet_node)
NFEATURES = 10  # p, vel, stress

### MARK: HYPERPARAMETERS

EPOCHS = 2
BATCH_SIZE = 1    
INPUT_SIZE = 8 # <- only for reference
EDGE_LMAX = 1
NODE_LMAX = 1
HIDDEN_LMAX = 1
NUM_LAYERS = 1
TASK = 'node'
NORM = 'batch'
OUTPUT_SIZE = 4
HIDDEN_SIZE= 16
NEIGHBOURS = 3 # 10 for moebius data
SUBSAMPLE_DATASET = 1
OPT = 'Adam'
# SCHEDULER = 'ExponentialLR' 
SCHEDULER = None 
if SCHEDULER == 'ExponentialLR':
    GAMMA = 0.95 # 0.8317 25 epochs, 0.8913 40 epochs 0.9261 60 epoch
LEARNING_RATE= 3e-3
DEVICE = dev
EARLY_STOP = 10

### MARK: DATASET

DATASET = 4

if DATASET == 1:
    NFLUID_PROBES = 125
    NIO_PROBES = 1
    INLET_ID = 0  # inlet id
    NFRAME = 1 

    NSIM = 1
    DATADIR = os.path.join("../.data/Extracted_data") # generated data
    BATCH_SIZE = 1

if DATASET == 2:
    NFLUID_PROBES = 61
    NIO_PROBES = 1
    INLET_ID = 0  # inlet id
    NFRAME = 1 

    NSIM = 10
    DATADIR = os.path.join("../.data/Dataset_10sims_5G2N") # generated data
    BATCH_SIZE = 1

if DATASET == 3:
    NFLUID_PROBES = 61 
    NIO_PROBES = 1
    INLET_ID = 0  # inlet id
    NFRAME = 1 

    NSIM = 1000
    DATADIR = os.path.join("../.data/Dataset_1000sims_5G2N") # generated data
    BATCH_SIZE = 1

if DATASET == 4:
    NFLUID_PROBES = 253
    NIO_PROBES = 1
    INLET_ID = 0  # inlet id
    NFRAME = 1 

    NSIM = 100
    DATADIR = os.path.join("../.data/Dataset_100sims_7G2N_healthy") # generated data
    BATCH_SIZE = 1

if DATASET == 5:
    NFLUID_PROBES = 2045
    NIO_PROBES = 1
    INLET_ID = 0  # inlet id
    NFRAME = 1 

    NSIM = 1000
    DATADIR = os.path.join("../.data/Dataset_1000sims_10G2N") # generated data
    BATCH_SIZE = 1

if DATASET == 6:
    NFLUID_PROBES = 12284
    NIO_PROBES = 1
    INLET_ID = 0  # inlet id
    NFRAME = 1 

    NSIM = 1
    DATADIR = os.path.join("../.data/Dataset_1sim_12G3N") # generated data
    BATCH_SIZE = 1

if DATASET == 7:
    NFLUID_PROBES = 509
    NIO_PROBES = 1
    INLET_ID = 0  # inlet id
    NFRAME = 1 

    NSIM = 1
    DATADIR = os.path.join("../.data/Dataset_1sim_8G2N_0.1inletvel") # generated data
    BATCH_SIZE = 1

if DATASET == 8:
    NFLUID_PROBES = 253
    NIO_PROBES = 1
    INLET_ID = 0  # inlet id
    NFRAME = 1 

    NSIM = 10000
    DATADIR = os.path.join("../.data/Dataset_10000sims_7G2N")
    BATCH_SIZE = 1