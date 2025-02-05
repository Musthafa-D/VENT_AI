# Equivariant prediction of steady state air flows with an equivariant graph neural network models. 

**Goal**: develop a steady state predictor of fluid dnamic quantities based on graph networks, capable of describing air flows patterns based on simulation-generated data.
The task is for steady state so only the last temporal "snapshot" of simulation is kept in this preprocessing step.

**Model**: taken from the original [SEGNN repository](https://github.com/RobDHess/Steerable-E3-GNN), with some personally added comments in the code.
The model is explained in detail in the original paper [*Geometric and Physical Quantities Improve E(3) Equivariant Message Passing*](https://arxiv.org/abs/2110.02905) (2022) by Brandstetter et al. 
Suggested reading [here](https://robdhess.github.io/Steerable-E3-GNN/).

This work is also inspired by [*SE(3) symmetry lets graph neural networks learn arterial velocity estimation from small datasets*](https://arxiv.org/abs/2302.08780) (2023) by Suk et al. 
Link to their repository [here](https://github.com/sukjulian/segnn-hemodynamics).

**Approach**: work with a point cloud that stores geometrical and physical information of points sampled from the computational CFD grid. The geometrical features are:

- point cloud of 3D nodes sampled from the computational grid, with labels on each node
- pressure (scalar), velocity (vector), stress tensor (3x3 symmetric)
- SDF: closest signed distance of a node from the surrounding walls
- MIS: Maximum Inscribed Sphere

- A label identifies the type of node:
    0: fluid node
    1: inlet node
    2: outlet node

Nodes are sampled as a fraction of the simulation grid, typically 1k-10k poits, depending on GPU memory availability.
The majority of nodes are fluid ones, the remaining are inlet or outlet boundary nodes (decided at simulation production time).
The MIS value is here preprocessed with an in-house approach to remove artifacts produced during initial calculation.

In addition, a small percentage of fluid nodes arbitrarily near the inlet boundary, are provided with pressure and velocity.
This should help the model generalize better when the steady state does not only depend on the geometry but on inlet conditions as well. 

### Installation
The method relies on specific version of pytorch_geometryic (or PyG). In particular version **2.0.3**
Use a conda environment and install from the SEGNN_STEADYSTATE/environment.yml file, for example:

conda env create -n segnn -f SEGNN_STEADYSTATE/environment.yml
conda activate -n segnn

Then run the jupyter notebooks with vscode by selecting the segnn conda environment.

### Folder SIMULATE_STEADYSTATE
Stuff to run a bunch of simulations with Moebius. Refer to the Moebius manual for detailed infos.

### Folder SEGNN_STEADYSTATE
Stuff to train and test. Subfolder segnn has scripts and utilities for the SO(3) equivariant GNN.

### Data
Kept in a separate folder, as specified in SEGNN_STEADYSTATE/params.py. It has the following subfolders:
- **probes**: nodes from point cloud: coordinates and labels
- **fields**: per-node fluid dynamic quantities
- **processed**: per-simulation .pt files
- **raw**: Dataset.pt containing the entire .pt graph data
- **checkpoints**: checkpoints for the training phase


### Pipeline
- If you already have simulation .np files and .pt, skip the **Simulation** part and jump to **Training and Testing**
**! Warning**, post-simulation .pt files depend on the version of torch/pyg used. So strange errors may appear if the raw/Dataset.raw was generated from a different version.

**params.py**: stores all relevant parameters for simulation and training

### Simulation (SIMULATE_STEADYSTATE)

**1 run_multi_sim.sh**: launches a bunch of simulations and generates **.np** files that contain the nodes (**probes**) and fluid dynamics data (**fields**)

**2 save_pt_tensor_from_npz_npy_files.ipynb**: reads all  .np files and processes them in single .pt files (PyTorch tensor file)

### Training and Testing (SEGNN_STEADYSTATE)

**1 read_dataset.ipynb**: preprocess the .pt file from every simulation and unpack in individual .pt files (storing a "data" object from PyG).

**2a Train.ipynb**: Training and validation. Best validation model is saved;

**2b Test.ipynb**: Used to test the model and to produce some plots;

### Ancillary files

**Utility_functions.py**: read data (dataset class), show graphs, preprocess equivariant features, etc;

**tryToRead_single_simulation_npz_and_npy_files.ipynb**: checks if the python pipeline reads and plots data from a single Moebius simulation. Not needed for the traininig/testing pipeline.

**first_script.ipynb**: Experiments with a single graph and the SEGNN model, to test rotation equivariance


### WandB
- From the **wandb.ai** dashboard, track training runs in real time.
From *Runs* check high level information about runtime, state, and other values put in the wandb.init() call in the script.
From *Workspace* check metrics stored in real time to manage your experiments: training loss, validation loss and so on.