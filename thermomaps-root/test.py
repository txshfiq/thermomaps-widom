import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from ising.observables import Demixed
from ising.samplers import SwendsenWangSampler, SingleSpinFlipSampler
from ising.base import IsingModel

from data.trajectory import EnsembleTrajectory, MultiEnsembleTrajectory
from data.dataset import MultiEnsembleDataset
from data.generic import Summary

from tm.core.prior import GlobalEquilibriumHarmonicPrior, UnitNormalPrior
from tm.core.backbone import ConvBackbone
from tm.core.diffusion_model import DiffusionTrainer, SteeredDiffusionSampler
from tm.core.diffusion_process import VPDiffusion
from tm.architectures.unet_1d import Unet1D




trajectories = []
for temperature in np.arange(0.3, 3.5, 0.2):
  IM = IsingModel(sampler=SingleSpinFlipSampler, size = 8, warmup = 10, z = np.round(temperature,1))
  IM.simulate(steps = 500, observables = [Demixed()], sampling_frequency = 1)
  trajectories.append(IM.trajectory)
dataset = MultiEnsembleDataset(trajectories, Summary())
df = dataset.to_dataframe()


sub_df = df[df['temperature'].isin([1.5,2.5])]
sub_dataset = dataset.from_dataframe(sub_df)

train_loader, test_loader = sub_dataset.to_TMLoader(train_size=0.8, test_size=0.2, state_variables=['temperature'])
