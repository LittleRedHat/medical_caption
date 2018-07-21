import torch
import torch.nn as nn
import argparse
import sys
sys.path.append('..')
from dataset.IU_Chest_XRay import ChestIUXRayDataset


def parse_args():
    parser = argparse.ArgumentParser("IU Chest XRay Report Generation")
    parser.add_argument()




def train():
    pass



def predict():
    pass

def main():
    train_dataset = ChestIUXRayDataset()

