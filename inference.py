"""
An Inference run is a run of a previous stored 
    
"""

import yaml

import argparse


if __name__ == '__main__':    
    ap = argparse.ArgumentParser()
    ap.add_argument("-run", required=False, help="Name of the environment you want to run : optional run number eg. Cartpole:10")
    
    