from datetime import datetime
from time import time
import numpy as np
from pynvml.smi import nvidia_smi
import json

###############################
#Author : Bertrand CABOT from IDRIS(CNRS)
#
########################


class Chronometer:
    def __init__(self, test, rank):
        self.test = test
        self.rank = rank
        self.time_perf_train = []
        self.time_perf_load = []
        self.time_perf_forward = []
        self.time_perf_backward = []
        self.power = []
        self.start_proc = None
        self.start_training = None
        self.start_dataload = None
        self.start_backward = None
        self.start_forward = None
        self.start_valid = None
        self.val_time = None
        self.time_point = None
        if rank == 0: self.nvsmi = nvidia_smi.getInstance()
        
    def power_measurement(self):
        if self.rank == 0:
            powerquery = self.nvsmi.DeviceQuery('power.draw')['gpu']
            for g in range(len(powerquery)):
                self.power.append(powerquery[g]['power_readings']['power_draw'])
    
    def tac_time(self, clear=False):
        if self.time_point == None or clear:
            self.time_point = time()
            return
        else:
            new_time = time() - self.time_point
            self.time_point = time()
            return new_time
    
    def clear(self):
        self.time_perf_train = []
        self.time_perf_load = []
        self.time_perf_forward = []
        self.time_perf_backward = []
        
    def start(self):
        if self.rank == 0: self.start_proc = datetime.now()
            
    def dataload(self):
        if self.rank == 0 and self.test:
            if self.start_dataload==None: self.start_dataload = time()
            else:
                self.time_perf_load.append(time() - self.start_dataload)
                self.start_dataload = None
                
    def training(self):
        if self.rank == 0 and self.test:
            if self.start_training==None: self.start_training = time()
            else:
                self.time_perf_train.append(time() - self.start_training)
                self.start_training = None
                
    def forward(self):
        if self.rank == 0 and self.test:
            if self.start_forward==None: self.start_forward = time()
            else:
                self.time_perf_forward.append(time() - self.start_forward)
                self.power_measurement()
                self.start_forward = None
                
    def backward(self):
        if self.rank == 0 and self.test:
            if self.start_backward==None: self.start_backward = time()
            else:
                self.time_perf_backward.append(time() - self.start_backward)
                self.start_backward = None
                
                
    def display(self):
        if self.rank == 0:
            print(">>> BENCHMARK RESULSTS >>>>>>>>")
            if self.test:
                print(">>> Training step (GPU computing) time : avg {} seconds (+/- {})".format(np.median(self.time_perf_train[1:]), np.std(self.time_perf_train[1:])))
                print(">>> SUBRESULTS")
                print(">>> Dataloading step time: avg {} seconds (+/- {})".format(np.mean(self.time_perf_load[1:]), np.std(self.time_perf_load[1:])))
                if len(self.power)>0: print(">>> Peak Power during training: {} W)".format(np.max(self.power)))
                
