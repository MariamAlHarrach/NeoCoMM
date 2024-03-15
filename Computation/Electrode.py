import numpy as np
import matplotlib.pyplot as plt
from scipy import fft, ifft, signal
import math
from scipy.io import loadmat
#plt.rcParams.update({'font.size': 26})


class ElectrodeModel:
    def __init__(self, re=62.5, sigma=33e-5, Etype=0,cyl=0,h=2000):
        # Electrode properties
        # re ---> electrode radius in micrometers
        # sigma ---> Tissue/Electrolyte conductivity in ohm - 1.mm - 1
        # Etype is for electrode material
        # Cyl is the electrode shape Cyl=0 => disc, cyl=1 => cylinder
        self.h=h
        self.re = re
        self.sigma = sigma
        self.Cs = 1 * 1e-8  # shunt capacity (shunt capacity to the ground value usually standard value)
        if cyl==1:
            self.s_e= 2* np.pi * self.re *self.h
        else:
            self.s_e = np.pi * self.re ** 2  # surface area in micrometer^2
        # Compute the spreading resistance value

        self.Rs = np.sqrt(np.pi)/(self.sigma * 1e-3 * 4 * np.sqrt(self.s_e))

        self.w = np.logspace(0, 4, 50)  # frequency vector

        # 0     ->      platinum (Pt)
        # 1     ->      Stainless steel (SS)

        # Model 1 : non-coated
        Rct_1 = {0: 1.5524e+14, 1:4.48e+13 }
        Cdl_1 = {0: 2.4794e-15, 1:  2.72e-13}
        n_1   = {0: 0.8833    , 1: 0.9}


        self.c = Cdl_1.get(Etype)  # F/mm2
        self.r = Rct_1.get(Etype)
        self.n = n_1.get(Etype)
        self.Rct = self.r / self.s_e
        self.Cdl = self.c * self.s_e


    def Zelec(self, w):
        Ze = np.zeros(shape=(1, len(w)), dtype=complex)
        for i in range(0, len(w)):
            Ze[0, i] = self.Rs + 1 / (1/self.Rct + (self.Cdl * 2 * np.pi * w[i] * 1j) ** self.n)
        return Ze


    def TF(self, w):
        # He = np.zeros(shape=(1, len(w)), dtype=complex)
        # for i in range(0, len(w)):
        #     He[0, i] = 1 / ((self.Rs * self.Cs * 2 * np.pi * w[i] * 1j) + ((self.Cs * 2 * np.pi * w[i] * 1j) / (1 / self.Rct + (self.Cdl * 2 * np.pi * w[i] * 1j) ** self.n)) + 1)
        He = 1 / ((self.Rs * self.Cs * 2 * np.pi * w * 1j) + ((self.Cs * 2 * np.pi * w * 1j) / (1 / self.Rct + (self.Cdl * 2 * np.pi * w * 1j) ** self.n)) + 1)

        return He

    def GetVelec(self, V, Fs):
        #  He is the electrode interface transfer function
        #  V is the LFP of the electrode
        Vf = fft.fft(V)
        freq = np.fft.fftfreq(len(V), d=1/Fs)
        #       print(freq)
        He = self.TF(freq)
        Velec = fft.ifft(He * Vf)
        return Velec.real

