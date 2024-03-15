__author__ = 'Mariam'

# -*- coding: utf-8 -*-
import numpy as np

# values are obtained from [Wang et al. 2018] as mean values of the different subtypes
# the soma of PC is modeled as a cone in layers 2/3, 5 and 6 and as a sphere in layer 4 for Spiny stellate cells


class Neuron:
    def __init__(self, type, layer, subtype=None):
        self.type = type
        self.layer = layer
        if subtype != None:
            self.subtype = subtype
        else:
            self.subtype = None
        self.d_soma = {  # soma diameter in micrometers/ type
            '1': 18.,  # Layer 2/3
            '2': 11.77,  # Layer 4
            '3': 24.44,  # Layer 5
            '4': 17.30,  # Layer 6

        }

        self.h_soma = {  # soma height in micrometers/ type
            '1': 6.32,
            '2': 7.57,
            '3': 23.90,
            '4': 16.94,

        }

        self.S_soma={ # surface area of soma (mean value from wang 2018) rat
            '1': 180.4,
            '2': 236.5,
            '3': 391,
            '4': 188.1,
        }

        self.S_dend={ # surface area of dendrites (mean value from wang 2018) rat
            '1': 5191,
            '2': 4845,
            '3': 16582,
            '4': 5620,
        }

        self.l_Ais={ # length 20-60 um from [gulledge2016neuron] rat
            '1': 40,
            '2': 40,
            '3': 40,
            '4': 40,
        }
        
        self.S_Ais={ # surface 20-60 um from [gulledge2016neuron] rat
            '1': 2*np.pi*40*10,
            '2': 2*np.pi*40*10,
            '3': 2*np.pi*40*10,
            '4': 2*np.pi*40*10,
        }


        self.d_soma_Ais={ # distance between some and the AIS 10-45 um from [gulledge2016neuron] rat
            '1': np.random.random_integers(10, 45, 1),
            '2': np.random.random_integers(10, 45, 1),
            '3': np.random.random_integers(10, 45, 1),
            '4': np.random.random_integers(10, 45, 1),
        }

        self.Adendrite_treelength = {
            '1': 275.8,
            '2': 499.,
            '3': 792.,
            '4': 593.,

        }

        self.Adendrite_treelength_h = {
            '1': 1000.,
            '2': 600.,
            '3': 1600.,
            '4': 900,

        }

        self.D1 = {
            '1': 164.,
            '2': 144.,
            '3': 150.,
            '4': 139.,

        }

        self.D2 = {
            '1': 25,
            '2': 216,
            '3': 329,
            '4': 114,

        }

        self.D3 = {
            '1': 86,
            '2': 139,
            '3': 314,
            '4': 340,

        }

        self.R2 = {
            '1': 10,
            '2': 10,
            '3': 10,
            '4': 10,

        }

        self.Adendrite_treewidth = {
            '1': 195,
            '2': 186,
            '3': 293,
            '4': 286,

        }

        self.Bdendrite_length = {
            '1': 176,
            '2': 203,
            '3': 260,
            '4': 208,
            'PV_BC': 150,
            'PV_Ch': 150,
            'SST': 250,
            'VIP': 150,
            'RLN': 150

        }

        self.Bdendrite_width = {
            '1': 206,
            '2': 272,
            '3': 293,
            '4': 243,
            'PV_BC': 150,
            'PV_Ch': 50,
            'SST': 100,
            'VIP': 60,
            'RLN': 150
        }

        self.axon_length = {
            '1': 735,
            '2': 1057,
            '3': 1014,
            '4': 682,
            'PV_BC': 300,  # interneuron
            'PV_Ch': 250,  # interneuron
            'SST': 250,  # interneuron
            'VIP': 600,  # interneuron
            'RLN1': 150,  # layer1
            'RLN2': 250  # layer2

        }

        self.axon_width = {
            '1': 410,
            '2': 400,
            '3': 400,
            '4': 400,
            'PV_BC': 200,  # interneuron
            'PV_Ch': 300,  # interneuron
            'SST': 150,  # interneuron
            'VIP': 150,  # interneuron
            'RLN1': 400,  # layer 1
            'RLN2': 120  # layer2/3

        }


        if type == 0:  # PC
            ###soma
            self.dsoma = self.d_soma[str(self.layer)]  # soma diameter
            self.hsoma = self.h_soma[str(self.layer)]  # soma height
            self.S_s=self.S_soma[str(self.layer)]
            self.S_d=self.S_dend[str(self.layer)]
            self.S_a=self.S_Ais[str(self.layer)]


            ####Axon
            if self.subtype in [0, 1]:  # if UPC,TPC
                self.AX_up = self.axon_length[str(self.layer)] * 1 / 3
                self.AX_down = -self.axon_length[str(self.layer)] * 2 / 3
            else:
                self.AX_up = self.axon_length[str(self.layer)] * 2 / 3
                self.AX_down = -self.axon_length[str(self.layer)] * 1 / 3

            self.AX_w = self.axon_width[str(self.layer)] / 2
            self.AIS_l=float(self.l_Ais[str(self.layer)])

            ####dendrites

            # d1 first cylinder length
            # d2 second cylinder length
            # d3 third cylinder/cone length

            self.d1 = self.D1[str(self.layer)]
            self.d2 = self.D2[str(self.layer)]
            self.d3 = self.D3[str(self.layer)]
            # total dendrite length
            self.Adend_l = self.D1[str(self.layer)] + self.D2[str(self.layer)] + self.D3[str(self.layer)]
            #
            self.Adend_w = self.Adendrite_treewidth[str(self.layer)] / 2
            self.Bdend_w = self.Bdendrite_width[str(self.layer)] / 2
            self.Bdend_l = self.Bdendrite_length[str(self.layer)] / 2

            if self.subtype == 0:
                self.c1_up = self.d1 / 2
                self.c1_down = -self.d1 / 2

                self.c2_up = self.c1_up + self.d2
                self.c2_down = self.c1_up

                self.c3_up = self.c2_up + self.d3
                self.c3_down = self.c2_up
                self.mid_dend=(self.d2+self.d3+self.d1-self.d1 / 2)/2


            elif self.subtype == 1:  # untufted
                self.c1_up = self.d1 / 2
                self.c1_down = -self.d1 / 2

                self.c2_up = self.c1_up + self.d2 + self.d3
                self.c2_down = self.c1_up

                self.mid_dend=(self.d2+self.d3+self.d1-self.d1 / 2)/2

            elif self.subtype == 2:  # inverted
                self.c1_up = self.d1 / 2
                self.c1_down = -self.d1 / 2

                self.c2_up = self.c1_down
                self.c2_down = self.c1_down - self.d2 - self.d3
                self.mid_dend=(self.d2+self.d3+self.d1-self.d1 / 2)/2


            elif self.subtype == 3:  # Bipolar
                self.c1_up = self.Adendrite_treelength[str(self.layer)] * 1 / 6
                self.c1_down = -self.Adendrite_treelength[str(self.layer)] * 1 / 6

                self.c2_up = self.Adendrite_treelength[str(self.layer)] * 3 / 4
                self.c2_down = self.c1_up

                self.c3_up = self.c1_down
                self.c3_down = -self.Adendrite_treelength[str(self.layer)] * 1 / 4
                self.mid_dend=self.Adendrite_treelength[str(self.layer)]/2-self.Adendrite_treelength[str(self.layer)] * 1 / 6



            elif self.subtype == 4:  # ssc
                self.c1_up = self.Bdendrite_length[str(self.layer)] * 1 / 2
                self.c2_down = -self.Bdendrite_length[str(self.layer)] * 1 / 2
                self.mid_dend = self.Bdendrite_length[str(self.layer)] * 1 / 4

            self.r1 = self.Bdendrite_width[str(self.layer)] / 2
            self.r2 = self.R2[str(self.layer)]
            self.r3 = self.Adendrite_treewidth[str(self.layer)]

        elif type == 1:  # PV
            if self.subtype == 0:
                self.c1_up = self.Bdendrite_length['PV_BC'] / 2
                self.c1_down = -self.Bdendrite_length['PV_BC'] / 2

                self.Bdend_w = self.Bdendrite_width['PV_BC'] / 2

                self.AX_up = self.axon_length['PV_BC'] / 2
                self.AX_down = -self.axon_length['PV_BC'] / 2
                self.AX_w = self.axon_width['PV_BC'] / 2
            else:
                self.c1_up = self.Bdendrite_length['PV_Ch']
                self.c1_down = 0
                self.Bdend_w = self.Bdendrite_width['PV_Ch'] / 2

                self.AX_up = 0
                self.AX_down = -self.axon_length['PV_Ch']
                self.AX_w = self.axon_width['PV_Ch'] / 2

        elif type == 2:  # SST
            self.c1_up = self.Bdendrite_length['SST'] / 2
            self.c1_down = -self.Bdendrite_length['SST'] / 2
            self.Bdend_w = self.Bdendrite_width['SST'] / 2

            self.AX_up = self.axon_length['SST'] / 2
            self.AX_down = -self.axon_length['SST'] / 2
            self.AX_w = self.axon_width['SST'] / 2
            self.AX_w2 = self.AX_w * 3

        elif type == 3:  # VIP

            self.AX_up = 0
            self.AX_down = -self.axon_length['VIP']
            self.AX_w = self.axon_width['VIP'] / 2

            self.c1_up = self.Bdendrite_length['VIP'] / 2
            self.c1_down = -self.Bdendrite_length['VIP'] / 2
            self.Bdend_w = self.Bdendrite_width['VIP'] / 2

        if type == 4:  # RLN
            self.c1_up = self.Bdendrite_length['RLN'] / 2
            self.c1_down = -self.Bdendrite_length['RLN'] / 2
            self.Bdend_w = self.Bdendrite_width['RLN'] / 2

            if self.layer == 0:
                self.AX_up = self.axon_length['RLN1'] / 2
                self.AX_down = -self.axon_length['RLN1'] / 2
                self.AX_w = self.axon_width['RLN1'] / 2
            else:
                self.AX_up = self.axon_length['RLN2'] / 2
                self.AX_down = -self.axon_length['RLN2'] / 2
                self.AX_w = self.axon_width['RLN2'] / 2

    def is_Same(self,type, layer, subtype):
        if type == self.type and layer == self.layer and  subtype == self.subtype :
            return True
        else:
            return False

    def update_type(self, type, layer, subtype=None):
        # print(type,layer,subtype)
        self.type = type
        self.layer = layer
        if subtype != None:
            self.subtype = subtype
        else:
            self.subtype = None
        if type == 0:  # PC
            ###soma
            self.dsoma = self.d_soma[str(self.layer)]  # soma diameter
            self.hsoma = self.h_soma[str(self.layer)]  # soma height
            self.S_s=self.S_soma[str(self.layer)]
            self.S_d=self.S_dend[str(self.layer)]
            self.S_a=self.S_Ais[str(self.layer)]
            ####Axon
            if self.subtype in [0, 1]:  # if UPC,TPC
                self.AX_up = self.axon_length[str(self.layer)] * 1 / 3
                self.AX_down = -self.axon_length[str(self.layer)] * 2 / 3
            else:
                self.AX_up = self.axon_length[str(self.layer)] * 2 / 3
                self.AX_down = -self.axon_length[str(self.layer)] * 1 / 3

            self.AX_w = self.axon_width[str(self.layer)] / 2
            self.AIS_l = float(self.l_Ais[str(self.layer)])

            ####dendrites

            # d1 first cylinder length
            # d2 second cylinder length
            # d3 third cylinder/cone length

            self.d1 = self.D1[str(self.layer)]
            self.d2 = self.D2[str(self.layer)]
            self.d3 = self.D3[str(self.layer)]
            # total dendrite length
            self.Adend_l = self.D1[str(self.layer)] + self.D2[str(self.layer)] + self.D3[str(self.layer)]
            #
            self.Adend_w = self.Adendrite_treewidth[str(self.layer)] / 2
            self.Bdend_w = self.Bdendrite_width[str(self.layer)] / 2
            self.Bdend_l = self.Bdendrite_length[str(self.layer)] / 2

            if self.subtype == 0:
                self.c1_up = self.d1 / 2
                self.c1_down = -self.d1 / 2

                self.c2_up = self.c1_up + self.d2
                self.c2_down = self.c1_up

                self.c3_up = self.c2_up + self.d3
                self.c3_down = self.c2_up
                self.mid_dend = (self.d2 + self.d3 + self.d1 - self.d1 / 2) / 2


            elif self.subtype == 1:  # untufted
                self.c1_up = self.d1 / 2
                self.c1_down = -self.d1 / 2

                self.c2_up = self.c1_up + self.d2 + self.d3
                self.c2_down = self.c1_up

                self.mid_dend = (self.d2 + self.d3 + self.d1 - self.d1 / 2) / 2

            elif self.subtype == 2:  # inverted
                self.c1_up = self.d1 / 2
                self.c1_down = -self.d1 / 2

                self.c2_up = self.c1_down
                self.c2_down = self.c1_down - self.d2 - self.d3
                self.mid_dend = (self.d2 + self.d3 + self.d1 - self.d1 / 2) / 2


            elif self.subtype == 3:  # Bipolar
                self.c1_up = self.Adendrite_treelength[str(self.layer)] * 1 / 6
                self.c1_down = -self.Adendrite_treelength[str(self.layer)] * 1 / 6

                self.c2_up = self.Adendrite_treelength[str(self.layer)] * 3 / 4
                self.c2_down = self.c1_up

                self.c3_up = self.c1_down
                self.c3_down = -self.Adendrite_treelength[str(self.layer)] * 1 / 4
                self.mid_dend = self.Adendrite_treelength[str(self.layer)] / 2 - self.Adendrite_treelength[
                    str(self.layer)] * 1 / 6



            elif self.subtype == 4:  # ssc
                self.c1_up = self.Bdendrite_length[str(self.layer)] * 1 / 2
                self.c2_down = -self.Bdendrite_length[str(self.layer)] * 1 / 2
                self.mid_dend = self.Bdendrite_length[str(self.layer)] * 1 / 4

            self.r1 = self.Bdendrite_width[str(self.layer)] / 2
            self.r2 = self.R2[str(self.layer)]
            self.r3 = self.Adendrite_treewidth[str(self.layer)]
        elif type == 1:  # PV
            if self.subtype == 0:
                self.c1_up = self.Bdendrite_length['PV_BC'] / 2
                self.c1_down = -self.Bdendrite_length['PV_BC'] / 2

                self.Bdend_w = self.Bdendrite_width['PV_BC'] / 2

                self.AX_up = self.axon_length['PV_BC'] / 2
                self.AX_down = -self.axon_length['PV_BC'] / 2
                self.AX_w = self.axon_width['PV_BC'] / 2
            else:
                self.c1_up = self.Bdendrite_length['PV_Ch']
                self.c1_down = 0
                self.Bdend_w = self.Bdendrite_width['PV_Ch'] / 2

                self.AX_up = 0
                self.AX_down = -self.axon_length['PV_Ch']
                self.AX_w = self.axon_width['PV_Ch'] / 2

        elif type == 2:  # SST
            self.c1_up = self.Bdendrite_length['SST'] / 2
            self.c1_down = -self.Bdendrite_length['SST'] / 2
            self.Bdend_w = self.Bdendrite_width['SST'] / 2

            self.AX_up = self.axon_length['SST'] / 2
            self.AX_down = -self.axon_length['SST'] / 2
            self.AX_w = self.axon_width['SST'] / 2
            self.AX_w2 = self.AX_w * 3

        elif type == 3:  # VIP

            self.AX_up = 0
            self.AX_down = -self.axon_length['VIP']
            self.AX_w = self.axon_width['VIP'] / 2

            self.c1_up = self.Bdendrite_length['VIP'] / 2
            self.c1_down = -self.Bdendrite_length['VIP'] / 2
            self.Bdend_w = self.Bdendrite_width['VIP'] / 2

        if type == 4:  # RLN
            self.c1_up = self.Bdendrite_length['RLN'] / 2
            self.c1_down = -self.Bdendrite_length['RLN'] / 2
            self.Bdend_w = self.Bdendrite_width['RLN'] / 2

            if self.layer == 0:
                self.AX_up = self.axon_length['RLN1'] / 2
                self.AX_down = -self.axon_length['RLN1'] / 2
                self.AX_w = self.axon_width['RLN1'] / 2
            else:
                self.AX_up = self.axon_length['RLN2'] / 2
                self.AX_down = -self.axon_length['RLN2'] / 2
                self.AX_w = self.axon_width['RLN2'] / 2



if __name__ == '__main__':
    neuron = Neuron(type=0, layer=1)
    print(neuron.dsoma)
