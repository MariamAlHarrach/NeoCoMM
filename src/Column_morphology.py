__author__ = 'Mariam'

# -*- coding: utf-8 -*-
import numpy as np
import copy


#Column morphology for different species: human/rats/mice


class Column:
    def __init__(self,type=1):
        #Type: 0 for human, 1 for rats, 2 for mice
        Type=str(type)
        self.Len = {  # Column length on um [Defelipe et al. 2002] doi:10.1023/A:1024130211265
            '0': 2622.,  # human
            '1': 1827.,  # rat
            '2': 1210.,  # mouse
        }
        #Thickness of each layer in um [Defelipe et al. 2002] doi:10.1023/A:1024130211265
        self.L1_d={
            '0':235.,
            '1':123.,
            '2':69.,
        }


        self.L23_d={
            '0':295.+405.+370.,
            '1':457.,
            '2':235.,
        }

        self.L4_d={
            '0':285.,
            '1':152.,
            '2':208.,
        }

        self.L5_d={
            '0':552.,
            '1':321.+209.,
            '2':248.,
        }

        self.L6_d={
            '0':480.,
            '1':565.,
            '2':451.,
        }
        # density of neurons in the different layers in neurons/mm3 [[Defelipe et al. 2002] doi:10.1023/A:1024130211265]

        self.Dens1={
            '0': 8333,
            '1': 3472,
            '2': 18229,
        }

        self.Dens23={
            '0': 27205,
            '1': 61670,
            '2': 137645,
        }

        self.Dens4={
            '0': 46167,
            '1': 90965,
            '2': 181362,
        }

        self.Dens5={
            '0':  23076,
            '1': 40202,
            '2': 77765,
        }

        self.Dens6={
            '0':  16774,
            '1': 64286,
            '2': 122092,
        }



        self.Syn_NB={
            '0': 29807,
            '1':18018,
            '2':21133,
        }
        # column radius in um
        self.R={
            '0':300.,
            '1':210.,
            '2':210.,
        }
        self.Curvature = 2000
        #column length
        self.L=self.Len[Type]
        #Column diameter
        self.D=2*self.R[Type]
        #thickness of each layer of the column
        self.L_th=np.array([self.L1_d[Type], self.L23_d[Type], self.L4_d[Type], self.L5_d[Type], self.L6_d[Type]])
        #Number of synapses per cell for each column
        self.NB_Syn =self.Syn_NB[Type]
        #cell density in each layer
        self.Celldens=np.array([self.Dens1[Type], self.Dens23[Type], self.Dens4[Type], self.Dens5[Type], self.Dens6[Type]])*1e-9 #in neurons/um3
        #xortical column volume
        self.Volume=self.L_th*np.pi*self.R[Type]*self.R[Type]
        #cell number in each layer
        self.Layer_nbCells=self.Celldens* self.L_th*np.pi*self.R[Type]*self.R[Type]
        ##reduced numbers for simulation
        self.Layer_nbCells=self.Layer_nbCells/10
        #from Markram et al. 2015 rat model
        self.PYRpercent = np.array([0, 0.7, 0.9, 0.8, 0.9])
        self.INpercent = 1 - self.PYRpercent
        self.PVpercent = np.array([0, 0.3, 0.55, 0.45, 0.45])
        self.SSTpercent = np.array([0, 0.2, 0.25, 0.40, 0.35])
        self.VIPpercent = np.array([0, 0.40, 0.20, 0.15, 0.20])
        self.RLNpercent = np.array([1, 0.1, 0, 0, 0])
        self.GetCelltypes()
        # self.Layer_nbCells = np.array([322,7524, 4656, 6114,
        #                           12651]) / d  # markram et al. 2015total number of cells/neocortical column for each layer (L1-L2/3-L4-L5-L6)



        self.PCsubtypes_Percentage=np.array([[0,0,0,0,0],
                                      [0.9,0,0.1,0,0],
                                      [0.5,0.36,0,0,0.14],
                                      [0.81,0.19,0,0,0],
                                      [0.39,0.17,0.20,0.24,0]]) #TPC,UPC,IPC,BPC,SSC

        # self.PCsubtypes_Per=np.array([[0,0,0,0,0],
        #                               [0.9*self.NB_PYR[1],0,0.1*self.NB_PYR[1],0,0],
        #                               [0.5*self.NB_PYR[2],0.36*self.NB_PYR[2],0,0,0.14*self.NB_PYR[2]],
        #                               [self.NB_PYR[3]*0.81,self.NB_PYR[3]*0.19,0,0,0],
        #                               [self.NB_PYR[4]*0.39,self.NB_PYR[4]*0.17,self.NB_PYR[4]*0.20,self.NB_PYR[4]*0.24,0]]) #TPC,UPC,IPC,BPC,SSC
        self.PCsubtypes_Per= self.PCsubtypes_Percentage * self.NB_PYR[:, np.newaxis]
        self.List_celltypes = np.array([np.array([0]*self.Layer_nbCells_pertype[0][l] + [1]*self.Layer_nbCells_pertype[1][l] + [2]*self.Layer_nbCells_pertype[2][l] + [3]*self.Layer_nbCells_pertype[3][l] + [4]*self.Layer_nbCells_pertype[4][l]).astype(int) for l in range(len(self.Layer_nbCells))])
        self.GetCellsubtypes()
        #compute index for all column
        nbcells = np.sum(self.Layer_nbCells)
        self.layerS = np.zeros(nbcells,dtype=int)
        self.typeS = np.zeros(nbcells,dtype=int)
        self.indexS = np.zeros(nbcells,dtype=int)
        for ind in range(nbcells):
            self.layerS[ind], self.typeS[ind], self.indexS[ind] = self.All2layer(ind, self.Layer_nbCells, self.NB_PYR, self.NB_PV, self.NB_SST, self.NB_VIP,
                                               self.NB_RLN, self.List_celltypes)

    def update_morphology(self,L=None,
                          D=None,
                          Curvature=None,
                          L_th=None,
                          Layer_nbCells=None,
                          PYRpercent=None,  
                        PVpercent=None, 
                          SSTpercent=None, 
                          VIPpercent=None, 
                          RLNpercent=None, 
                          NB_PYR = None,
                            NB_PV_BC = None,
                            NB_PV_ChC = None,
                            NB_IN = None,
                            NB_PV = None,
                            NB_SST = None,
                            NB_VIP = None,
                            NB_RLN = None
    ):
        if not L is None:
            self.L=L
        #Column diameter 
        if not D is None:
            self.D=D

        # Column Curvature 
        if not Curvature is None:
            self.Curvature = Curvature 
        #thickness of each layer of the column 
        if not L_th is None:
            self.L_th=L_th
        #Number of synapses per cell for each column 
        #cell number in each layer 
        if not Layer_nbCells is None:
            self.Layer_nbCells=Layer_nbCells 
        #from Markram et al. 2015 rat model
        if not PYRpercent is None:
            self.PYRpercent=PYRpercent  
        self.INpercent = 1 - self.PYRpercent
        if not PVpercent is None:
            self.PVpercent=PVpercent   
        if not SSTpercent is None:
            self.SSTpercent=SSTpercent   
        if not VIPpercent is None:
            self.VIPpercent=VIPpercent   
        if not RLNpercent is None:
            self.RLNpercent=RLNpercent   
        
        self.GetCelltypes(NB_PYR = NB_PYR,
                            NB_PV_BC = NB_PV_BC,
                            NB_PV_ChC = NB_PV_ChC,
                            NB_IN = NB_IN,
                            NB_PV = NB_PV,
                            NB_SST = NB_SST,
                            NB_VIP = NB_VIP,
                            NB_RLN = NB_RLN)
        # self.Layer_nbCells = np.array([322,7524, 4656, 6114,
        #                           12651]) / d  # markram et al. 2015total number of cells/neocortical column for each layer (L1-L2/3-L4-L5-L6)



        # self.PCsubtypes_Per=np.array([[0,0,0,0,0],
        #                               [0.9*self.NB_PYR[1],0,0.1*self.NB_PYR[1],0,0],
        #                               [0.5*self.NB_PYR[2],0.36*self.NB_PYR[2],0,0,0.14*self.NB_PYR[2]],
        #                               [self.NB_PYR[3]*0.81,self.NB_PYR[3]*0.19,0,0,0],
        #                               [self.NB_PYR[4]*0.39,self.NB_PYR[4]*0.17,self.NB_PYR[4]*0.20,self.NB_PYR[4]*0.24,0]]) #TPC,UPC,IPC,BPC,SSC
        self.PCsubtypes_Per= self.PCsubtypes_Percentage * self.NB_PYR[:, np.newaxis]
        self.List_celltypes = np.array([np.array([0]*self.Layer_nbCells_pertype[0][l] + [1]*self.Layer_nbCells_pertype[1][l] + [2]*self.Layer_nbCells_pertype[2][l] + [3]*self.Layer_nbCells_pertype[3][l] + [4]*self.Layer_nbCells_pertype[4][l]).astype(int) for l in range(len(self.Layer_nbCells))])
        self.GetCellsubtypes()
        nbcells = np.sum(self.Layer_nbCells)
        self.layerS = np.zeros(nbcells,dtype=int)
        self.typeS = np.zeros(nbcells,dtype=int)
        self.indexS = np.zeros(nbcells,dtype=int)
        for ind in range(nbcells):
            self.layerS[ind], self.typeS[ind], self.indexS[ind] = self.All2layer(ind, self.Layer_nbCells, self.NB_PYR, self.NB_PV, self.NB_SST, self.NB_VIP,
                                               self.NB_RLN, self.List_celltypes)

    def GetCelltypes(self, NB_PYR = None,
                            NB_PV_BC = None,
                            NB_PV_ChC = None,
                            NB_IN = None,
                            NB_PV = None,
                            NB_SST = None,
                            NB_VIP = None,
                            NB_RLN = None): 
        if NB_PYR is None:
            NB_PYR = self.PYRpercent * self.Layer_nbCells
            self.NB_PYR = NB_PYR.astype(int)
        else:
            self.NB_PYR = NB_PYR

        if NB_IN is None:
            NB_IN = self.INpercent * self.Layer_nbCells
            self.NB_IN = NB_IN.astype(int)
        else:
            self.NB_IN =NB_IN

        if NB_PV is None:
            NB_PV = self.PVpercent * self.NB_IN
            self.NB_PV = NB_PV.astype(int)
        else:
            self.NB_PV =NB_PV
        
        if NB_PV_BC is None:
            NB_PV_BC = 0.7 * self.NB_PV
            self.NB_PV_BC = NB_PV_BC.astype(int)
        else:
            self.NB_PV_BC = NB_PV_BC
        if NB_PV_ChC is None:
            NB_PV_BC = 0.7 * self.NB_PV
            self.NB_PV_ChC = (self.NB_PV - self.NB_PV_BC).astype(int)
        else:
            self.NB_PV_ChC = NB_PV_ChC
            
        if NB_SST is None:
            NB_SST = self.SSTpercent * self.NB_IN
            self.NB_SST = NB_SST.astype(int)
            print('SST', self.NB_SST)
        else:
            self.NB_SST = NB_SST
        if NB_VIP is None:
            NB_VIP = self.VIPpercent * self.NB_IN
            self.NB_VIP = NB_VIP.astype(int)
            print('VIP', self.NB_VIP)
        else:
            self.NB_VIP = NB_VIP
        if NB_RLN is None:
            NB_RLN = self.RLNpercent * self.NB_IN
            self.NB_RLN = NB_RLN.astype(int)
            print('RLN', self.NB_RLN)
        else:
            self.NB_RLN = NB_RLN

        #####External afferences
        self.NB_DPYR = int(np.sum(self.NB_PYR) * 0.07)
        self.NB_Th = int(np.sum(self.NB_PYR) * 0.07)
        
        self.Layer_nbCells = self.NB_PYR+self.NB_SST+self.NB_PV+self.NB_VIP+self.NB_RLN  # total number of cells/neocortical column for each layer (L1-L2/3-L4-L5-L6)
        self.Layer_nbCells_pertype=[self.NB_PYR,self.NB_PV,self.NB_SST,self.NB_VIP,self.NB_RLN]


    def GetCellsubtypes(self):
        self.List_cellsubtypes =copy.deepcopy(self.List_celltypes)
        PCsubtypes_Per = np.cumsum(self.PCsubtypes_Per,axis=1)

        for l in range(len(self.Layer_nbCells)):
            for cell in range(int(self.Layer_nbCells[l])):

                if (self.List_celltypes[l][cell] == 0):  # getsubtype of PC
                    if cell < PCsubtypes_Per[l][0]:
                        subtype = 0  # TPC
                    elif (cell >= PCsubtypes_Per[l][0]) and (cell < PCsubtypes_Per[l][1]):
                        subtype = 1  # UPC
                    elif (cell >= PCsubtypes_Per[l][1]) and (cell < PCsubtypes_Per[l][2]):
                        subtype = 2  # IPC
                    elif (cell >= PCsubtypes_Per[l][2]) and (cell < PCsubtypes_Per[l][3]):
                        subtype = 3  # BPC
                    elif (cell >= PCsubtypes_Per[l][3]) and (cell < PCsubtypes_Per[l][4]):
                        subtype = 4  # SSC

                    self.List_cellsubtypes[l][cell] = subtype

                #if PV check is chandeliers or Basket
                elif (self.List_celltypes[l][cell] == 1):  # PV get subtype
                    if (cell - self.NB_PYR[l]) < self.NB_PV_BC[l]:
                        subtype = 0  # BC
                    else:
                        subtype = 1  # Chandelier
                    self.List_cellsubtypes[l][cell] = subtype
                else:
                    self.List_cellsubtypes[l][cell] = -1

    def All2layer(self, indexall, NB_Cells, NB_pyr, NB_PV, NB_SST, NB_VIP, NB_RLN,
                  celltype):  # tranform index of a cell in the network into layer and index in the layer
        layer = []
        new_i = []  # index in the layer
        i = []  # index per type
        if indexall < NB_Cells[0]:  # layer 1
            layer = 0
            new_i = indexall
        elif indexall >= NB_Cells[0:1] and indexall < np.sum(NB_Cells[0:2]):  # layer 2/3
            layer = 1
            new_i = indexall - NB_Cells[0]
        elif indexall >= np.sum(NB_Cells[0:2]) and indexall < np.sum(NB_Cells[0:3]):  # Layer 4
            layer = 2
            new_i = indexall - np.sum(NB_Cells[0:2])
        elif indexall >= np.sum(NB_Cells[0:3]) and indexall < np.sum(NB_Cells[0:4]):  # Layer 5
            layer = 3
            new_i = indexall - np.sum(NB_Cells[0:3])
        elif indexall >= np.sum(NB_Cells[0:4]) and indexall < np.sum(NB_Cells[0:5]):  # Layer 6
            layer = 4
            new_i = indexall - np.sum(NB_Cells[0:4])

        type = int(celltype[layer][new_i])

        if type == 0:  # PC
            i = new_i + np.sum(NB_pyr[0:layer])
        if type == 1:  # PV
            i = new_i + np.sum(NB_PV[0:layer]) - np.sum(NB_pyr[layer])
        if type == 2:  # SST
            i = new_i + np.sum(NB_SST[0:layer]) - np.sum(NB_pyr[layer]) - np.sum(NB_PV[layer])
        if type == 3:  # VIP
            i = new_i + np.sum(NB_VIP[0:layer]) - np.sum(NB_pyr[layer]) - np.sum(NB_PV[layer]) - np.sum(NB_SST[layer])
        if type == 4:  # RLN
            i = new_i + np.sum(NB_RLN[0:layer]) - np.sum(NB_pyr[layer]) - np.sum(NB_PV[layer]) - np.sum(
                NB_SST[layer]) - np.sum(NB_VIP[layer])

        return layer, type, i


if __name__ == '__main__':
    Column = Column(type=0)





