__author__ = 'Mariam'


import numpy as np
from math import *
from scipy import signal
from Tissue import Cell_morphology
from scipy.spatial import distance


class LFP:
    def __init__(self, Fs=25000, type=0,re=10,h=2000, tx=0, ty=0,pos=None):
        # Electrode properties
        #re: electrode radius
        #tx:  tetax angle
        #ty: teta y angle
        #type=0 ---> disc, 1 ---> Cylinder
        self.type=type
        if pos is None:
            self.electrode_pos = [0e-3, 0e-3, 2000e-3]
        else:
            self.electrode_pos=pos #in micrometers
        self.r_e = re
        self.tx = tx
        self.ty = ty
        self.h=h

        self.C=Cell_morphology.Neuron(0,1)
        #Geometrical characteristics of the Soma
        S_soma = np.array([self.C.S_soma['1'],self.C.S_soma['2'],self.C.S_soma['3'],self.C.S_soma['4']])  # soma surface for layers 2/3,4,5 and 6 resp in micrometres
        l_soma = np.array([self.C.h_soma['1'],self.C.h_soma['2'],self.C.h_soma['3'],self.C.h_soma['4']])   # soma height for layers 2/3,4,5 and 6 resp in micrometres
        #Geometrical characteristics of the dendrite
        S_dend = np.array([self.C.S_dend['1'],self.C.S_dend['2'],self.C.S_dend['3'],self.C.S_dend['4']])  # dendrite surface for layers 2/3,4,5 and 6 resp in micrometres
        l_dend = np.array([self.C.Adendrite_treelength['1'],self.C.Adendrite_treelength['2'],self.C.Adendrite_treelength['3'],self.C.Adendrite_treelength['4']])  # dendrite tree length
        #Geometrical characteristics of the AIS
        S_AIS = np.array([self.C.S_Ais['1'],self.C.S_Ais['2'],self.C.S_Ais['3'],self.C.S_Ais['4']])
        l_AIS = np.array([self.C.l_Ais['1'],self.C.l_Ais['2'],self.C.l_Ais['3'],self.C.l_Ais['4']])   # soma height for layers 2/3,4,5 and 6 resp in micrometres


        # # proportion of soma to total area
        # p = [0.09, 0.04, 0.042, 0.062] #0.15

        #distance between sink and source
        self.lsd = l_soma / 2 + l_dend / 2
        self.lsa = l_soma / 2 + l_AIS / 2
        Ssd = S_soma+S_dend
        Ssa = S_soma+S_AIS

        gc = 1e-8  # intercompartment conductance 1mS/cm2 --->1e-8 mS/micrometre

        # Electrolyte conductivity
        self.sigma = 33 * 1e-5  # conductivity = 33.0 * 10e-5 ohm - 1.mm - 1

        self.Ksd = gc * self.lsd * Ssd / (4 * np.pi * self.sigma)
        self.Ksa = gc *  self.lsa * Ssa / (4 * np.pi * self.sigma)

        ####Tables of monopolar Percentages for LFP computing#######"
        #for Layer 23
        PC23=np.array(
            [[0, 20, 0, 0,0, 80], #if the input is at layer 1 of the dendrite then 20 % of the output is at layer 2/3 and 80% at the soma
             [10, 0, 0, 0, 0, 90],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0]])#in the case of inverted the input to the dendrites can be at layer 4
        #for Layer 4

        PC4=np.array(
            [[0, 30, 0, 0,0, 90],
             [20, 0, 0, 0, 0, 80],
             [10, 0, 0, 0, 0, 90],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0]])

        #for Layer 5

        PC5=np.array(
            [[0, 30, 0, 0,0, 70], #if the input is at layer 1 of the dendrite then 30 % of the output is at layer 2/3 and 70% at the soma
             [20, 0, 0, 0, 0, 80],
             [10, 0, 0, 0, 0, 90],
             [20, 10, 10, 0, 0, 50],
             [0, 0, 0, 0, 0, 0]])

        PC6=np.array(
            [[0, 0, 0, 0, 0, 0], #if the input is at layer 1 of the dendrite then 30 % of the output is at layer 2/3 and 70% at the soma
             [0, 0, 30, 0, 0, 70],
             [0, 10, 0, 20, 0, 70],
             [0, 0, 0, 0, 0, 100],
             [0, 0, 0, 10, 0, 90]])

        self.P=np.array([PC23,PC4,PC5,PC6])

    def getDiscPts(self, rad, step=0.01):


        i_coords, j_coords = np.meshgrid(np.arange(-rad,rad,step), np.arange(-rad,rad,step), indexing='ij')

        corrds = np.stack((i_coords.flatten(), j_coords.flatten(), 0*i_coords.flatten()) ).T

        dist = np.linalg.norm(corrds-np.array([0,0,0]),axis=1)

        D  = corrds[dist<=rad]

        return D


    def getCylPts(self, rad=400,th=2000,max_pts=1000):

        z = np.linspace(0, th, int(np.sqrt(max_pts)))
        theta = np.linspace(0, 2 * np.pi, int(np.sqrt(max_pts)))
        theta_grid, z_grid = np.meshgrid(theta, z)
        x_grid = rad * np.cos(theta_grid)
        y_grid = rad * np.sin(theta_grid)
        Cyl=np.stack((x_grid.flatten(), y_grid.flatten(), z_grid.flatten())).T
        return Cyl



    def get_electrode_coordinates(self, nbpots=1000):
        if self.r_e ==0:
            Ds = np.array([self.electrode_pos])
        else:
            if self.type==0: ###disc
                D = self.getDiscPts(self.r_e,step=1/((nbpots/(np.pi*self.r_e*self.r_e))**0.5))
            elif self.type==1: ###Cylinder
                D = self.getCylPts(rad=self.r_e, th=self.h, max_pts=nbpots)
            rx = np.array([[1, 0, 0], [0, cos(radians(self.tx)), - sin(radians(self.tx))],
                           [0, sin(radians(self.tx)), cos(radians(self.tx))]])
            ry = np.array([[cos(radians(self.ty)), 0, sin(radians(self.ty))],
                           [0, 1, 0], [-sin(radians(self.ty)), 0, cos(radians(self.ty))]])
            Ds =  np.matmul(D, np.matmul(rx, ry)) + self.electrode_pos
        return Ds


    def addnoise(self, lfp, SNR=35):
        Plfp = np.mean(lfp ** 2)
        Pnoise = Plfp / 10 ** (SNR / 10)
        noise = np.random.normal(0, np.sqrt(Pnoise), len(lfp))
        lfpn = lfp + noise
        return lfpn



    def get_surfaces(self,Cellpos,List_celltypes,List_cellsubtypes):
        S_d = []
        S_S = []
        S_A = []


        target = Cell_morphology.Neuron(0, 1)
        ## update neuron and get coordinates
        for layer in range(len(Cellpos)):
            for n in range(Cellpos[layer].shape[0]):
                try:
                    x=List_celltypes[layer][n]
                    if x == 0:
                        x = List_celltypes[layer][n]
                except:
                    x = List_celltypes[layer][0][n]
                if x== 0:
                    pos = Cellpos[layer][n]
                    # subtype == 0  # TPC  subtype = 1  # UPC subtype = 2  # IPC subtype = 3  # BPC subtype = 4  # SSC
                    try:
                        y =  List_cellsubtypes[layer][n]
                        target.update_type(0, layer=layer, subtype=y)
                    except:
                        y = List_cellsubtypes[layer][0][n]
                    subtype = y
                    target.update_type(0, layer=layer, subtype=subtype)

                    S_S.append(target.S_s)
                    S_d.append(target.S_d)
                    S_A.append(target.S_a)

        S=np.array(S_S)
        D=np.array(S_d)
        A=np.array(S_A)

        return S,D, A

    def get_projection(self,Cellpos,List_celltypes,List_cellsubtypes,Layertop_pos):

        ## initialize the coordiantes of the neuron's parts
        CellPosition_d = []
        CellPosition_d1 = []
        CellPosition_d23 = []
        CellPosition_d4 = []
        CellPosition_d5 = []
        CellPosition_d6 = []
        CellPosition_s_mid = []
        CellPosition_a = []
        ## baseline neuron
        target = Cell_morphology.Neuron(0, 1)
        ## update neuron and get coordinates
        for layer in range(len(Cellpos)):
            for n in range(Cellpos[layer].shape[0]):
                try:
                    x=List_celltypes[layer][n]
                    if x == 0:
                        x = List_celltypes[layer][n]
                except:
                    x = List_celltypes[layer][0][n]
                if x== 0:
                    pos = Cellpos[layer][n]
                    # subtype == 0  # TPC  subtype = 1  # UPC subtype = 2  # IPC subtype = 3  # BPC subtype = 4  # SSC
                    try:
                        y =  List_cellsubtypes[layer][n]
                        target.update_type(0, layer=layer, subtype=y)
                    except:
                        y = List_cellsubtypes[layer][0][n]
                    subtype = y
                    target.update_type(0, layer=layer, subtype=subtype)
                    d1 = np.array([pos[0], pos[1], Layertop_pos[4]])
                    d23 = np.array([pos[0], pos[1], Layertop_pos[3]])
                    d4 = np.array([pos[0], pos[1], Layertop_pos[2]])
                    d5 = np.array([pos[0], pos[1], Layertop_pos[1]])
                    d6 = np.array([pos[0], pos[1], Layertop_pos[0]])
                    s_mid = np.array([pos[0], pos[1], pos[2]])


                    CellPosition_d1.append(d1)
                    CellPosition_d23.append(d23)
                    CellPosition_d4.append(d4)
                    CellPosition_d5.append(d5)
                    CellPosition_d6.append(d6)
                    CellPosition_s_mid.append(s_mid)

                    if subtype in [0, 1, 3, 4]:## subtype ==TPC, UPC, BPC or SSC
                        CellPosition_d.append(np.array([pos[0], pos[1], pos[2] + target.mid_dend]))
                        CellPosition_a.append(np.array([pos[0], pos[1], pos[2] - target.AIS_l - target.hsoma / 2]))

                    else: ##fir IPC
                        CellPosition_d.append(np.array([pos[0], pos[1], pos[2] - target.mid_dend]))
                        CellPosition_a.append(np.array([pos[0], pos[1], pos[2] + target.AIS_l + target.hsoma / 2]))
        ### list to array


        CellPosition_a = np.array(CellPosition_a)
        CellPosition_d = np.array(CellPosition_d)
        CellPosition_d1 = np.array(CellPosition_d1)
        CellPosition_d23 = np.array(CellPosition_d23)
        CellPosition_d4 = np.array(CellPosition_d4)
        CellPosition_d5 = np.array(CellPosition_d5)
        CellPosition_d6 = np.array(CellPosition_d6)
        CellPosition_s_mid = np.array(CellPosition_s_mid)
        CellPosition_dend = np.array([CellPosition_d1,CellPosition_d23,CellPosition_d4,CellPosition_d5,CellPosition_d6])


        ###get electrode coordinates
        E_xyz = self.get_electrode_coordinates(nbpots=100)

        ###compute the distance between the electrode and the location of the transmembrane current
        Dis_d = distance.cdist(E_xyz, CellPosition_d, 'euclidean')
        Dis_d1 = distance.cdist(E_xyz, CellPosition_d1, 'euclidean')
        Dis_d23 = distance.cdist(E_xyz, CellPosition_d23, 'euclidean')
        Dis_d4 = distance.cdist(E_xyz, CellPosition_d4, 'euclidean')
        Dis_d5 = distance.cdist(E_xyz, CellPosition_d5, 'euclidean')
        Dis_d6 = distance.cdist(E_xyz, CellPosition_d6, 'euclidean')
        Dis_s_mid = distance.cdist(E_xyz, CellPosition_s_mid, 'euclidean')
        Dis_a = distance.cdist(E_xyz, CellPosition_a, 'euclidean')
        Dis_dend =np.array([distance.cdist(E_xyz, CellPosition_dend[id], 'euclidean')  for id in [0,1,2,3,4]])



        return  Dis_d, Dis_d1,Dis_d23, Dis_d4,Dis_d5,Dis_d6, Dis_s_mid, Dis_a,Dis_dend


    def compute_LFP_CSD(self,I_all,PPS,Cellpos,List_celltypes,List_cellsubtypes,Layertop_pos):

        PPSE = PPS['PPSE']
        PPSI = PPS['PPSI']
        pyrPPSI_s = PPS['PPSI_s']
        pyrPPSI_a = PPS['PPSI_a']
        pyrPPSE_Dpyr=PPS['PPSE_Dpyr']
        pyrPPSE_Th=PPS['PPSE_Th']
        #positions
        Dis_d, Dis_d1, Dis_d23, Dis_d4, Dis_d5, Dis_d6, Dis_s_mid, Dis_a,Dis_dend = self.get_projection(Cellpos, List_celltypes,
                                                                                               List_cellsubtypes,
                                                                                               Layertop_pos)

        S_S,S_d,S_A= self.get_surfaces(Cellpos, List_celltypes, List_cellsubtypes)

        self.pyrI_S= I_all['pyrI_S']
        self.pyrI_d= I_all['pyrI_d']
        self.pyrI_A= I_all['pyrI_A']

        ###compute currents sources from voltage dependant channels



        if np.size(self.pyrI_S)==1:
            # I_S = np.array([1e-8*S_S[k] * self.pyrI_S[0][0][k] for k in range(len(S_S))]) # PYRI in mA/cm2, I_S in mA
            # I_d = np.array([1e-8*S_d[k] * self.pyrI_d[0][0][k] for k in range(len(S_S))])
            # I_A = np.array([1e-8*S_A[k] * self.pyrI_A[0][0][k] for k in range(len(S_S))])


            I_S = 1e-8*self.pyrI_S[0][0]
            I_d =1e-8*self.pyrI_d[0][0]
            I_A =1e-8*self.pyrI_A[0][0]

        else:
            # I_S = np.array([1e-8*S_S[k] * self.pyrI_S[k] for k in range(len(S_S))])
            # I_d = np.array([1e-8*S_d[k] * self.pyrI_d[k] for k in range(len(S_S))])
            # I_A = np.array([1e-8*S_A[k] * self.pyrI_A[k] for k in range(len(S_S))])

            I_S = 1e-8*self.pyrI_S
            I_d =1e-8*self.pyrI_d
            I_A =1e-8*self.pyrI_A

        LFPsoma = np.mean(np.dot(1 / Dis_s_mid,I_S), axis=0)
        LFPdend = np.mean(np.dot(1 / Dis_d, I_d), axis=0)
        LFPAIS = np.mean(np.dot(1 / Dis_a, I_A), axis=0)


        ####Compute currents from synapses
        # I_Syn_S = np.array([S_S[k] * pyrPPSI_s[k] for k in range(len(S_S))])
        # I_Syn_A = np.array([S_A[k] * pyrPPSI_a[k] for k in range(len(S_S))])
        # I_Syn_DE=np.zeros(np.shape(PPSE))
        # I_Syn_DI=np.zeros(np.shape(PPSE))
        #
        # for l in range(5):
        #     I_Syn_DE[l] = np.array([S_d[k] * PPSE[l][k] for k in range(len(S_S))])
        #     I_Syn_DI[l] = np.array([S_d[k] * PPSI[l][k] for k in range(len(S_S))])


        LFP_SynD_E = np.zeros(PPSE[0].shape[1])
        LFP_SynD_I = np.zeros(PPSE[0].shape[1])
        LFP_SynS_I = np.zeros(PPSE[0].shape[1])
        LFP_SynA_I = np.zeros(PPSE[0].shape[1])
        LFP_Syn_DPYR = np.zeros(PPSE[0].shape[1])

        for l in range(5):
            LFP_SynD_E +=  np.mean(1e-8*np.dot(1/ Dis_d, PPSE[l,:, :]),axis=0)
            LFP_SynD_I +=  np.mean(1e-8*np.dot(1/ Dis_d, PPSI[l,:, :]),axis=0)

        LFP_SynS_I += np.mean(1e-8*np.dot(1 / Dis_s_mid, pyrPPSI_s), axis=0)
        LFP_SynA_I += np.mean(1e-8*np.dot(1 / Dis_a, pyrPPSI_a), axis=0)
        LFP_Syn_DPYR += np.mean(1e-8*np.dot(1 / Dis_d, pyrPPSE_Dpyr), axis=0)



        LFP= -1*(LFPsoma+LFPdend+LFPAIS)+(LFP_SynD_E+LFP_SynD_I+LFP_SynS_I+LFP_SynA_I)
        LFP=LFP / (4 * np.pi * self.sigma*1e-3)

        return LFP



