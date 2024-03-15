import numpy as np
from numba.experimental import jitclass
from numba import boolean, int32, float64,uint8
# Type = 1 = Pyramidal cell for CA1

def get_Variable_Names():
    return  ['g_leak_s','g_leak_d','g_leak_a','g_Na_s','g_KDR_s','g_CaL_s','g_AHP_s','g_AHP_BK_s','g_m_s','g_KNa',
             'g_Na_d','g_KDR_d','g_KA_d','g_CaT_d','g_h_d','g_CaR_d','g_AHP_d','g_AHP_BK_d','g_m_d',
             'g_Na_a','g_KDR_a',
             'E_leak','E_Na','E_K','E_Ca','E_h',
             'g_AMPA','E_AMPA','g_GABA','E_GABA','g_NMDA','E_NMDA',
             'Cm','Cm_d','p_SD' , 'p_SA','g_c_SA','g_c_SD','PERCENT','Noise','tau_GABA_s','tau_GABA_d','tau_Ex']

spec = [
    ('Type' ,float64),
    ('Layer' ,float64),
    ('E_leak' ,float64),
    ('E_Na' ,float64),
    ('E_K' ,float64),
    ('E_Ca' ,float64),
    ('E_h' ,float64),
    ('g_KDR_s' ,float64),
    ('g_Na_s' ,float64),
    ('g_CaL_s' ,float64),
    ('g_AHP_s' ,float64),
    ('g_AHP_BK_s' ,float64),
    ('g_m_s' ,float64),
    ('g_KNa' ,float64),
    ('g_Na_d' ,float64),
    ('g_KDR_d' ,float64),
    ('g_KA_d' ,float64),
    ('g_CaT_d' ,float64),
    ('g_CaR_d' ,float64),
    ('g_h_d' ,float64),
    ('g_AHP_d' ,float64),
    ('g_AHP_BK_d' ,float64),
    ('g_m_d' ,float64),
    ('g_Na_a', float64),
    ('g_KDR_a', float64),
    ('g_AMPA' ,float64),
    ('g_NMDA' ,float64),
    ('g_GABA' ,float64),
    ('E_AMPA' ,float64),
    ('E_NMDA' ,float64),
    ('E_GABA' ,float64),
    ('g_c' ,float64),
    ('g_c_SA' ,float64),
    ('g_c_SD', float64),
    ('g_leak' ,float64),
    ('g_leak_s' ,float64),
    ('g_leak_d', float64),
    ('g_leak_a', float64),
    ('Cm' ,float64),
    ('Cm_d', float64),
    ('p_SD' ,float64),
    ('p_SA', float64),
    ('Noise' ,float64),
    ('noise',float64),
    ('I_soma_stim' ,float64),
    ('I_dend_stim' ,float64),
    ('PERCENT' ,float64),
    ('I_synSoma' ,float64),
    ('I_synDend' ,float64),
    ('I_synAis' , float64),
    ('I_AMPA_CA1' ,float64),
    ('I_GABA_CA1' ,float64),
    ('s_GABA_CA1' ,float64),
    ('F' ,float64),
    ('R' ,float64),
    ('T' ,float64),
    ('Q' ,float64),
    ('alpha_Na' ,float64),
    ('Rpump' ,float64),
    ('Na_eq' ,float64),
    ('A_s' ,float64),
    ('A_d' ,float64),
    ('Cdur' ,float64),
    ('deadTime' ,float64),
    ('Cmax' ,float64),
    ('preThresh' ,float64),
    ('alpha' ,float64),
    ('beta' ,float64),
    ('lastRelease' ,float64[:]),
    ('Rinf' ,float64[:]),
    ('Rtau' ,float64[:]),
    ('Ra' ,float64[:]),
    ('R0' ,float64[:]),
    ('R1' ,float64[:]),
    ('C' ,float64[:]),
    ('B' ,float64[:]),
    ('mg' ,float64),
    ('dt' ,float64),
    ('NbODEs' ,int32),
    ('NbODEs_s_AMPA' ,int32),
    ('NbODEs_s_GABA' ,int32),
    ('dydx' ,float64[:]),
    ('dydx1' ,float64[:]),
    ('dydx2' ,float64[:]),
    ('dydx3' ,float64[:]),
    ('y' ,float64[:]),
    ('yt' ,float64[:]),
    ('ds_AMPA1' ,float64[:]),
    ('ds_AMPA2' ,float64[:]),
    ('ds_AMPA3' ,float64[:]),
    ('s_AMPA' ,float64[:]),
    ('s_AMPAo' ,float64[:]),
    ('ds_GABA1' ,float64[:]),
    ('ds_GABA2' ,float64[:]),
    ('ds_GABA3' ,float64[:]),
    ('s_GABA' ,float64[:]),
    ('s_GABAo' ,float64[:]),
    ('fonc_Ca_s' ,float64),
    ('fonc_Ca_d' ,float64),
    ('fonc_ghk_s' ,float64),
    ('fonc_ghk_d' ,float64),
    ('h_inf_s' ,float64),
    ('m_inf_s' ,float64),
    ('m_KDR_inf_s' ,float64),
    ('h_inf_d' ,float64),
    ('m_inf_d' ,float64),
    ('m_KDR_inf_d' ,float64),
    ('m_KA_inf_d' ,float64),
    ('h_KA_inf_d' ,float64),
    ('tau_m_KA' ,float64),
    ('tau_h_KA' ,float64),
    ('m_CaT_inf_d' ,float64),
    ('h_CaT_inf_d' ,float64),
    ('m_CaR_inf_d' ,float64),
    ('h_CaR_inf_d' ,float64),
    ('tau_m_CaT' ,float64),
    ('tau_h_CaT' ,float64),
    ('m_CaL_inf_s' ,float64),
    ('tau_m_CaL' ,float64),
    ('m_h_inf_d' ,float64),
    ('tau_m_h' ,float64),
    ('m_AHP_inf_s' ,float64),
    ('tau_m_AHP_s' ,float64),
    ('m_AHP_inf_d' ,float64),
    ('tau_m_AHP_d' ,float64),
    ('m_AHP_BK_inf_s' ,float64),
    ('tau_m_AHP_BK_s' ,float64),
    ('m_AHP_BK_inf_d' ,float64),
    ('tau_m_AHP_BK_d' ,float64),
    ('m_m_inf_s' ,float64),
    ('tau_m_m_s' ,float64),
    ('m_m_inf_d' ,float64),
    ('tau_m_m_d' ,float64),
    ('I_Na_s' ,float64),
    ('I_Na_a', float64),
    ('I_KDR_s' ,float64),
    ('I_KDR_a', float64),
    ('I_Na_d' ,float64),
    ('I_KDR_d' ,float64),
    ('I_KA_d' ,float64),
    ('I_CaT_d' ,float64),
    ('I_CaR_d' ,float64),
    ('I_CaL_s' ,float64),
    ('I_h_d' ,float64),
    ('I_AHP_s' ,float64),
    ('I_AHP_d' ,float64),
    ('I_AHP_BK_s' ,float64),
    ('I_AHP_BK_d' ,float64),
    ('I_m_s' ,float64),
    ('I_m_d' ,float64),
    ('I_leak_s' ,float64),
    ('I_leak_d' ,float64),
    ('I_leak_a', float64),
    ('I_soma_stim' ,float64),
    ('I_dend_stim' ,float64),
    ('I_axon_stim', float64),
    ('drive_channel_d' ,float64),
    ('drive_channel_s' ,float64),
    ('I_KNa' ,float64),
    ('T_fonc' ,float64[:]),
    ('F_fonc' ,float64[:]),
    ('fifteenpoxer3' ,float64),
    ('xghk' ,float64),
    ('qtm_KA' ,float64),
    ('m_KDR_inf_a', float64),
    ('tauxm_KDR_inf_a', float64),
    ('m_Na_inf_a', float64),
    ('tauxm_Na_inf_a', float64),
    ('h_Na_inf_a', float64),
    ('tauxh_Na_inf_a', float64),
    ('tau_GABA_s',float64),
    ('tau_GABA_d', float64),
    ('tau_Ex',float64),

]

@jitclass(spec)
class pyrCellneo:
    ##soma and dendrite current
# I_Na: sodium currrent
# I_KDR: Delayed rectifier postassium current
# I_AHP: post-burst afterhyperpolarization potassium current (activated by intracellular calcium ions)
# I_m: muscarinic current
# I_leak: leak current
    ## soma
# I_CaL: Ltype Calcium current
    ## Dendrites
# I_CaT: T type calcium current
# I_CaR: R type calcium current
# I_KA: fast-inactivating A-type currents Potassim
# I_h: hyperpolarisation activated current

    def __init__(self,Layer):
        self.Type = 1
        self.Layer = Layer
        # parameter from GUI
        self.E_leak = -70
        self.E_Na = 50
        self.E_K = -95
        self.E_Ca = 125
        self.E_h = -10

        #Lévynoise

        # soma
        self.g_KDR_s = 6.
        self.g_Na_s = 70.
        self.g_CaL_s = 0.5
        self.g_AHP_s = 0.1
        self.g_AHP_BK_s = 2
        self.g_leak_s = 0.18
        ###
        if self.Layer in [3, 4]:
            self.g_m_s = 3.1
            self.g_KNa = 0
        else:
            self.g_KNa = 0.5
            self.g_m_s = 3.5

        # dendrites
        self.g_Na_d = 14
        self.g_KDR_d = 2
        self.g_KA_d = 55
        self.g_CaT_d = 1
        self.g_leak_d = 0.18

        if self.Layer in [3, 4]:
            self.g_CaR_d = 3
            self.g_AHP_d = 10
        else:
            self.g_CaR_d = 0.75
            self.g_AHP_d = 20

        self.g_h_d = 0.4
        self.g_AHP_BK_d = 1
        self.g_m_d = 0.1

        # AIS
        self.g_Na_a=200
        self.g_KDR_a=200
        self.g_leak_a = 0.18

        # synapses
        self.g_AMPA = 8#18
        self.g_NMDA = 0.15#0.35
        self.g_GABA = 25#18
        self.E_AMPA = 0.0
        self.E_NMDA = 0.0
        self.E_GABA = -75#-55

        self.g_c = 1
        self.g_c_SA = 1
        self.g_c_SD = 1
        self.Cm = 1
        self.Cm_d = 2

        self.p_SD = 0.15 #P Soma/dendrite
        self.p_SA = 0.95 #P Soma/axon
        self.g_leak = 0.18

        self.I_soma_stim = 0.0
        self.I_dend_stim = 0.0
        self.Noise = 0.05
        # self.noise=0.01
        self.PERCENT = 0.1
        self.I_synSoma = 0.0
        self.I_synDend = 0.0
        self.I_synAis = 0.0
        self.I_AMPA_CA1 = 0.0
        self.I_GABA_CA1 = 0.0
        self.s_GABA_CA1 = 0.0
        self.tau_GABA_s = 2.
        self.tau_GABA_d = 2.
        self.tau_Ex = 5.
        #internal parameters
        self.F = 96480
        self.R = 8.315
        self.T = 273.16
        self.Q = self.F / (self.R * self.T)
        self.alpha_Na = 0.01
        self.Rpump = 0.060  # * Publi: 0.018 * /
        self.Na_eq = 9.5
        self.A_s = 0.015
        self.A_d = 0.035

        #NMDA
        self.Cdur = 1.
        self.deadTime = 1.
        self.Cmax = 1.
        self.preThresh = 0.
        self.alpha = 0.072
        self.beta = 0.0066
        self.mg = 1.

        #ODE vectors
        self.dt = 1. / 2048.
        self.NbODEs = 29
        self.NbODEs_s_AMPA = 1
        self.NbODEs_s_GABA = 1
        self.init_vector()

        self.fifteenpoxer3 = np.power(15.0, 3.)
        self.xghk = (0.0853 * (34. + 273.16)) / 2.
        self.qtm_KA = np.exp(((37. - 24.) / 10.) * np.log(5.))

        self.setParameters()
        self.updateParameters()




    def init_vector(self):
        self.dydx = np.zeros(self.NbODEs, )
        self.dydx1 = np.zeros(self.NbODEs, )
        self.dydx2 = np.zeros(self.NbODEs, )
        self.dydx3 = np.zeros(self.NbODEs, )
        self.y = np.zeros(self.NbODEs, )
        self.yt = np.zeros(self.NbODEs, )

        self.ds_AMPA1 = np.zeros(self.NbODEs_s_AMPA, )
        self.ds_AMPA2 = np.zeros(self.NbODEs_s_AMPA, )
        self.ds_AMPA3 = np.zeros(self.NbODEs_s_AMPA, )
        self.s_AMPA = np.zeros(self.NbODEs_s_AMPA, )
        self.s_AMPAo = np.zeros(self.NbODEs_s_AMPA, )

        self.ds_GABA1 = np.zeros(self.NbODEs_s_GABA, )
        self.ds_GABA2 = np.zeros(self.NbODEs_s_GABA, )
        self.ds_GABA3 = np.zeros(self.NbODEs_s_GABA, )
        self.s_GABA = np.zeros(self.NbODEs_s_GABA, )
        self.s_GABAo = np.zeros(self.NbODEs_s_GABA, )

        self.I_soma_stim = 0.0
        self.I_dend_stim = 0.0
        self.I_axon_stim=0.0

        #init NMDA
        self.Cdur = 1.
        self.deadTime = 1.
        self.Cmax = 1.
        self.preThresh = 0.
        self.alpha = 0.072
        self.beta = 0.0066

        self.lastRelease = -1.e99 * np.ones(self.NbODEs_s_AMPA, )
        self.Rinf = 0. * np.ones(self.NbODEs_s_AMPA, )
        self.Rtau = 0. * np.ones(self.NbODEs_s_AMPA, )
        self.Ra = 0. * np.ones(self.NbODEs_s_AMPA, )
        self.R0 = 0. * np.ones(self.NbODEs_s_AMPA, )
        self.R1 = 0. * np.ones(self.NbODEs_s_AMPA, )
        self.C = 0. * np.ones(self.NbODEs_s_AMPA, )
        self.B = 0. * np.ones(self.NbODEs_s_AMPA, )
        self.mg = 1.





    #create aliases
    @property
    def Vs_(self):
        return self.y[0]

    @Vs_.setter
    def Vs_(self, value):
        self.y[0] = value

    @property
    def dVs_dt(self):
        return self.dydx[0]

    @dVs_dt.setter
    def dVs_dt(self, value):
        self.dydx[0] = value

    @property
    def Vd_(self):
        return self.y[1]

    @Vd_.setter
    def Vd_(self, value):
        self.y[1] = value

    @property
    def dVd_dt(self):
        return self.dydx[1]

    @dVd_dt.setter
    def dVd_dt(self, value):
        self.dydx[1] = value

    @property
    def m_s_(self):
        return self.y[4]

    @m_s_.setter
    def m_s_(self, value):
        self.y[4] = value

    @property
    def dm_s_dt(self):
        return self.dydx[4]

    @dm_s_dt.setter
    def dm_s_dt(self, value):
        self.dydx[4] = value

    @property
    def h_s_(self):
        return self.y[2]

    @h_s_.setter
    def h_s_(self, value):
        self.y[2] = value

    @property
    def dh_s_dt(self):
        return self.dydx[2]

    @dh_s_dt.setter
    def dh_s_dt(self, value):
        self.dydx[2] = value

    @property
    def m_KDR_s_(self):
        return self.y[3]

    @m_KDR_s_.setter
    def m_KDR_s_(self, value):
        self.y[3] = value

    @property
    def dm_KDR_s_dt(self):
        return self.dydx[3]

    @dm_KDR_s_dt.setter
    def dm_KDR_s_dt(self, value):
        self.dydx[3] = value

    @property
    def m_d_(self):
        return self.y[5]

    @m_d_.setter
    def m_d_(self, value):
        self.y[5] = value

    @property
    def dm_d_dt(self):
        return self.dydx[5]

    @dm_d_dt.setter
    def dm_d_dt(self, value):
        self.dydx[5] = value

    @property
    def h_d_(self):
        return self.y[6]

    @h_d_.setter
    def h_d_(self, value):
        self.y[6] = value

    @property
    def dh_d_dt(self):
        return self.dydx[6]

    @dh_d_dt.setter
    def dh_d_dt(self, value):
        self.dydx[6] = value

    @property
    def m_KDR_d_(self):
        return self.y[7]

    @m_KDR_d_.setter
    def m_KDR_d_(self, value):
        self.y[7] = value

    @property
    def dm_KDR_d_dt(self):
        return self.dydx[7]

    @dm_KDR_d_dt.setter
    def dm_KDR_d_dt(self, value):
        self.dydx[7] = value

    @property
    def m_KA_d_(self):
        return self.y[8]

    @m_KA_d_.setter
    def m_KA_d_(self, value):
        self.y[8] = value

    @property
    def dm_KA_d_dt(self):
        return self.dydx[8]

    @dm_KA_d_dt.setter
    def dm_KA_d_dt(self, value):
        self.dydx[8] = value

    @property
    def h_KA_d_(self):
        return self.y[9]

    @h_KA_d_.setter
    def h_KA_d_(self, value):
        self.y[9] = value

    @property
    def dh_KA_d_dt(self):
        return self.dydx[9]

    @dh_KA_d_dt.setter
    def dh_KA_d_dt(self, value):
        self.dydx[9] = value

    @property
    def m_CaT_d_(self):
        return self.y[10]

    @m_CaT_d_.setter
    def m_CaT_d_(self, value):
        self.y[10] = value

    @property
    def dm_CaT_d_dt(self):
        return self.dydx[10]

    @dm_CaT_d_dt.setter
    def dm_CaT_d_dt(self, value):
        self.dydx[10] = value

    @property
    def h_CaT_d_(self):
        return self.y[11]

    @h_CaT_d_.setter
    def h_CaT_d_(self, value):
        self.y[11] = value

    @property
    def dh_CaT_d_dt(self):
        return self.dydx[11]

    @dh_CaT_d_dt.setter
    def dh_CaT_d_dt(self, value):
        self.dydx[11] = value

    @property
    def Ca_s_(self):
        return self.y[12]

    @Ca_s_.setter
    def Ca_s_(self, value):
        self.y[12] = value

    @property
    def dCa_s_dt(self):
        return self.dydx[12]

    @dCa_s_dt.setter
    def dCa_s_dt(self, value):
        self.dydx[12] = value

    @property
    def m_CaL_s_(self):
        return self.y[13]

    @m_CaL_s_.setter
    def m_CaL_s_(self, value):
        self.y[13] = value

    @property
    def dm_CaL_s_dt(self):
        return self.dydx[13]

    @dm_CaL_s_dt.setter
    def dm_CaL_s_dt(self, value):
        self.dydx[13] = value

    @property
    def Ca_d_(self):
        return self.y[14]

    @Ca_d_.setter
    def Ca_d_(self, value):
        self.y[14] = value

    @property
    def dCa_d_dt(self):
        return self.dydx[14]

    @dCa_d_dt.setter
    def dCa_d_dt(self, value):
        self.dydx[14] = value

    @property
    def m_h_d_(self):
        return self.y[15]

    @m_h_d_.setter
    def m_h_d_(self, value):
        self.y[15] = value

    @property
    def dm_h_d_dt(self):
        return self.dydx[15]

    @dm_h_d_dt.setter
    def dm_h_d_dt(self, value):
        self.dydx[15] = value

    @property
    def m_CaR_d_(self):
        return self.y[16]

    @m_CaR_d_.setter
    def m_CaR_d_(self, value):
        self.y[16] = value

    @property
    def dm_CaR_d_dt(self):
        return self.dydx[16]

    @dm_CaR_d_dt.setter
    def dm_CaR_d_dt(self, value):
        self.dydx[16] = value

    @property
    def h_CaR_d_(self):
        return self.y[17]

    @h_CaR_d_.setter
    def h_CaR_d_(self, value):
        self.y[17] = value

    @property
    def dh_CaR_d_dt(self):
        return self.dydx[17]

    @dh_CaR_d_dt.setter
    def dh_CaR_d_dt(self, value):
        self.dydx[17] = value

    @property
    def m_AHP_s_(self):
        return self.y[18]

    @m_AHP_s_.setter
    def m_AHP_s_(self, value):
        self.y[18] = value

    @property
    def dm_AHP_s_dt(self):
        return self.dydx[18]

    @dm_AHP_s_dt.setter
    def dm_AHP_s_dt(self, value):
        self.dydx[18] = value

    @property
    def m_AHP_d_(self):
        return self.y[19]

    @m_AHP_d_.setter
    def m_AHP_d_(self, value):
        self.y[19] = value

    @property
    def dm_AHP_d_dt(self):
        return self.dydx[19]

    @dm_AHP_d_dt.setter
    def dm_AHP_d_dt(self, value):
        self.dydx[19] = value

    @property
    def m_AHP_BK_s_(self):
        return self.y[20]

    @m_AHP_BK_s_.setter
    def m_AHP_BK_s_(self, value):
        self.y[20] = value

    @property
    def dm_AHP_BK_s_dt(self):
        return self.dydx[20]

    @dm_AHP_BK_s_dt.setter
    def dm_AHP_BK_s_dt(self, value):
        self.dydx[20] = value

    @property
    def m_AHP_BK_d_(self):
        return self.y[21]

    @m_AHP_BK_d_.setter
    def m_AHP_BK_d_(self, value):
        self.y[21] = value

    @property
    def dm_AHP_BK_d_dt(self):
        return self.dydx[21]

    @dm_AHP_BK_d_dt.setter
    def dm_AHP_BK_d_dt(self, value):
        self.dydx[21] = value

    @property
    def m_m_s_(self):
        return self.y[22]

    @m_m_s_.setter
    def m_m_s_(self, value):
        self.y[22] = value

    @property
    def dm_m_s_dt(self):
        return self.dydx[22]

    @dm_m_s_dt.setter
    def dm_m_s_dt(self, value):
        self.dydx[22] = value

    @property
    def m_m_d_(self):
        return self.y[23]

    @m_m_d_.setter
    def m_m_d_(self, value):
        self.y[23] = value

    @property
    def dm_m_d_dt(self):
        return self.dydx[23]

    @dm_m_d_dt.setter
    def dm_m_d_dt(self, value):
        self.dydx[23] = value

    @property
    def Na_intra_(self):
        return self.y[24]

    @Na_intra_.setter
    def Na_intra_(self, value):
        self.y[24] = value

    @property
    def dNa_intra_dt(self):
        return self.dydx[24]

    @dNa_intra_dt.setter
    def dNa_intra_dt(self, value):
        self.dydx[24] = value

    @property
    def m_KDR_a_(self):
        return self.y[25]

    @m_KDR_a_.setter
    def m_KDR_a_(self, value):
        self.y[25] = value

    @property
    def dm_KDR_a_dt(self):
        return self.dydx[25]

    @dm_KDR_a_dt.setter
    def dm_KDR_a_dt(self, value):
        self.dydx[25] = value


    @property
    def m_Na_a_(self):
        return self.y[26]

    @m_Na_a_.setter
    def m_Na_a_(self, value):
        self.y[26] = value

    @property
    def dm_Na_a_dt(self):
        return self.dydx[26]

    @dm_Na_a_dt.setter
    def dm_Na_a_dt(self, value):
        self.dydx[26] = value


    @property
    def h_Na_a_(self):
        return self.y[27]

    @h_Na_a_.setter
    def h_Na_a_(self, value):
        self.y[27] = value

    @property
    def dh_Na_a_dt(self):
        return self.dydx[27]

    @dh_Na_a_dt.setter
    def dh_Na_a_dt(self, value):
        self.dydx[27] = value

    @property
    def Va_(self):
        return self.y[28]

    @Va_.setter
    def Va_(self, value):
        self.y[28] = value

    @property
    def dVa_dt(self):
        return self.dydx[28]

    @dVa_dt.setter
    def dVa_dt(self, value):
        self.dydx[28] = value


    def setParameters(self):
        #[self applyInterfaceParameters:self];

        self.y[0] = self.E_leak
        self.y[1] = self.E_leak
        self.y[28] = self.E_leak
        self.y[2] = self.h_inf_CA1(self.y[0])#alpha_h_CA1(self.y[0])/(alpha_h_CA1(self.y[0])+beta_h_CA1(self.y[0]));
        self.y[3] = self.m_KDR_inf_CA1(self.y[0])#alpha_n_CA1(self.y[0])/(alpha_n_CA1(self.y[0])+beta_n_CA1(self.y[0]));
        self.y[4] = self.m_inf_CA1(self.y[0])
        self.y[5] = self.m_inf_CA1(self.y[1])
        self.y[6] = self.h_inf_CA1(self.y[1])
        self.y[7] = self.m_KDR_inf_CA1(self.y[1])
        self.y[8] = self.m_KA_inf_CA1(self.y[1])
        self.y[9] = self.h_KA_inf_CA1(self.y[1])
        self.y[12] = 0.00005
        self.y[14] = 0.00005
        self.y[10] = self.m_CaT_inf_CA1(self.y[1])
        self.y[11] = self.h_CaT_inf_CA1(self.y[1])
        self.y[13] = self.m_CaL_inf_CA1(self.y[1])
        self.y[15] = self.m_h_inf_CA1(self.y[1])
        self.y[16] = self.m_CaR_inf_CA1(self.y[1])
        self.y[17] = self.h_CaR_inf_CA1(self.y[1])
        self.y[18] = self.m_AHP_inf_CA1(self.y[12])
        self.y[19] = self.m_AHP_inf_CA1(self.y[14])
        self.y[20] = self.m_AHP_BK_inf_CA1(self.y[0],self.y[12])
        self.y[21] = self.m_AHP_BK_inf_CA1(self.y[1],self.y[14])
        self.y[22] = self.m_m_inf_CA1(self.y[0])
        self.y[23] = self.m_m_inf_CA1(self.y[1])
        self.y[24] = 9.5
        self.y[25]=self.m_KDR_inf_axon(self.y[28])
        self.y[26]=self.m_Na_inf_axon(self.y[28])
        self.y[27]=self.h_Na_inf_axon(self.y[28])
    def seednoise(self,seed):
        if seed:
            np.random.seed(seed)
    # def bruitGaussien(self,m, s):
    #     return np.random.normal(m, s)



    def bruitGaussien(self,m, s):
        return np.random.normal(m, s)

    def I_leak_CA1(self,g_leak, PERCENT, V, E_leak):
        return self.bruitGaussien(g_leak,PERCENT * g_leak) * (V - E_leak)
    # def I_leak_CA1(self,g_leak, V, E_leak):
    #     return g_leak * (V - E_leak)

    def updateParameters(self):
        self.fonc_Ca_s = self.fonc_Ca_CA1(self.y[12])
        self.fonc_Ca_d = self.fonc_Ca_CA1(self.y[14])
        self.fonc_ghk_s = self.fonc_ghk_CA1(self.y[0],self.y[12])
        self.fonc_ghk_d = self.fonc_ghk_CA1(self.y[1],self.y[14])
        self.h_inf_s = self.h_inf_CA1(self.y[0])
        self.m_inf_s = self.m_inf_CA1(self.y[0])
        self.m_KDR_inf_s = self.m_KDR_inf_CA1(self.y[0])
        self.h_inf_d = self.h_inf_CA1(self.y[1])
        self.m_inf_d = self.m_inf_CA1(self.y[1])
        self.m_KDR_inf_d = self.m_KDR_inf_CA1(self.y[1])
        self.m_KA_inf_d = self.m_KA_inf_CA1(self.y[1])
        self.h_KA_inf_d = self.h_KA_inf_CA1(self.y[1])
        self.tau_m_KA = self.tau_m_KA_CA1(self.y[1])
        self.tau_h_KA = self.tau_h_KA_CA1(self.y[1])
        self.m_CaT_inf_d = self.m_CaT_inf_CA1(self.y[1])
        self.h_CaT_inf_d = self.h_CaT_inf_CA1(self.y[1])
        self.m_CaR_inf_d = self.m_CaR_inf_CA1(self.y[1])
        self.h_CaR_inf_d = self.h_CaR_inf_CA1(self.y[1])
        self.tau_m_CaT = self.tau_m_CaT_CA1(self.y[1])
        self.tau_h_CaT = self.tau_h_CaT_CA1(self.y[1])
        self.m_CaL_inf_s = self.m_CaL_inf_CA1(self.y[1])
        self.tau_m_CaL = self.tau_m_CaL_CA1(self.y[1])
        self.m_h_inf_d = self.m_h_inf_CA1(self.y[1])
        self.tau_m_h = self.tau_m_h_CA1(self.y[1])
        self.m_AHP_inf_s = self.m_AHP_inf_CA1(self.y[12])
        self.tau_m_AHP_s = self.tau_m_AHP_CA1(self.y[12])
        self.m_AHP_inf_d = self.m_AHP_inf_CA1(self.y[14])
        self.tau_m_AHP_d = self.tau_m_AHP_CA1(self.y[14])
        self.m_AHP_BK_inf_s = self.m_AHP_BK_inf_CA1(self.y[0],self.y[12])
        self.tau_m_AHP_BK_s = self.tau_m_AHP_BK_CA1(self.y[0],self.y[12])
        self.m_AHP_BK_inf_d = self.m_AHP_BK_inf_CA1(self.y[1],self.y[14])
        #printf("-->%f %f %f %f\n",self.m_AHP_BK_inf_s ,self.m_AHP_BK_inf_d , self.y[1], self.y[12])
        #printf("self.m_AHP_BK_inf_d:%f\n",self.m_AHP_BK_inf_d)
        self.tau_m_AHP_BK_d = self.tau_m_AHP_BK_CA1(self.y[1],self.y[14])
        self.m_m_inf_s = self.m_m_inf_CA1(self.y[0])
        self.tau_m_m_s = self.tau_m_m_CA1(self.y[0])
        self.m_m_inf_d = self.m_m_inf_CA1(self.y[1])
        self.tau_m_m_d = self.tau_m_m_CA1(self.y[1])

        self.I_Na_s = self.I_Na_CA1(self.g_Na_s, self.y[4], self.y[2], self.y[0], self.E_Na)
        self.I_KDR_s = self.I_KDR_CA1(self.g_KDR_s, self.y[3], self.y[0], self.E_K)
        self.I_Na_d = self.I_Na_CA1(self.g_Na_d, self.y[5], self.y[6], self.y[1], self.E_Na)
        self.I_KDR_d = self.I_KDR_CA1(self.g_KDR_d, self.y[7], self.y[1], self.E_K)
        self.I_KA_d = self.I_KA_CA1(self.g_KA_d, self.y[8], self.y[9], self.y[1], self.E_K)
        self.I_CaT_d = self.I_CaT_CA1(self.g_CaT_d,self.y[10], self.y[11], self.fonc_Ca_s, self.fonc_ghk_s, self.y[1], self.E_Ca)
        self.I_CaR_d = self.I_CaR_CA1(self.g_CaR_d,self.y[16], self.y[17], self.y[1], self.E_Ca)
        self.I_CaL_s = self.I_CaL_CA1(self.g_CaL_s,self.y[13], self.fonc_Ca_d, self.fonc_ghk_d, self.y[1], self.E_Ca)
        self.I_h_d = self.I_h_CA1(self.g_h_d, self.y[15], self.y[1], self.E_h)
        self.I_AHP_s = self.I_AHP_CA1(self.g_AHP_s, self.y[18], self.y[0], self.E_K)
        self.I_AHP_d = self.I_AHP_CA1(self.g_AHP_d, self.y[19], self.y[1], self.E_K)
        self.I_AHP_BK_s = self.I_AHP_BK_CA1(self.g_AHP_BK_s, self.y[20], self.y[0], self.E_K)
        self.I_AHP_BK_d = self.I_AHP_BK_CA1(self.g_AHP_BK_d, self.y[21], self.y[1], self.E_K)
        self.I_m_s = self.I_m_CA1(self.g_m_s, self.y[22], self.y[0], self.E_K)
        self.I_m_d = self.I_m_CA1(self.g_m_d, self.y[23], self.y[1], self.E_K)
        self.I_leak_s = self.I_leak_CA1(self.g_leak_s, self.PERCENT, self.y[0], self.E_leak)
        self.I_leak_d = self.I_leak_CA1(self.g_leak_d, self.PERCENT, self.y[1], self.E_leak)
        #self.I_soma_stim = [self getI_soma_stim]
        #self.I_dend_stim = [self getI_dend_stim]
        self.drive_channel_d = self.drive_channel_CA1((self.I_CaT_d) + (self.I_CaR_d))
        self.drive_channel_s = self.drive_channel_CA1(self.I_CaL_s)

        self.I_KNa = self.I_KNa_CA1(self.g_KNa, self.y[24], self.y[0], self.E_K)

        self.m_KDR_inf_a=self.m_KDR_inf_axon(self.y[28])
        # print('mKDR',self.m_KDR_inf_a)
        self.tauxm_KDR_inf_a=self.tauxm_KDR_inf_axon(self.y[28])
        # print('tauxmKDR',self.tauxm_KDR_inf_a)

        self.m_Na_inf_a=self.m_Na_inf_axon(self.y[28])
        self.tauxm_Na_inf_a=self.tauxm_Na_inf_axon(self.y[28])

        self.h_Na_inf_a=self.h_Na_inf_axon(self.y[28])
        self.tauxh_Na_inf_a=self.tauxh_Na_inf_axon(self.y[28])

        self.I_KDR_a=self.I_KDR_axon(self.g_KDR_a,self.y[25],self.y[28],self.E_K)
        self.I_Na_a=self.I_Na_axon(self.g_Na_a,self.y[26],self.y[27],self.y[28],self.E_Na)
        self.I_leak_a = self.I_leak_CA1(self.g_leak_a, self.PERCENT, self.y[28], self.E_leak)

    def rk4(self):
        self.yt = self.y+0. #y origine at t
        self.dydx1=self.derivT()#K1
        self.y = self.yt + self.dydx1 * self.dt / 2
        self.dydx2=self.derivT()#K2
        self.y = self.yt + self.dydx2 * self.dt / 2
        self.dydx3=self.derivT()#K3
        self.y = self.yt + self.dydx3 * self.dt
        self.derivT() #K4
        self.y =self.yt + self.dt/6. *(self.dydx1+2*self.dydx2+2*self.dydx3+self.dydx) #y at t+dt

    def Euler(self):
        self.yt = self.y+0. #y origine at t
        self.dydx=self.derivT()#K1
        self.y = self.yt + self.dydx * self.dt+np.sqrt(self.dt)*self.bruitGaussien(0,1)*self.yt*self.Noise

    def derivT(self):

        self.dVs_dt = 1. / self.Cm * (-self.I_KNa - self.I_Na_s - self.I_KDR_s - self.I_CaL_s - self.I_AHP_s - self.I_AHP_BK_s - self.I_m_s - self.I_leak_s - self.I_synSoma - (self.g_c / self.p_SD * (self.Vs_- self.Vd_)) - (self.g_c / self.p_SA * (self.Vs_- self.Va_)) + self.I_soma_stim)
        self.dVd_dt = 1. / self.Cm_d * (-self.I_Na_d - self.I_KDR_d - self.I_KA_d - self.I_CaT_d - self.I_CaR_d - self.I_h_d - self.I_AHP_d - self.I_AHP_BK_d - self.I_m_d - self.I_leak_d - self.I_synDend - (self.g_c / (1 - self.p_SD) * (-self.Vs_ + self.Vd_)) + self.I_dend_stim)
        self.dVa_dt = 1. / self.Cm * (-self.I_Na_a - self.I_KDR_a -self.I_leak_a  - self.I_synAis - (self.g_c /(1-self.p_SA) * (self.Va_- self.Vs_)) + self.I_axon_stim)
        self.dh_s_dt = ((self.h_inf_s - self.h_s_) / (0.5))
        self.dm_s_dt = ((self.m_inf_s - self.m_s_) / (0.05))
        self.dm_KDR_s_dt = ((self.m_KDR_inf_s - self.m_KDR_s_) / (2.2))
        self.dh_d_dt = ((self.h_inf_d - self.h_d_) / (0.5))
        self.dm_d_dt = ((self.m_inf_d - self.m_d_) / (0.05))
        self.dm_KDR_d_dt = ((self.m_KDR_inf_d - self.m_KDR_d_) / (2.2))
        if not self.tau_m_KA == 0:
            self.dm_KA_d_dt = ((self.m_KA_inf_d - self.m_KA_d_) / (self.tau_m_KA))
        if not self.tau_h_KA == 0:
            self.dh_KA_d_dt = ((self.h_KA_inf_d - self.h_KA_d_) / (self.tau_h_KA))
        self.dCa_s_dt = self.drive_channel_s + ((0.0001 - self.Ca_s_) / 200.) * 1.
        self.dCa_d_dt = self.drive_channel_d + ((0.0001 - self.Ca_d_) / 200.) * 1.
        if not self.tau_m_CaT == 0:
            self.dm_CaT_d_dt = ((self.m_CaT_inf_d - self.m_CaT_d_) / (self.tau_m_CaT))
        if not self.tau_h_CaT == 0:
            self.dh_CaT_d_dt = ((self.h_CaT_inf_d - self.h_CaT_d_) / (self.tau_h_CaT))
        self.dm_CaR_d_dt = ((self.m_CaR_inf_d - self.m_CaR_d_) / (50.))
        self.dh_CaR_d_dt = ((self.h_CaR_inf_d - self.h_CaR_d_) / (5.))
        if not self.tau_m_CaL == 0:
            self.dm_CaL_s_dt = ((self.m_CaL_inf_s - self.m_CaL_s_) / (self.tau_m_CaL))
        if not self.tau_m_h == 0:
            self.dm_h_d_dt = ((self.m_h_inf_d - self.m_h_d_) / (self.tau_m_h))
        if not self.tau_m_AHP_s == 0:
            self.dm_AHP_s_dt = ((self.m_AHP_inf_s - self.m_AHP_s_) / (self.tau_m_AHP_s))
        if not self.tau_m_AHP_d == 0:
            self.dm_AHP_d_dt = ((self.m_AHP_inf_d - self.m_AHP_d_) / (self.tau_m_AHP_d))
        if not self.tau_m_AHP_BK_s == 0:
            self.dm_AHP_BK_s_dt = ((self.m_AHP_BK_inf_s - self.m_AHP_BK_s_) / (self.tau_m_AHP_BK_s))
        if not self.tau_m_AHP_BK_d == 0:
            self.dm_AHP_BK_d_dt = ((self.m_AHP_BK_inf_d - self.m_AHP_BK_d_) / (self.tau_m_AHP_BK_d))
        # printf("d:%g %g %g %g\n", dm_AHP_BK_d_dt, m_AHP_BK_inf_d, m_AHP_BK_d_, tau_m_AHP_BK_d)
        if not self.tau_m_m_s == 0:
            self.dm_m_s_dt = ((self.m_m_inf_s - self.m_m_s_) / (self.tau_m_m_s))
        if not self.tau_m_m_d == 0:
            self.dm_m_d_dt = ((self.m_m_inf_d - self.m_m_d_) / (self.tau_m_m_d))
        Na_intra_power3 = self.Na_intra_ * self.Na_intra_ * self.Na_intra_
        self.dm_Na_a_dt=(self.m_Na_inf_a - self.m_Na_a_)/self.tauxm_Na_inf_a
        self.dh_Na_a_dt=(self.h_Na_inf_a - self.h_Na_a_)/self.tauxh_Na_inf_a
        self.dm_KDR_a_dt = ((self.m_KDR_inf_a - self.m_KDR_a_) / self.tauxm_KDR_inf_a)

        Na_eq_power3 = self.Na_eq * self.Na_eq * self.Na_eq
        self.dNa_intra_dt = -self.alpha_Na * (self.A_s * (self.I_Na_s) ) - self.Rpump * (Na_intra_power3 / (Na_intra_power3 + self.fifteenpoxer3) - (Na_eq_power3 / (Na_eq_power3 + self.fifteenpoxer3)))
        # self.dNa_intra_dt = -self.alpha_Na * (self.A_s * (self.I_Na_s) / * + self.A_d * self.I_NaP * /) - self.Rpump * (np.power(self.Na_intra_, 3.) / (np.power(self.Na_intra_, 3) + np.power(15.0, 3.)) - (np.power(self.Na_eq, 3.) / (np.power(self.Na_eq, 3.) + np.power(15., 3.))))

        return self.dydx + 0.

    #---------Gate Steady state and time constant equations---------
    def m_inf_CA1(self,V):
        return 1./(1.+np.exp(-(V+40.)/3.))

    def h_inf_CA1(self,V):
        return 1./(1.+np.exp((V+45.)/3.))

    def m_KDR_inf_CA1(self,V):
        return 1./(1.+np.exp(-(V+42.)/2.))

    def foncMax_CA1(self,v1,v2):
        if(v1<v2):
            return v2
        else:
            return v1

    def zeta(self,V):
        return  -1.8 - (1./(1. + np.exp((V+40.)/5.)))

    def alpha_m_KA_CA1(self,V):
        # self.Q = self.F/(self.R*(self.T+37.))
        return np.exp(0.001 * self.zeta(V) * (V+1.) * self.F/(self.R*(self.T+37.)))

    def m_KA_inf_CA1(self, V):
        return (1./(1. + self.alpha_m_KA_CA1(V)))

    def h_KA_inf_CA1(self, V):
        # self.Q = self.F/(self.R*(self.T+37.))
        alpha = np.exp(0.003 * (V+56.) * self.F/(self.R*(self.T+37.)))
        return (1./(1. + alpha))

    def tau_m_KA_CA1(self, V):
        # self.Q = self.F/(self.R*(self.T+37.))
        beta = np.exp(0.00039 * self.zeta(V) * (V+1.) * self.F/(self.R*(self.T+37.)))
        # qt = np.exp(((37.-24.)/10.)*np.log(5.))
        v1 = beta / (0.1 * self.qtm_KA * (1+ self.alpha_m_KA_CA1(V)))
        return self.foncMax_CA1(v1,0.1)

    def tau_h_KA_CA1(self, V):
        return self.foncMax_CA1((0.26*(V+50)),2.)

    def alpha_m_CaT_CA1(self, V):
        n = -0.196 * (V - 19.88)
        d = np.exp(-(V - 19.88)/10.) - 1.
        return n/d

    def beta_m_CaT_CA1(self, V):
        return 0.046 * np.exp(-V/22.73)

    def alpha_h_CaT_CA1(self, V):
        return 0.00016 * np.exp(-(V+57.)/19.)

    def beta_h_CaT_CA1(self, V):
        n = 1.
        d = np.exp(-(V-15.)/10.) + 1.
        return n/d

    def m_CaT_inf_CA1(self, V):
        return self.alpha_m_CaT_CA1(V)/(self.alpha_m_CaT_CA1(V) + self.beta_m_CaT_CA1(V))

    def h_CaT_inf_CA1(self, V):
        return self.alpha_h_CaT_CA1(V)/(self.alpha_h_CaT_CA1(V) + self.beta_h_CaT_CA1(V))

    def m_CaR_inf_CA1(self, V):
        return 1./(1.+np.exp(-(V+48.5)/3.))

    def h_CaR_inf_CA1(self, V):
        return 1./(1.+np.exp(V+53.))

    def tau_m_CaT_CA1(self, V):
        return 1./(self.alpha_m_CaT_CA1(V) + self.beta_m_CaT_CA1(V))

    def tau_h_CaT_CA1(self, V):
        return 1./(0.68 * (self.alpha_h_CaT_CA1(V) + self.beta_h_CaT_CA1(V)))

    def alpha_m_CaL_CA1(self, V):
        n = -0.055 * (V + 27.01)
        d = np.exp(-(V + 27.01)/3.8) - 1.
        return n/d

    def beta_m_CaL_CA1(self, V):
        return 0.94 * np.exp(-(V+63.01)/17.)

    def m_CaL_inf_CA1(self, V):
        alpha_m_CaL_CA1 = self.alpha_m_CaL_CA1(V)
        return alpha_m_CaL_CA1/(alpha_m_CaL_CA1 + self.beta_m_CaL_CA1(V))

    def tau_m_CaL_CA1(self, V):
        return 1./(5*(self.alpha_m_CaL_CA1(V) + self.beta_m_CaL_CA1(V)))

    def drive_channel_CA1(self, ICa):
        result = -10000.*ICa/(0.2*96480.)
        if(result>0.):
            return result * 1.e-3  # mA
        else:
            return 0.0

    def fonc_Ca_CA1(self, Ca):
        return 0.001/(0.001+Ca)

    def fonc_f_CA1(self, z):
        if(np.abs(z)<0.0001):
            return 1.-(z/2.)
        else:
            return z/(np.exp(z)-1.)

    def fonc_ghk_CA1(self, V, Cai):
        # x = (0.0853*(34.+273.16))/2.
        nu = V/self.xghk
        return -self.xghk*(1.-(Cai/2.)*np.exp(nu))*self.fonc_f_CA1(nu)

    def m_h_inf_CA1(self, V):
        return (1. - (1. / (1. + np.exp(-(V+90.)/8.5))))

    def tau_m_h_CA1(self, V):
        if(V>(-30.)):
            return 1.
        else:
            n = 2.
            d = np.exp(-((V+145.)/17.5))+np.exp((V+16.8)/16.5)
            return (n/d)+10.

    def fonc_Cac_CA1(self, Ca_in):
        return ((Ca_in/0.025)*(Ca_in/0.025))

    def m_AHP_inf_CA1(self, Ca_in):
        return self.fonc_Cac_CA1(Ca_in)/(1.+ self.fonc_Cac_CA1(Ca_in))

    def tau_m_AHP_CA1(self, Ca_in):
        v1 = 1./(0.003*(1.+self.fonc_Cac_CA1(Ca_in))*np.power(3.,(34.-22.)/10.))
        return self.foncMax_CA1(v1,0.5)

    def m_AHP_BK_inf_CA1(self, V, Ca_in):
        self.Q = self.F/(self.R*(self.T+34.))
        if Ca_in ==0:
            alpha = 0
        elif (1.+(0.18/Ca_in)*np.exp(-1.68*V*self.Q)) ==0.:
            alpha = 1e20
        else:
            alpha = 0.48/(1.+(0.18/Ca_in)*np.exp(-1.68*V*self.Q))
        if (0.011*np.exp(-2.*V*self.Q)) == 0. :
            beta=0
        elif (1.+(Ca_in/(0.011*np.exp(-2.*V*self.Q))))==0.:
            beta = 0
        else:
            beta = 0.28/(1.+(Ca_in/(0.011*np.exp(-2.*V*self.Q))))
        tau = 1./(alpha+beta)
        return alpha*tau

    def tau_m_AHP_BK_CA1(self, V, Ca_in):
        self.Q = self.F/(self.R*(self.T+34.))
        if Ca_in == 0:
            alpha = 0
        elif (1.+(0.18/Ca_in)*np.exp(-1.68*V*self.Q))==0:
            alpha = 1e20
        else:
            alpha = 0.48/(1.+(0.18/Ca_in)*np.exp(-1.68*V*self.Q))
        if (0.011*np.exp(-2.*V*self.Q)) == 0.:
            beta = 0
        elif (1.+(Ca_in/(0.011*np.exp(-2.*V*self.Q))))==0.:
            beta = 0
        else:
            beta = 0.28/(1.+(Ca_in/(0.011*np.exp(-2.*V*self.Q))))
        return 1./(alpha+beta)

    def m_m_inf_CA1(self, V):
        #alpha = 0.001*(V+30.)/(1-np.exp(-(V+30.)/9.));
        #beta = -0.001*(V+30.)/(1-np.exp((V+30.)/9.));
        alpha = 0.016*(np.exp((V+52.7)/23.))
        beta = 0.016*(np.exp(-(V+52.7)/18.8))
        tau = 1./(alpha+beta)
        return alpha*tau

    def tau_m_m_CA1(self, V):
        #alpha = 0.001*(V+30.)/(1-np.exp(-(V+30.)/9.));
        #beta = -0.001*(V+30.)/(1-np.exp((V+30.)/9.));
        alpha = 0.016*(np.exp((V+52.7)/23.))
        beta = 0.016*(np.exp(-(V+52.7)/18.8))
        return 1./(alpha+beta)

    def m_Na_inf_axon(self,V):
        # Alpha=0.8*(17.2-V)/(np.exp((17.2-V)/4)-1)
        # Beta=0.7*(V-42.2)/(np.exp((V-42.2)/5)-1)
        # return Alpha/(Alpha+Beta)
        return 1/(1+np.exp((-V-34.5)/10))

    def tauxm_Na_inf_axon(self, V):
        # Alpha = 0.8 * (17.2 - V) / (np.exp((17.2 - V) / 4) - 1)
        # Beta = 0.7 * (V - 42.2) / (np.exp((V - 42.2) / 5) - 1)
        # return 1. / (Alpha + Beta)
        if V<=-26.5:
            taux=0.025+0.14*np.exp((V+26.5)/10)
        else:
            taux=0.02+0.145*np.exp((-V-26.5)/10)

        return taux

    def h_Na_inf_axon(self,V):
        # Alpha=0.32*np.exp((42-V)/18)
        # Beta=10/(np.exp((42-V)/5)+1)
        # return Alpha/(Alpha+Beta)
        return 1/(1+np.exp((V+59.4)/10.7))

    def tauxh_Na_inf_axon(self, V):
        # Alpha=0.32*np.exp((42-V)/18)
        # Beta=10/(np.exp((42-V)/5)+1)
        # return 1. / (Alpha + Beta)
        return 0.15+1.15/(1+np.exp((V+33.5)/15))


    def m_KDR_inf_axon(self,V):
        # Alpha=0.03*(17.2-V)/(np.exp((17.2-V)/5)-1)
        # Beta=0.45*((12-V)/40)
        # return Alpha/(Alpha+Beta)
        # print(V)
        return 1/(1+np.exp((-V-29.5)/10))

    def tauxm_KDR_inf_axon(self, V):
        # Alpha=0.03*(17.2-V)/(np.exp((17.2-V)/5)-1)
        # Beta=0.45*((12-V)/40)
        # return 1. / (Alpha + Beta)
        if V<=-10:
            taux=0.25+4.35*np.exp((V+10)/10)
        else:
            taux=0.25+4.35*np.exp((-V-10)/10)

        return taux



    #--------------------  COURANT -----------------------
    def I_Na_CA1(self, g_Na, m_inf, h, Vs, E_Na):
        return g_Na*m_inf*m_inf*h*(Vs-E_Na)

    def I_KNa_CA1(self, g_KNa, Na_intra, Vs, E_K):#I_KNa in the original objectiveC code
        w = 0.37 / (1.0 + (38.7/ np.power(Na_intra,3.5) ) )
        return g_KNa * w * (Vs-E_K)

    def I_KDR_CA1(self, g_KDR, m, V, E_K):
        return g_KDR * m * m * (V-E_K)

    def I_KA_CA1(self, g_KA, m, h, V, E_K):
        return g_KA * m * h * (V-E_K)

    def I_CaT_CA1(self, g_CaT, m_CaT, h_CaT, fonc_Ca, fonc_ghk, V, E_Ca):
        return g_CaT * m_CaT * m_CaT * h_CaT * fonc_Ca * fonc_ghk # * (V-E_Ca)

    def I_CaR_CA1(self, g_CaR, m_CaR, h_CaR, V, E_Ca):
        return g_CaR * m_CaR * m_CaR * m_CaR * h_CaR * (V-E_Ca)

    def I_CaL_CA1(self, g_CaL, m_CaL, fonc_Ca, fonc_ghk, V, E_Ca):
        return g_CaL * m_CaL * fonc_Ca * fonc_ghk # * (V-E_Ca)

    def I_h_CA1(self, g_h, m, V, E_h):
        return g_h * m * (V-E_h)

    def I_AHP_CA1(self, g_AHP, m_AHP, V, E_K):
        return g_AHP * m_AHP * m_AHP * m_AHP * (V-E_K)

    def I_AHP_BK_CA1(self, g_AHP, m_AHP, V, E_K):
        #printf("--> %f %f %f %f %f\n",g_AHP, m_AHP,V, E_K, g_AHP * m_AHP * (V-E_K));
        return g_AHP * m_AHP * (V-E_K)

    def I_m_CA1(self, g_m, m_m, V, E_K):
        #return 0.0001 * g_m * m_m * (V-E_K);
        return g_m * m_m  * m_m* (V-E_K)

    def I_Na_axon(self,g_Na,m,h,V,E_Na):
        return g_Na*m*m*m*h*(V-E_Na)

    def I_KDR_axon(self,g_KDR,m,V,E_K):
        return g_KDR*m*m*m*m*(V-E_K)
    # -----------------------------------------------------

    def I_AMPA(self,Vd):
        return ((self.g_AMPA * 0.1) / 3.32) * self.s_AMPA * (Vd - self.E_AMPA)

    def I_AMPA2(self,Vpre):
        self.computeI_AMPA(Vpre)
        Vd = np.array([self.VDend() for i in Vpre])
        # Vd = np.array([self.VSoma() if s_d == 1 else self.VDend() for s_d in PreSynaptic_Soma_Dend_AMPA])
        return ((self.g_AMPA * 0.1) / 3.32) * self.s_AMPA * (Vd - self.E_AMPA)

    def I_AMPA_PPSE(self,Vpre):
        # if (self.VDend() - self.E_AMPA) == 0:
        #     return I_AMPA / 0.0000001
        # return I_AMPA / np.abs((self.VDend() - self.E_AMPA))
        return ((self.g_AMPA * 0.1) / 3.32) * self.s_AMPA* (self.E_leak - self.E_AMPA)

    def I_GABA(self,Vd):
        return ((self.g_GABA * 0.1) / 3.32) * self.s_GABA * (Vd - self.E_GABA)

    def I_GABA2(self,Vpre,PreSynaptic_Soma_GABA_d,PreSynaptic_Soma_GABA_s,PreSynaptic_Soma_GABA_a):
        self.computeI_GABA(Vpre,PreSynaptic_Soma_GABA_d)
        Vd = Vpre
        for i in range(len(Vpre)):
            if PreSynaptic_Soma_GABA_d[i]:
                Vd[i]=self.VDend()
            elif PreSynaptic_Soma_GABA_s[i]:
                Vd[i]=self.VSoma()
            elif PreSynaptic_Soma_GABA_a[i]:
                Vd[i]=self.VAis()
        # Vd = np.array([self.VSoma() if s_d == 1 else self.VDend() for s_d in PreSynaptic_Soma_Dend_GABA])
        return ((self.g_GABA * 0.1) / 3.32) * self.s_GABA * (Vd - self.E_GABA)

    def I_GABA_PPSI(self, I_GABA):
        # return I_GABA / np.abs(self.VDend() - self.E_GABA)
        return ((self.g_GABA * 0.1) / 3.32) * self.s_GABA* (self.E_leak - self.E_GABA)

    def I_NMDA2(self,Vpre,t):
        Vd = np.array([self.VDend() for i in Vpre])
        # Vd = np.array([self.VSoma() if s_d == 1 else self.VDend() for s_d in PreSynaptic_Soma_Dend_AMPA])
        # return np.array([self.computeI_NMDA(Vpre[k], t, Vd[k]) for k in range(len(Vpre))])
        return self.computeI_NMDA(Vpre, t, Vd)

    def I_NMDA_PPSE(self,I_NMDA):
        # if (self.VDend() - self.E_NMDA) == 0:
        #     return I_NMDA / 0.0000001
        # return I_NMDA / np.abs((self.VDend() - self.E_NMDA))
        return  self.Ra * self.B * self.g_NMDA* (self.E_leak - self.E_NMDA)

    # trasmission synaptique
    def T_fonc_CA1(self,Vpre):
        return 2.84 / (1 + np.exp(-(Vpre - 2.) / self.tau_Ex))

    def F_fonc_CA1(self,Vpre,d_s):
        Vd = Vpre
        for i in range(len(Vpre)):
            if d_s[i]==0: #soma
                Vd[i] =  1. / (1. + np.exp(-(Vpre[i] - 0.) / self.tau_GABA_s))
            else:
                Vd[i] =  1. / (1. + np.exp(-(Vpre[i] - 0.) / self.tau_GABA_d))
        # Vd[d_s] = 1. / (1. + np.exp(-(Vpre[d_s] - 0.) / self.tau_GABA_s))
        # Vd[np.bitwise_not(d_s)] = 1. / (1. + np.exp(-(Vpre[np.bitwise_not(d_s)] - 0.) / self.tau_GABA_d))
        return Vd



    def computeI_AMPA(self,Vpre):
        self.T_fonc = 1.1 * self.T_fonc_CA1(Vpre)
        #rk4
        self.s_AMPAo = self.s_AMPA + 0.  # y origine at t
        self.ds_AMPA1 = self.derivs_AMPA_CA1()
        # K1
        self.s_AMPA = self.s_AMPAo + self.ds_AMPA1 * self.dt / 2
        self.ds_AMPA2 = self.derivs_AMPA_CA1()
        # K2
        self.s_AMPA = self.s_AMPAo + self.ds_AMPA2 * self.dt / 2
        self.ds_AMPA3 = self.derivs_AMPA_CA1()
        # K3
        self.s_AMPA = self.s_AMPAo + self.ds_AMPA3 * self.dt
          # K4
        self.s_AMPA = self.s_AMPAo + self.dt / 6. * (self.ds_AMPA1 + 2 * self.ds_AMPA2 + 2 * self.ds_AMPA3 + self.derivs_AMPA_CA1())
        # y at t+dt
        return self.s_AMPA



    def derivs_AMPA_CA1(self):
        return self.T_fonc * (1 - self.s_AMPA) - 0.19 * self.s_AMPA


    def computeI_GABA(self, Vpre,d_s):
        self.F_fonc = 10. * self.F_fonc_CA1(Vpre,d_s)
        # rk4
        self.s_GABAo = self.s_GABA + 0.  # y origine at t
        self.ds_GABA1 = self.derivs_GABA_CA1()
        # K1
        self.s_GABA = self.s_GABAo + self.ds_GABA1 * self.dt / 2
        self.ds_GABA2 = self.derivs_GABA_CA1()
        # K2
        self.s_GABA = self.s_GABAo + self.ds_GABA2 * self.dt / 2
        self.ds_GABA3 = self.derivs_GABA_CA1()
        # K3
        self.s_GABA = self.s_GABAo + self.ds_GABA3 * self.dt
        # K4
        self.s_GABA = self.s_GABAo + self.dt / 6. * (self.ds_GABA1 + 2 * self.ds_GABA2 + 2 * self.ds_GABA3 + self.derivs_GABA_CA1())
        # y at t+dt
        return self.s_GABA

    def derivs_GABA_CA1(self):
        return self.F_fonc * (1. - self.s_GABA) - 0.07 * self.s_GABA
        # dyGABAdx[0] = 10. * F_fonc * (1. - yGABA[0]) - 0.09899 * yGABA[0]; // *racine(2)  dyGABAdx[0] = 10. * F_fonc * (1. - yGABA[0]) - 0.14 * yGABA[0]; // *2



    def computeI_NMDA(self,Vpre,t,Vs_d):
        for i in range(len(Vpre)):
            q = t - self.lastRelease[i] - self.Cdur

            if (q > self.deadTime):
                if (Vpre[i] > self.preThresh):
                    self.C[i] = self.Cmax
                    self.R0[i] = self.Ra[i]
                    self.lastRelease[i] = t
            elif (q < 0.):
               pass
            elif ( self.C[i] == self.Cmax):
                self.R1[i] = self.Ra[i]
                self.C[i] = 0.
            if (self.C[i] > 0):
                self.Rinf[i] = self.Cmax * self.alpha / (self.Cmax * self.alpha + self.beta)
                self.Rtau[i] = (1. / (self.Cmax * self.alpha + self.beta))
                self.Ra[i] = self.Rinf[i] + (self.R0[i]-self.Rinf[i]) * np.exp(-(t-self.lastRelease[i]) / self.Rtau[i])
            else:
                self.Ra[i] = self.R1[i] * np.exp(-1.0 * self.beta * (t-(self.lastRelease[i]+self.Cdur)))

        self.B = 1. / (1 + np.exp(0.062 * (-Vs_d)) * self.mg / 10.)  # 3.57); # mgblock(Vm)

        return self.Ra * self.B * self.g_NMDA * (Vs_d - self.E_NMDA)

    def VsOutput(self):
        return self.y[0]

    def VSoma(self):
        return self.y[0]

    def VDend(self):
        return self.y[1]

    def VAis(self):
        return self.y[28]

    def init_I_syn(self):
        self.I_synSoma = 0.
        self.I_synDend = 0.
        self.I_synAis = 0.

    def add_I_synSoma(self,I, vect):
        # self.I_synSoma += np.sum(I * vect)
        for i in range(len(I)):
            self.I_synSoma += I[i] * vect[i]

    def add_I_synDend(self, I, vect):
        # self.I_synDend += np.sum(I * vect)
        for i in range(len(I)):
            self.I_synDend += I[i] * vect[i]

    def add_I_synDend_Bis(self, I):
        # self.I_synDend += np.sum(I * vect)
        for i in range(len(I)):
            self.I_synDend += I[i]

    def add_I_synAis(self, I, vect):
        # self.I_synDend += np.sum(I * vect)
        for i in range(len(I)):
            self.I_synAis += I[i] * vect[i]
