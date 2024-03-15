import numpy as np
from numba.experimental import jitclass
from numba import boolean, int32, float64,uint8

def get_Variable_Names():
    return  ['g_leak','g_Na','g_K','g_Ca','g_KCa','g_h','g_KNa',
             'E_leak','E_Na','E_K','E_Ca','E_h',
             'g_AMPA','E_AMPA','g_GABA','E_GABA','g_NMDA','E_NMDA',
             'Cm','p','PERCENT']

# Type = 2 = Interneuron Somatostatine
spec = [
    ('Type' ,float64),
    ('Layer' ,float64),
    ('E_leak' ,float64),
    ('E_Na' ,float64),
    ('E_K' ,float64),
    ('E_Ca' ,float64),
    ('E_h' ,float64),
    ('g_Na' ,float64),
    ('g_KNa' ,float64),
    ('g_K' ,float64),
    ('g_Ca' ,float64),
    ('g_h' ,float64),
    ('g_KCa' ,float64),
    ('g_AMPA' ,float64),
    ('g_NMDA' ,float64),
    ('g_GABA' ,float64),
    ('E_AMPA' ,float64),
    ('E_NMDA' ,float64),
    ('E_GABA' ,float64),
    ('g_leak' ,float64),
    ('g_c' ,float64),
    ('p' ,float64),
    ('I_soma_stim' ,float64),
    ('Cm' ,float64),
    ('PERCENT' ,float64),
    ('I_synSoma' ,float64),
    ('F' ,float64),
    ('R' ,float64),
    ('T' ,float64),
    ('Q' ,float64),
    ('alpha_Na' ,float64),
    ('Rpump' ,float64),
    ('Na_eq' ,float64),
    ('A_s' ,float64),
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
    ('NbODEs', int32),
    ('NbODEs_s_AMPA', int32),
    ('NbODEs_s_GABA', int32),
    ('dydx', float64[:]),
    ('dydx1', float64[:]),
    ('dydx2', float64[:]),
    ('dydx3', float64[:]),
    ('y', float64[:]),
    ('yt', float64[:]),
    ('ds_AMPA1', float64[:]),
    ('ds_AMPA2', float64[:]),
    ('ds_AMPA3', float64[:]),
    ('s_AMPA', float64[:]),
    ('s_AMPAo', float64[:]),
    ('ds_GABA1', float64[:]),
    ('ds_GABA2', float64[:]),
    ('ds_GABA3', float64[:]),
    ('s_GABA', float64[:]),
    ('s_GABAo', float64[:]),
    ('m_inf_Na' ,float64),
    ('m_inf_Ca' ,float64),
    ('alpha_h' ,float64),
    ('alpha_n' ,float64),
    ('beta_h' ,float64),
    ('beta_n' ,float64),
    ('H_inf' ,float64),
    ('tau_H' ,float64),
    ('KCa_fonc' ,float64),
    ('I_Na' ,float64),
    ('I_K' ,float64),
    ('I_Ca' ,float64),
    ('I_KCa' ,float64),
    ('I_h' ,float64),
    ('I_leak' ,float64),
    ('m_inf' ,float64),
    ('alpha_h' ,float64),
    ('alpha_n' ,float64),
    ('beta_h' ,float64),
    ('beta_n' ,float64),
    ('I_Na' ,float64),
    ('I_K' ,float64),
    ('I_leak' ,float64),
    ('I_KNa' ,float64),
    ('T_fonc' ,float64[:]),
    ('F_fonc' ,float64[:]),
    ('fifteenpoxer3' ,float64),]



@jitclass(spec)
class RLNcell:

    def __init__(self):
        self.Type = 5
        self.Layer = 1
        # pot de repos
        self.E_leak = -60
        self.E_Na = 55
        self.E_K = -90
        self.E_Ca = 130
        self.E_h = -30

        #
        self.g_Na = 30
        self.g_KNa = 0
        self.g_K = 9
        self.g_Ca = 1
        self.g_h = 0.15
        self.g_KCa = 10
        # self.g_KNa_OLM = 10

        # synapses
        self.g_AMPA= 6
        self.g_NMDA = 0
        self.g_GABA = 1.38
        self.E_AMPA = 0.0
        self.E_NMDA = 0.0
        self.E_GABA = -75

        #Conductandes generales
        self.g_leak = 0.1
        self.g_c = 0.1
        self.p = 0.05

        #Courants imposés
        self.I_soma_stim = 0.0

        #parametres de membranes
        self.Cm = 1

        self.PERCENT = 0.1
        self.I_synSoma = 0.0

        #internal parameters
        self.F = 96480
        self.R = 8.315
        self.T = 273.16
        self.Q = self.F / (self.R * self.T)
        self.alpha_Na = 0.01
        self.Rpump = 0.060  # * Publi: 0.018 * /
        self.Na_eq = 9.5
        self.A_s = 0.015

        # NMDA
        self.Cdur = 1.
        self.deadTime = 1.
        self.Cmax = 1.
        self.preThresh = 0.
        self.alpha = 0.072
        self.beta = 0.0066
        # self.lastRelease = -1.e99
        # self.Rinf = 0.
        # self.Rtau = 0.
        # self.Ra = 0.
        # self.R0 = 0.
        # self.R1 = 0.
        # self.C = 0.
        self.mg = 1.

        #ODE vectors
        self.dt = 1. / 2048.
        self.NbODEs = 6
        self.NbODEs_s_AMPA = 1
        self.NbODEs_s_GABA = 1
        self.init_vector()


        self.setParameters()
        self.updateParameters()

        self.fifteenpoxer3 = np.power(15.0, 3.)


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

        self.Cdur = 1.
        self.deadTime = 1.
        self.Cmax = 1.
        self.preThresh = 0.
        self.alpha = 0.072
        self.beta = 0.0066
        # self.lastRelease = -1.e99
        # self.Rinf = 0.
        # self.Rtau = 0.
        # self.Ra = 0.
        # self.R0 = 0.
        # self.R1 = 0.
        # self.C = 0.
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
    def V_OLM_(self):
        return self.y[0]

    @V_OLM_.setter
    def V_OLM_(self, value):
        self.y[0] = value

    @property
    def dV_OLM_dt(self):
        return self.dydx[0]

    @dV_OLM_dt.setter
    def dV_OLM_dt(self, value):
        self.dydx[0] = value

    @property
    def h_OLM_(self):
        return self.y[1]

    @h_OLM_.setter
    def h_OLM_(self, value):
        self.y[1] = value

    @property
    def dh_OLM_dt(self):
        return self.dydx[1]

    @dh_OLM_dt.setter
    def dh_OLM_dt(self, value):
        self.dydx[1] = value

    @property
    def n_OLM_(self):
        return self.y[2]

    @n_OLM_.setter
    def n_OLM_(self, value):
        self.y[2] = value

    @property
    def dn_OLM_dt(self):
        return self.dydx[2]

    @dn_OLM_dt.setter
    def dn_OLM_dt(self, value):
        self.dydx[2] = value

    @property
    def Ca_OLM_(self):
        return self.y[3]

    @Ca_OLM_.setter
    def Ca_OLM_(self, value):
        self.y[3] = value

    @property
    def dCa_OLM_dt(self):
        return self.dydx[3]

    @dCa_OLM_dt.setter
    def dCa_OLM_dt(self, value):
        self.dydx[3] = value

    @property
    def H_OLM_(self):
        return self.y[4]

    @H_OLM_.setter
    def H_OLM_(self, value):
        self.y[4] = value

    @property
    def dH_OLM_dt(self):
        return self.dydx[4]

    @dH_OLM_dt.setter
    def dH_OLM_dt(self, value):
        self.dydx[4] = value

    # ajout IKNa

    @property
    def Na_intra_(self):
        return self.y[5]

    @Na_intra_.setter
    def Na_intra_(self, value):
        self.y[5] = value

    @property
    def dNa_intra_dt(self):
        return self.dydx[5]

    @dNa_intra_dt.setter
    def dNa_intra_dt(self, value):
        self.dydx[5] = value






    def setParameters(self):
        #[self applyInterfaceParameters:self]
        self.y[0] = self.E_leak
        self.y[1] = self.alpha_h_OLM(self.y[0]) / (self.alpha_h_OLM(self.y[0]) + self.beta_h_OLM(self.y[0]))
        self.y[2] = self.alpha_n_OLM(self.y[0]) / (self.alpha_n_OLM(self.y[0]) + self.beta_n_OLM(self.y[0]))
        self.y[3] = 0.0
        self.y[4] = self.H_inf_OLM(self.y[0]) * self.tau_H_OLM(self.y[0])
        self.y[5] = 9.5

    def bruitGaussien(self,s, m):
        return np.random.normal(m, s)

    def I_leak_OLM(self,g_leak, PERCENT, V, E_leak):
        return self.bruitGaussien(PERCENT * g_leak, g_leak) * (V - E_leak)

    def updateParameters(self):
        self.m_inf_Na = self.m_inf_Na_OLM(self.y[0])
        self.m_inf_Ca = self.m_inf_Ca_OLM(self.y[0])
        self.alpha_h = self.alpha_h_OLM(self.y[0])
        self.alpha_n = self.alpha_n_OLM(self.y[0])
        self.beta_h = self.beta_h_OLM(self.y[0])
        self.beta_n = self.beta_n_OLM(self.y[0])
        self.H_inf = self.H_inf_OLM(self.y[0])
        self.tau_H = self.tau_H_OLM(self.y[0])
        self.KCa_fonc = self.KCa_fonc_OLM(self.y[3])

        self.I_Na = self.I_Na_OLM(self.g_Na, self.m_inf_Na, self.y[1], self.y[0], self.E_Na)
        self.I_K = self.I_K_OLM(self.g_K, self.y[2], self.y[0], self.E_K)
        self.I_Ca = self.I_Ca_OLM(self.g_Ca, self.m_inf_Ca, self.y[0], self.E_Ca)
        self.I_KCa = self.I_KCa_OLM(self.g_KCa, self.KCa_fonc, self.y[0], self.E_K)
        self.I_h = self.I_h_OLM(self.g_h, self.y[4], self.y[0], self.E_K)
        self.I_leak = self.I_leak_OLM(self.g_leak, self.PERCENT, self.y[0], self.E_leak)

        # self.I_soma_stim = [self getI_soma_stim];

        # ajout IKNa definir y

        self.I_KNa = self.I_KNa_OLM(self.g_KNa, self.y[5], self.y[0], self.E_K)

    def rk4(self):
        self.yt = self.y+0. #y origine at t
        self.dydx1=self.derivT()#K1
        self.y = self.yt + self.dydx1 * self.dt / 2
        self.dydx2=self.derivT()#K2
        self.y = self.yt + self.dydx2 * self.dt / 2
        self.dydx3=self.derivT()#K3
        self.y = self.yt + self.dydx3 * self.dt
        self.derivT() #K4
        self.y =self.yt + self.dt/6. *(self.dydx1+2*self.dydx2+2*self.dydx3+self.dydx)#y at t+dt

    def derivT(self):
        # ajout cinet Na intra
        self.dV_OLM_dt = 1. / self.Cm * (
                    -self.I_KNa - self.I_Na - self.I_K - self.I_KCa - self.I_h - self.I_leak + self.I_soma_stim - self.I_synSoma)

        Na_intra_power3 = self.Na_intra_ * self.Na_intra_ * self.Na_intra_
        Na_eq_power3 = self.Na_eq * self.Na_eq * self.Na_eq

        self.dNa_intra_dt = -self.alpha_Na * (self.A_s * (self.I_Na) ) - self.Rpump * (
                    Na_intra_power3 / (Na_intra_power3 + self.fifteenpoxer3) - (
                        Na_eq_power3 / (Na_eq_power3 + self.fifteenpoxer3)))
        #!!!!! I_Na_s louche dans Inter_OLM_C.c

        self.dh_OLM_dt = 5. * (self.alpha_h * (1 - self.h_OLM_) - self.beta_h * self.h_OLM_)
        self.dn_OLM_dt = 5. * (self.alpha_n * (1 - self.n_OLM_) - self.beta_n * self.n_OLM_)
        self.dCa_OLM_dt = -0.002 * self.I_Ca - self.Ca_OLM_ / 80.
        self.dH_OLM_dt = (1 / self.tau_H) * (self.H_inf - self.H_OLM_)
        return self.dydx + 0.

    # ---------Gate Steady state and time constant equations - --------
    def m_inf_Na_OLM(self, V_OLM):
        alpha = (-0.1 * (V_OLM + 35.)) / (np.exp(-0.1 * (V_OLM + 35.)) - 1.)
        beta = 4. * (np.exp(-(V_OLM + 60.) / 18.))
        return alpha / (alpha + beta)

    # -------------Rate constant equations - --------
    def alpha_h_OLM(self, V_OLM):
        return 0.04 * np.exp(-(V_OLM + 58.) / 20.)


    def beta_h_OLM(self, V_OLM):
        n = 1.
        d = np.exp(-0.1 * (V_OLM + 28.)) + 1.
        return n / d

    # ---------------------------------------
    def alpha_n_OLM(self, V_OLM):
        n = -0.01 * (V_OLM + 34.)
        d = np.exp(-0.1 * (V_OLM + 34.)) - 1.
        return n / d

    def beta_n_OLM(self, V_OLM):
        return 0.125 * np.exp(-(V_OLM + 44.) / 80.)

    # ---------------------------------------
    def m_inf_Ca_OLM(self, V_OLM):
        n = 1.
        d = 1. + np.exp(-(V_OLM + 20.) / 9.)
        return n / d

    # ---------------------------------------
    def KCa_fonc_OLM(self, Ca):
        return Ca / (Ca + 30)

    # ---------------------------------------
    def H_inf_OLM(self, V_OLM):
        return 1. / (1. + np.exp((V_OLM + 80.) / 10.))


    def tau_H_OLM(self, V_OLM):
        n = 200.
        d = np.exp((V_OLM + 70.) / 20.) + np.exp(-(V_OLM + 70.) / 20.)
        return (n / d) + 5.

    # ---------------------------------------
    def F_fonc_OLM(self, Vpre):
        return 1. / (1. + np.exp(-(Vpre - 0.) /  2))#1. / (1. + np.exp(-(Vpre - teta_syn) / K))

    def T_fonc_OLM(self, Vpre):
        return 2.84 / (1 + np.exp(-(Vpre - 2.) / 5.))#Tmax / (1 + np.exp(-(Vpre - Vp) / Kp))

    # ---------------------------------------ajout IKNa

    def I_KNa_OLM(self, g_KNa_OLM, Na_intra, Vs, E_K):
        w = 0.37 / (1.0 + (38.7 / pow(Na_intra, 3.5)))
        return g_KNa_OLM * w * (Vs - E_K)

    # --------------------  COURANT - ----------------------
    def I_Na_OLM(self, g_Na_OLM, m_inf_Na_OLM, h_OLM, V_OLM, E_Na_OLM):
        return g_Na_OLM * m_inf_Na_OLM * m_inf_Na_OLM * m_inf_Na_OLM * h_OLM * (V_OLM - E_Na_OLM)


    def I_K_OLM(self, g_K_OLM, n_OLM, V_OLM, E_K_OLM):
        return g_K_OLM * n_OLM * n_OLM * n_OLM * n_OLM * (V_OLM - E_K_OLM)


    def I_Ca_OLM(self, g_Ca_OLM, m_inf_Ca_OLM, V_OLM, E_Ca_OLM):
        return g_Ca_OLM * m_inf_Ca_OLM * m_inf_Ca_OLM * (V_OLM - E_Ca_OLM)


    def I_KCa_OLM(self, g_KCa_OLM, KCa_fonc, V_OLM, E_K_OLM):
        return g_KCa_OLM * KCa_fonc * (V_OLM - E_K_OLM)


    def I_h_OLM(self, g_h_OLM, H_OLM, V_OLM, E_h_OLM):
        return g_h_OLM * H_OLM * (V_OLM - E_h_OLM)

    # -----------------------------------------------------
    def I_AMPA(self, Vd):
        return ((self.g_AMPA * 0.1) / 1.25) * self.s_AMPA * (Vd - self.E_AMPA)

    def I_AMPA2(self,Vpre):
        self.computeI_AMPA(Vpre)
        Vd = np.array([self.VSoma() for i in Vpre])
        return ((self.g_AMPA * 0.1) / 1.25) * self.s_AMPA * (Vd - self.E_AMPA)

    def I_GABA(self, Vd):
        return ((self.g_GABA * 0.1) / 1.25) * self.s_GABA * (Vd - self.E_GABA)

    def I_GABA2(self,Vpre):
        self.computeI_GABA(Vpre)
        # Vd = np.array([self.VSoma() for s_d in PreSynaptic_Soma_Dend_GABA])
        return ((self.g_GABA * 0.1) / 1.25) * self.s_GABA * (self.VSoma() - self.E_GABA)

    def I_NMDA2(self,Vpre,t):
        Vd = np.array([self.VSoma() for i in Vpre])
        return self.computeI_NMDA(Vpre, t, Vd)

    def I_syn_OLM(self, I_GABA_OLM, I_AMPA_OLM, I_NMDA_OLM):
        return (I_GABA_OLM + I_AMPA_OLM + I_NMDA_OLM)

    # trasmission synaptique
    def computeI_AMPA(self, Vpre):
        self.T_fonc = 1.1 * self.T_fonc_OLM(Vpre)
        # rk4
        self.s_AMPAo = self.s_AMPA + 0.  # y origine at t
        self.ds_AMPA1 = self.derivs_AMPA_OLM()
        # K1
        self.s_AMPA = self.s_AMPAo + self.ds_AMPA1 * self.dt / 2
        self.ds_AMPA2 = self.derivs_AMPA_OLM()
        # K2
        self.s_AMPA = self.s_AMPAo + self.ds_AMPA2 * self.dt / 2
        self.ds_AMPA3 = self.derivs_AMPA_OLM()
        # K3
        self.s_AMPA = self.s_AMPAo + self.ds_AMPA3 * self.dt
        # K4
        self.s_AMPA = self.s_AMPAo + self.dt / 6. * (self.ds_AMPA1 + 2 * self.ds_AMPA2 + 2 * self.ds_AMPA3 + self.derivs_AMPA_OLM())
        # y at t+dt
        return self.s_AMPA

    def derivs_AMPA_OLM(self):
        return self.T_fonc * (1-self.s_AMPA) - 0.19 * self.s_AMPA

    def computeI_GABA(self, Vpre):
        self.F_fonc = 1. * self.F_fonc_OLM(Vpre)
        # rk4
        self.s_GABAo = self.s_GABA + 0.  # y origine at t
        self.ds_GABA1 = self.derivs_GABA_OLM()
        # K1
        self.s_GABA = self.s_GABAo + self.ds_GABA1 * self.dt / 2
        self.ds_GABA2 = self.derivs_GABA_OLM()
        # K2
        self.s_GABA = self.s_GABAo + self.ds_GABA2 * self.dt / 2
        self.ds_GABA3 = self.derivs_GABA_OLM()
        # K3
        self.s_GABA = self.s_GABAo + self.ds_GABA3 * self.dt
        # K4
        self.s_GABA = self.s_GABAo + self.dt / 6. * (self.ds_GABA1 + 2 * self.ds_GABA2 + 2 * self.ds_GABA3 + self.derivs_GABA_OLM())
        # y at t+dt
        return self.s_GABA

    def derivs_GABA_OLM(self):
        len(self.F_fonc)
        len(self.s_GABA)
        return self.F_fonc * (1. - self.s_GABA) - 0.07 * self.s_GABA

    def computeI_NMDA(self, Vpre, t, Vs_d):
        for i in range(len(Vpre)):
            q = t - self.lastRelease[i] - self.Cdur

            if (q > self.deadTime):
                if (Vpre[i] > self.preThresh):
                    self.C[i] = self.Cmax
                    self.R0[i] = self.Ra[i]
                    self.lastRelease[i] = t
            elif (q < 0.):
                pass
            elif (self.C[i] == self.Cmax):
                self.R1[i] = self.Ra[i]
                self.C[i] = 0.
            if (self.C[i] > 0):
                self.Rinf[i] = self.Cmax * self.alpha / (self.Cmax * self.alpha + self.beta)
                self.Rtau[i] = (1. / (self.Cmax * self.alpha + self.beta))
                self.Ra[i] = self.Rinf[i] + (self.R0[i] - self.Rinf[i]) * np.exp(
                    -(t - self.lastRelease[i]) / self.Rtau[i])
            else:
                self.Ra[i] = self.R1[i] * np.exp(-1.0 * self.beta * (t - (self.lastRelease[i] + self.Cdur)))

        self.B = 1. / (1 + np.exp(0.062 * (-Vs_d)) * self.mg / 10.)  # 3.57); # mgblock(Vm)

        return self.Ra * self.B * self.g_NMDA * (Vs_d - self.E_NMDA)


    def VsOutput(self):
        return self.y[0]

    def VSoma(self):
        return self.y[0]

    def init_I_syn(self):
        self.I_synSoma = 0.

    def add_I_synSoma(self, I):
        # self.I_synSoma += np.sum(I * vect)
        for i in range(len(I)):
            self.I_synSoma += I[i]