__author__ = "Yochum Maxime"
__copyright__ = "No Copyright"
__credits__ = ["Yochum Maxime"]
__email__ = "maxime.yochum@univ-rennes1.fr"
__status__ = "Prototype"

import sys
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import os
from NeoCOMM import ModelMicro_GUI
import numpy as np
from datetime import datetime

app = QApplication(sys.argv)
ex = ModelMicro_GUI(app)
ex.resize(3840//2 , 2160)
# ex.mascene_EEGViewer.update_plot()

#load a simulation
Filepath = r"Simulations_CiBM_article/Figure2_A.txt"
ex.LoadSimul(Filepath = Filepath)
ex.mascene_EEGViewer.e_win.setText('1000')


# make new tissue
if 0: # load a simu file
    ex.PlaceCell_func()

if 0: #press place button
    ex.connectivityCell_func()



for i in range(1):
    np.random.seed(None)

    if 0: #change à val in the interface
        # plage = [1000,5000]
        # nbcell = np.random.randint(plage[0],plage[1])
        # ex.nbcellsnbtotal.setText(str(nbcell))
        # ex.Nb_Cell_Changed('total')
        ex.PlaceCell_func()

    if 0: # change division in the afference matrix

        plage = [20,100]
        division = np.random.randint(plage[0], plage[1])
        ex.CC.update_connections(ex.CC.Afferences, fixed=not ex.r0.isChecked(),division = division)
            # ex.connectivityCell_func()


    for k in range(20):
        np.random.seed(None)
        i += 1
        if 1: #change à val in the interface
            plage = [60,120]
            Inj = np.random.randint(plage[0],plage[1])
            ex.i_inj_e.setText(str(Inj))

        if 1:  # change à val in the interface
            plage_JITTER = [4, 10]
            JITTER = np.random.randint(plage_JITTER[0], plage_JITTER[1])
            ex.varianceStim_e.setText(str(JITTER))

        if 1:
            plage_nbStim = [10, 40]
            nbStim = np.random.randint(plage_nbStim[0], plage_nbStim[1])
            ex.nbStim_e.setText(str(nbStim))

        if 1:  # change model parameter / faire un cluster
            ex.ModXNMMclicked() # open modify X
            ex.NewModifyXNMM.getcenter_fun() # get the center position
            # ex.NewModifyXNMM.getClosestCell_fun(type=0)# get the most closer pyr cell from the center
            # ex.NewModifyXNMM.SelectAllclick()
            # ex.NewModifyXNMM.consider_color.setChecked(False)
            #     #get parameter
            # g_AMPA0 = np.round(np.random.uniform(7, 30),1)
            # ex.NewModifyXNMM.Oneparam_list[0].setCurrentText('g_AMPA')
            # ex.NewModifyXNMM.Oneparam_list[1].setText(str(g_AMPA0))
            # ex.NewModifyXNMM.setOneparam()
            # g_NMDA0 = np.round(np.random.uniform(0.3, 2),2)
            # ex.NewModifyXNMM.Oneparam_list[0].setCurrentText('g_NMDA')
            # ex.NewModifyXNMM.Oneparam_list[1].setText(str(g_NMDA0))
            # ex.NewModifyXNMM.setOneparam()
            # g_GABA0 = np.random.randint(5, 50)
            # ex.NewModifyXNMM.Oneparam_list[0].setCurrentText('g_GABA')
            # ex.NewModifyXNMM.Oneparam_list[1].setText(str(g_GABA0))
            # ex.NewModifyXNMM.setOneparam()
            # E_GABA0 = np.random.randint(-75, -55)
            # ex.NewModifyXNMM.Oneparam_list[0].setCurrentText('E_GABA')
            # ex.NewModifyXNMM.Oneparam_list[1].setText(str(E_GABA0))
            # ex.NewModifyXNMM.setOneparam()
            #     # ex.NewModifyXNMM.Applyclick()
            # ex.NewModifyXNMM.ClearAllclick()
                # ex.NewModifyXNMM.close()
            for type in [1, 2, 3]:
                if type==1:
                    ex.NewModifyXNMM.getClosestCell_fun(type=type)  # get the most closer pyr cell from the center
                    ex.NewModifyXNMM.SelectAllclick()
                    ex.NewModifyXNMM.consider_color.setChecked(False)
                    # get parameter
                    g_AMPA = np.random.uniform(6, 14)
                    ex.NewModifyXNMM.Oneparam_list[0].setCurrentText('g_AMPA')
                    ex.NewModifyXNMM.Oneparam_list[1].setText(str(g_AMPA))
                    ex.NewModifyXNMM.setOneparam()
                    # g_NMDA = np.random.uniform(0.01, 1)
                    # ex.NewModifyXNMM.Oneparam_list[0].setCurrentText('g_NMDA')
                    # ex.NewModifyXNMM.Oneparam_list[1].setText(str(g_NMDA))
                    # ex.NewModifyXNMM.setOneparam()
                    # g_GABA = np.random.uniform(1, 3)
                    # ex.NewModifyXNMM.Oneparam_list[0].setCurrentText('g_GABA')
                    # ex.NewModifyXNMM.Oneparam_list[1].setText(str(g_GABA))
                    # ex.NewModifyXNMM.setOneparam()

                if type==2:
                    ex.NewModifyXNMM.getClosestCell_fun(type=type)  # get the most closer pyr cell from the center
                    ex.NewModifyXNMM.SelectAllclick()
                    ex.NewModifyXNMM.consider_color.setChecked(False)
                    # get parameter
                    g_AMPA = np.random.uniform(6, 10)
                    ex.NewModifyXNMM.Oneparam_list[0].setCurrentText('g_AMPA')
                    ex.NewModifyXNMM.Oneparam_list[1].setText(str(g_AMPA))
                    ex.NewModifyXNMM.setOneparam()
                    # g_NMDA = np.random.uniform(0.01, 1)
                    # ex.NewModifyXNMM.Oneparam_list[0].setCurrentText('g_NMDA')
                    # ex.NewModifyXNMM.Oneparam_list[1].setText(str(g_NMDA))
                    # ex.NewModifyXNMM.setOneparam()
                    g_GABA = np.random.uniform(1, 3)
                    ex.NewModifyXNMM.Oneparam_list[0].setCurrentText('g_GABA')
                    ex.NewModifyXNMM.Oneparam_list[1].setText(str(g_GABA))
                    ex.NewModifyXNMM.setOneparam()

                if type==3:
                    ex.NewModifyXNMM.getClosestCell_fun(type=type)  # get the most closer pyr cell from the center
                    ex.NewModifyXNMM.SelectAllclick()
                    ex.NewModifyXNMM.consider_color.setChecked(False)
                    # get parameter
                    g_AMPA = np.random.uniform(6, 10)
                    ex.NewModifyXNMM.Oneparam_list[0].setCurrentText('g_AMPA')
                    ex.NewModifyXNMM.Oneparam_list[1].setText(str(g_AMPA))
                    ex.NewModifyXNMM.setOneparam()
                    # g_NMDA = np.random.uniform(0.01, 1)
                    # ex.NewModifyXNMM.Oneparam_list[0].setCurrentText('g_NMDA')
                    # ex.NewModifyXNMM.Oneparam_list[1].setText(str(g_NMDA))
                    # ex.NewModifyXNMM.setOneparam()
                    # g_GABA = np.random.uniform(4, 7)
                    # ex.NewModifyXNMM.Oneparam_list[0].setCurrentText('g_GABA')
                    # ex.NewModifyXNMM.Oneparam_list[1].setText(str(g_GABA))
                    # ex.NewModifyXNMM.setOneparam()

            if 0: # change model parameter / faire un cluster
                # ex.ModXNMMclicked() # open modify X
                ex.NewModifyXNMM.getcenter_fun() # get the center position
                ex.NewModifyXNMM.getClosestCell_fun(type=0)# get the most closer pyr cell from the center
                set_QPushButton_background_color(ex.NewModifyXNMM.colorbutton,QColor("#10d3cc"))
                radius = np.random.randint(100, 200)
                ex.NewModifyXNMM.radius_e.setText(str(radius))
                ex.NewModifyXNMM.radiusApply_PBclick()
                #get parameter
                E_GABA = np.random.randint(-75, -50)
                for k in range(len(ex.NewModifyXNMM.label_List__NMM_var)):
                    if ex.NewModifyXNMM.label_List__NMM_var[k].text() == 'E_GABA':
                        ex.NewModifyXNMM.Edit_List_NMM_var[k].setText(str(E_GABA))
                ex.NewModifyXNMM.consider_color.setChecked(True)
                ex.NewModifyXNMM.Applyclick()
            ex.NewModifyXNMM.close()


            if 1: #simulate
                ex.Reset_states_clicked()
                ex.simulate()
                ex.mascene_EEGViewer.e_spacing.setText('1')
                ex.mascene_EEGViewer.udpate_plot_plus_slider()

            if 1:
                ex.update_graph()
                ex.parent.processEvents()
            replot = True
            ex.electrode_angle2_e.setText(str(0))
            # for z in [2700]:

                # if 1:  # comput LFPs
            ex.electrode_z_e.setText(str(3000) )

                    # ex.Compute_LFP2_type_CB.setCurrentIndex(0) # "Dipole 1 Comp"
                    # ex.Compute_LFP_fonc(clear=replot,s='r_' + str(r)+'z_' + str(z))
                    # replot = False

                    # ex.Compute_LFP2_type_CB.setCurrentIndex(1)  # "Dipole Multi Comp"
                    # ex.Compute_LFP_fonc(clear=False)

            # ex.Compute_LFP2_type_CB.setCurrentIndex(2)  # "Mono 1"
            # ex.Compute_LFP_fonc(clear=replot,s='r_Mono1')

            #
            ex.Compute_LFP2_type_CB.setCurrentIndex(0)  # "Mono 2"
            ex.Compute_LFP_fonc(clear=replot, s=  'CSD')
            # replot = False
            # "save/Res_July_2023_mice/sim_July_base_mice_int.txt"
            Filepath = "Saves/"
            FileName = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            # FileName += "_division_" + str(division)
            # FileName += "_Inj_" + str(Inj)
            # FileName += "_JITTER_" + str(JITTER)
            # FileName += "_E_GABA_" + str(E_GABA)
            FileName += "_g_AMPA_" + str(g_AMPA)
            # FileName += "_g_NMDA_" + str(int(g_NMDA*1000))
            FileName += "_g_GABA_" + str(g_GABA)
            FileName = FileName.replace('.','_')

#                FileName += "_run_" + str(i)
            ex.resize(3840 // 2, 2160)
            FilePathName = os.path.join(Filepath, FileName)
            if 1: #Save simulation
                ex.SaveSimul(Filepath = FilePathName +'.txt' )
                print( FilePathName +'.txt')
                print('saved')
            if 1:#Save LFP fig

                ex.mascene_LFPViewer.figure.savefig(FilePathName +'.png')

            if 1:#Save LFP fig
                ex.centralWidget.grab().save(FilePathName + '_all.png')


        print('done')