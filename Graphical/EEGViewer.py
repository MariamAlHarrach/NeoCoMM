__author__ = 'Maxime'
# -*- coding: utf-8 -*-


import matplotlib
import matplotlib as mpl
matplotlib.use('Qt5Agg')
mpl.rcParams['path.simplify'] = True
mpl.rcParams['path.simplify_threshold'] = 1.0
import matplotlib.style as mplstyle
mplstyle.use('fast')
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.ticker import MultipleLocator

from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import copy
from fractions import Fraction
import numpy as np
from scipy import signal
import csv
import pickle
import scipy
import pyedflib


def set_QPushButton_background_color(button=None, color=None):
    if color==None or button==None :
        return
    else :
        button.setAutoFillBackground(True)
        values = "{r}, {g}, {b} ".format(r = color.red(),
                                     g = color.green(),
                                     b = color.blue())
        button.setStyleSheet("QPushButton { background-color: rgb("+values+"); }")

def label_color_clicked(event,button):
    color = QColor(button.palette().button().color())
    colordial = QColorDialog(color)
    colordial.exec_()
    selectedcolor = colordial.currentColor()
    colordial.close()
    set_QPushButton_background_color(button,selectedcolor)
    pass


class LineEdit(QLineEdit):
    KEY = Qt.Key_Return
    def __init__(self, *args, **kwargs):
        QLineEdit.__init__(self, *args, **kwargs)
        QREV = QRegExpValidator(QRegExp("[+-]?\\d*[\\.]?\\d+"))
        QREV.setLocale(QLocale(QLocale.English))
        self.setValidator(QREV)


class LineEdit_Int(QLineEdit):
    KEY = Qt.Key_Return
    def __init__(self, *args, **kwargs):
        QLineEdit.__init__(self, *args, **kwargs)
        QREV = QRegExpValidator(QRegExp("[+-]?\\d+"))
        QREV.setLocale(QLocale(QLocale.English))
        self.setValidator(QREV)

class lfpViewer_EEGNewWindow(QMainWindow):
    def __init__(self, parent=None, Sigs_dict=None,
                 Sigs_Color=None,percentage=None):
        super(lfpViewer_EEGNewWindow, self).__init__()
        self.parent = parent
        self.resize(800, 600)
        self.setWindowTitle('Results')
        self.centralWidget = QWidget()
        self.setCentralWidget(self.centralWidget)
        self.mainHBOX_param_scene = QHBoxLayout()
        self.mascene = lfpViewer_EEG(self)
        self.mainHBOX_param_scene.addWidget(self.mascene)
        self.centralWidget.setLayout(self.mainHBOX_param_scene)
        self.mascene.update(Sigs_dict=Sigs_dict, Colors=Sigs_Color,percentage=percentage)

class lfpViewer_EEG(QMainWindow):
    def __init__(self, parent=None):
        super(lfpViewer_EEG, self).__init__()
        self.parent = parent

        #######################################
        self.centralWidget = QWidget()
        self.setCentralWidget(self.centralWidget)
        self.mainVBOX_param_scene = QVBoxLayout()
        # self.setMinimumHeight(700)
        # self.setMinimumWidth(800)
        self.mascene = EEG_plot(self)

        self.paramPlotV = QVBoxLayout()
        # self.horizontalSliders  = MySlider(Qt.Horizontal)
        self.horizontalSliders  = StyledTextScrollBar()
        self.horizontalSliders.text_pre = '0s'
        self.horizontalSliders.setFocusPolicy(Qt.StrongFocus)
        self.horizontalSliders.valueChanged.connect(self.movesliderfun)
        self.horizontalSliders.sliderPressed.connect(self.sliderPressedfun)
        self.horizontalSliders.sliderMoved.connect(self.sliderMovedfun)
        self.horizontalSliders.sliderReleased.connect(self.sliderReleasedfun)
        self.horizontalSliders.setMinimum(0)
        self.horizontalSliders.setMaximum(1)

        self.paramPlot = QHBoxLayout()
        self.paramPlot.setAlignment(Qt.AlignLeft)
        l_gain = QLabel('Gain')
        self.e_gain = LineEdit('0.1')
        l_win = QLabel('Window')
        self.e_win = LineEdit('400')
        l_spacing = QLabel('vertical spacing')
        self.e_spacing = LineEdit('1')
        l_linewidth = QLabel('linewidth')
        self.e_linewidth = LineEdit('1')
        # self.Color_Manage = QPushButton('Color Managment')
        self.Sig_Manage = QPushButton('Show signals')
        self.Filter_Manage = QPushButton('Filter Managment')
        self.Filter_apply = QCheckBox('Apply Filter')
        self.Pop_up = QPushButton('Pop up')
        self.SaveRes_PB = QPushButton('Export Signals')

        self.e_gain.returnPressed.connect(self.update_plot)
        self.e_win.returnPressed.connect(self.udpate_plot_plus_slider)

        self.e_spacing.returnPressed.connect(self.update_plot)
        self.e_linewidth.returnPressed.connect(self.update_plot)
        # self.Color_Manage.clicked.connect(self.Color_Manage_fun)
        self.Sig_Manage.clicked.connect(self.Sig_Manage_fun)
        self.Filter_Manage.clicked.connect(self.Filter_Manage_fun)
        self.Filter_apply.clicked.connect(self.Filter_apply_fun)
        self.Pop_up.clicked.connect(self.fun_popup)
        self.SaveRes_PB.clicked.connect(self.SaveRes_fun)

        self.paramPlot.addWidget(l_gain)
        self.paramPlot.addWidget(self.e_gain)
        self.paramPlot.addWidget(l_win)
        self.paramPlot.addWidget(self.e_win)
        self.paramPlot.addWidget(l_spacing)
        self.paramPlot.addWidget(self.e_spacing)
        self.paramPlot.addWidget(l_linewidth)
        self.paramPlot.addWidget(self.e_linewidth)
        # self.paramPlot.addWidget(self.Color_Manage)
        self.paramPlot.addWidget(self.Sig_Manage)
        # self.paramPlot.addWidget(self.Filter_Manage)
        # self.paramPlot.addWidget(self.Filter_apply)
        # self.paramPlot.addWidget(self.Pop_up)
        # self.paramPlot.addWidget(self.SaveRes_PB)
        self.paramPlot.addStretch(1)

        self.paramPlot2 = QHBoxLayout()
        self.paramPlot2.setAlignment(Qt.AlignLeft)
        self.Toolbox = QCheckBox('Toolbox')
        l_Ticks = QLabel('Ticks (s)')
        self.e_Ticks = LineEdit('10')
        self.Toolbox.clicked.connect(self.displaytoolbox)
        self.e_Ticks.returnPressed.connect(self.update_plot)
        self.paramPlot2.addWidget(self.Toolbox)
        self.paramPlot2.addWidget(l_Ticks)
        self.paramPlot2.addWidget(self.e_Ticks)
        self.paramPlot2.addWidget(self.Filter_Manage)
        self.paramPlot2.addWidget(self.Filter_apply)
        self.paramPlot2.addStretch(1)
        self.paramPlot2.addWidget(self.SaveRes_PB)

        self.paramPlotV.addWidget(self.horizontalSliders)
        self.paramPlotV.addLayout(self.paramPlot)
        self.paramPlotV.addLayout(self.paramPlot2)




        self.mainVBOX_param_scene.addWidget(self.mascene)
        self.mainVBOX_param_scene.addLayout(self.paramPlotV)

        self.centralWidget.setLayout(self.mainVBOX_param_scene)
        self.moveslider = False
        self.t = np.zeros(2)

    def setWindowSizeWithoutRedraw(self,val):
        self.e_win.blockSignals(True)
        self.e_win.setText(str(val))
        self.e_win.blockSignals(False)

    def fun_popup(self):
        Sigs_dict = copy.deepcopy(self.Sigs_dict)
        if not 't' in Sigs_dict:
            Sigs_dict['t'] = self.t
        newpopup = lfpViewer_EEGNewWindow(Sigs_dict=Sigs_dict, Sigs_Color=self.Sigs_Color,percentage = self.percentage)
        newpopup.show()

    def sliderPressedfun(self):
        self.horizontalSliders.valueChanged.disconnect()

    def sliderMovedfun(self, e):
        self.horizontalSliders.setValue(e)
        self.update_slider_texts()

    def sliderReleasedfun(self):
        self.horizontalSliders.valueChanged.connect(self.movesliderfun)
        self.movesliderfun()

    def movesliderfun(self):
        self.update_slider_texts()
        self.horizontalSliders.setEnabled(False)
        self.update_data()
        self.horizontalSliders.setEnabled(True)

    def Filter_apply_fun(self):
        if self.Filter_apply.isChecked():
            self.Sigs_dict = copy.deepcopy(self.Sigs_dict_o)
            for Filter_info in self.Filter_list:
                self.Sigs_dict = signalfilterbandpass_EEG(self.Sigs_dict, self.Fs, Filter_info)
        else:
            self.Sigs_dict = copy.deepcopy(self.Sigs_dict_o)
        self.mascene.modify_sigs()
        self.update_data()

    def updateslider(self):
        self.horizontalSliders.setMinimum(0)
        self.horizontalSliders.setMaximum(int(np.ceil((self.t[-1]-self.t[0]) / int(self.e_win.text())) - 1))
        self.horizontalSliders.setPageStep(1)
        self.update_slider_texts()

    def update_slider_texts(self):
        ts, te = self.get_ts_te()
        ts = np.round(self.t[ts])
        te = np.round(self.t[te - 1])
        # self.horizontalSliders.setPreText(str(datetime.timedelta(seconds=int(np.round(self.t[0])))))
        # self.horizontalSliders.setPostText(str(datetime.timedelta(seconds=int(np.round(self.t[-1])))))
        # self.horizontalSliders.setSliderText(str(datetime.timedelta(seconds=int(ts))) + '/' + str(datetime.timedelta(seconds=int(te))))
        self.horizontalSliders.setPreText( "{:.3f}".format(self.t[0]/1000)  + ' s')
        self.horizontalSliders.setPostText("{:.3f}".format(self.t[-1]/1000) + ' s')
        self.horizontalSliders.setSliderText("{:.3f}".format(ts/1000)  + ' s' + '/' + "{:.3f}".format(te/1000) + ' s')

        self.horizontalSliders.update()

    def get_ts_te(self):
        win = float(self.e_win.text())
        if win > self.t[-1] - self.t[0]:
            win = np.ceil(self.t[-1] - self.t[0])
            self.e_win.setText(str(int(win)))
        ts = int(win * (self.horizontalSliders.value()) * self.Fs)
        te = ts + int(win * self.Fs)
        if te > len(self.t):
            diff = te - len(self.t)
            ts = ts - diff
            if ts < 0:
                ts=0
            te = len(self.t)
        return ts, te

    def udpate_plot_plus_slider(self):
        self.updateslider()
        self.update_plot()

    def update_plot(self):
        self.mascene.update()

    def update_data(self):
        self.mascene.update_set_data()

    def update(self, Sigs_dict=None, Colors=None, percentage = None):
        try:
            Sigs_dict = copy.copy(self.parent.Sigs_dict)
            Colors = copy.copy(self.parent.Color)
            percentage = copy.copy(self.parent.percentage)

        except:
            pass

        print('inside draw ')


        self.Sigs_dict_o = copy.deepcopy(Sigs_dict)

        self.t = self.Sigs_dict_o.pop('t')
        self.Fs = int(1 / (self.t[1] - self.t[0]))
        self.LFP_Names = list(self.Sigs_dict_o.keys())
        if percentage<0:
            percentage=0
        elif percentage>100:
            percentage=100
        self.percentage = percentage

        self.LFP_Names_o = self.LFP_Names
        self.Sigs_Color_o = Colors
        self.Sigs_Color = copy.deepcopy(self.Sigs_Color_o)
        self.LFP_Names = copy.deepcopy(self.LFP_Names_o)
        # self.t = tp[cutint:]
        self.Fs = int(1 / (self.t[1] - self.t[0]))

        self.Sigs_dict = copy.deepcopy(self.Sigs_dict_o)
        perInt = int(100 - self.percentage)
        Frac = Fraction(perInt / 100).limit_denominator(100)
        for i, key in reversed(list(enumerate(self.Sigs_dict))):
            if np.mod(i,Frac.denominator) < Frac.numerator:
                self.Sigs_dict.pop(key)
                self.LFP_Names.pop(i)
                self.Sigs_Color.pop(i)



        if  hasattr(self, 'LFP_to_show'):
            if not len(self.LFP_to_show)==len(self.LFP_Names):
                self.LFP_to_show = [True] * len(self.LFP_Names)
            else:
                pass
                #self.LFP_to_show = [True] * len(self.LFP_Names)
        else:
            self.LFP_to_show = [True] * len(self.LFP_Names)

        if not hasattr(self, 'Filter_list'):
            self.Filter_list = []

        self.horizontalSliders.blockSignals(True)
        self.updateslider()
        self.horizontalSliders.blockSignals(False)
        self.mascene.modify_sigs()
        self.update_plot()
        self.Filter_apply_fun()
        self.mascene.modify_sigs()
        self.update_plot()

    def Get_Sig_as_array(self):
        Names = []
        LFPs = []
        for key in self.Sigs_dict.keys():
                if not key == 't':
                    Names.append(key)
                    LFPs.append(self.Sigs_dict[key])
        return np.array(LFPs), Names, self.Fs

    def displaytoolbox(self):
        self.mascene.displaytoolbar(val=self.Toolbox.isChecked())

    def Sig_Manage_fun(self):
        Norm = Sig_Managment(self)
        if Norm.exec_():
            self.mascene.modify_sigs()
            self.update_data()

    def Filter_Manage_fun(self):
        try:
            Norm = Filter_Management(self)
            if Norm.exec_():
                self.Filter_apply_fun()
                # self.mascene.modify_sigs()
                # self.update_data()
        except:
            pass

    def SaveRes_fun(self):
        fileName = QFileDialog.getSaveFileName(caption='Save parameters', filter=".edf (*.edf);;.csv (*.csv);;.pickle (*.pickle);;.mat (*.mat)")
        if (fileName[0] == ''):
            return

        tp = self.t
        Sigs_dict = self.Sigs_dict
        Sigs_dict["t"] = tp

        lims = 0
        for key, value in self.Sigs_dict.items():
            if key not in 't':
                lims = max(lims, np.max(np.abs(value)))
                lims *= 2

        if fileName[1] == '.pickle (*.pickle)':
            file_pi = open(fileName[0], 'wb')
            pickle.dump(Sigs_dict, file_pi, -1)
            file_pi.close()
        elif fileName[1] == '.edf (*.edf)':
            Fs = int(1. / (tp[1] - tp[0]))
            Sigs_dict.pop("t")
            N = len(Sigs_dict.keys())
            f = pyedflib.EdfWriter(fileName[0], N, file_type=1)
            for i, key in enumerate(Sigs_dict.keys()):
                f.setLabel(i, key)
                f.setSamplefrequency(i, Fs)
                f.setPhysicalMaximum(i, lims)
                f.setPhysicalMinimum(i, -lims)
            f.update_header()

            kiter = len(tp) // Fs
            for k in range(kiter):
                for i, key in enumerate(Sigs_dict.keys()):
                    f.writePhysicalSamples(Sigs_dict[key][k * Fs:(k + 1) * Fs].flatten())

            f.update_header()
            f.close()

        elif fileName[1] == '.mat (*.mat)':
            scipy.io.savemat(fileName[0], mdict=Sigs_dict)
        elif fileName[1] == '.csv (*.csv)':
            f = open(fileName[0], 'w')
            w = csv.writer(f, delimiter='\t', lineterminator='\n')
            w.writerow(Sigs_dict.keys())
            for values in Sigs_dict.values():
                w.writerow(['{:e}'.format(var) for var in values])
            f.close()



class EEG_plot(QGraphicsView):
    def __init__(self, parent=None):
        super(EEG_plot, self).__init__(parent)
        self.parent = parent
        self.setStyleSheet("border: 0px")
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.figure = Figure(facecolor='white')#Figure()
        self.figure.canvas.mpl_connect('button_press_event', self.onclick)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas,None)

        self.widget = QWidget()
        self.widget.setLayout(QVBoxLayout())
        self.widget.layout().setContentsMargins(0, 0, 0, 0)
        self.widget.layout().setSpacing(0)
        self.scroll = QScrollArea(self.widget)
        self.scroll.setWidget(self.canvas)


        self.axes = self.figure.add_subplot(111)
        self.axes.set_xlabel("Time (s)")

        # self.canvas.setGeometry(0, 0, 1500, 500)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.scroll)
        self.displaytoolbar()
        self.setLayout(layout)
        self.spacing = 5
        self.Ticksspace = 1
        self.Fs =1024
        self.t = np.array([0,1,2])

    def displaytoolbar(self,val=False):
        self.toolbar.setVisible(val)

    def mini_maxi(self):
        self.mini = np.inf
        self.maxi = -np.inf
        for i, key in enumerate(self.LFP_Names):
            mini = np.min(self.Sigs_dict[key])
            maxi = np.max(self.Sigs_dict[key])
            if mini < self.mini:
                self.mini = mini
            if maxi > self.maxi:
                self.maxi = maxi


    def modify_sigs(self):
        self.Sigs_dict = self.parent.Sigs_dict
        self.Sigs_Color = self.parent.Sigs_Color
        self.LFP_Names = self.parent.LFP_Names
        self.t = self.parent.t
        self.Fs= int(1/(self.t[1]-self.t[0]))

    def onclick(self,event):
        #     print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
        #           ('double' if event.dblclick else 'single', event.button,
        #            event.x, event.y, event.xdata, event.ydata))
        pass

    def pick_handler(self,event):
        if event.mouseevent.button == 3:
            text = ''
            if type(event.artist) is matplotlib.text.Text:
                text = event.artist._text
            elif type(event.artist) is matplotlib.lines.Line2D:
                text = event.artist._label
            menu = QMenu(self)
            HideAction = QAction(self)
            HideAction.setText("&Hide " + text)
            ShowAction = QAction(self)
            ShowAction.setText("&Show " + text)
            menu.addAction(HideAction)
            menu.addAction(ShowAction)
            HideAction.triggered.connect(lambda state, x=text  : self.HideAction_fun(state,x))
            ShowAction.triggered.connect(lambda state, x=text  : self.ShowAction_fun(state,x))

            # print(event.mouseevent.x,event.mouseevent.y,self.height(),(self.parent.height()-100)*self.spacing)
            menu.exec_( event.guiEvent.globalPos())



    def HideAction_fun(self,event,line=None):
        if not line == '':
            self.parent.LFP_to_show[self.parent.LFP_Names.index(line)]=False
        self.update_set_data()

    def ShowAction_fun(self,event,line=None):
        if not line == '':
            self.parent.LFP_to_show[self.parent.LFP_Names.index(line)]=True
        self.update_set_data()

    def update_set_data(self):
        win_num = self.parent.horizontalSliders.value()
        self.gain = float(self.parent.e_gain.text())
        self.win= float(self.parent.e_win.text())
        if not self.spacing == float(self.parent.e_spacing.text()):
            self.spacing = float(self.parent.e_spacing.text())
            spacing = True
        else:
            spacing = False
        self.linewidth = float(self.parent.e_linewidth.text())

        ts = int(self.win * (win_num) * self.Fs)
        te = ts + int(self.win * self.Fs)
        if te > len(self.t):
            diff = te - len(self.t)
            ts = ts - diff
            if ts < 0:
                ts=0
            te = len(self.t)

        decimate = len(self.t[ts:te]) // 10000 + 1

        for i, key in enumerate(self.LFP_Names):
            self.Lines[i].set_data(self.t[ts:te:decimate], self.gain * (self.Sigs_dict[key][ts:te:decimate]) + i * self.spacing)
            self.Lines[i].set_color(self.Sigs_Color[i])
            self.Lines[i].set_linewidth(self.linewidth)
            self.Lines[i].set_visible(True)
            if not self.parent.LFP_to_show[i]:
                self.Lines[i].set_visible(False)

        if spacing:
            self.axes.set_ylim((-self.spacing, (len(self.LFP_Names) + 1) * self.spacing))
            self.axes.set_yticks(np.arange(len(self.LFP_Names[:])) * self.spacing)
            self.axes.set_yticklabels(self.LFP_Names)
            for label in self.axes.get_yticklabels():  # make the xtick labels pickable
                label.set_picker(True)


        self.axes.set_xlim((self.t[ts], self.t[te-1]))
        self.axes.xaxis.grid(which='both', color='#B0B0B0', linestyle='-', linewidth=0.5)
        self.canvas.draw_idle()
        # self.canvas.draw()

    def update(self):
        win_num = self.parent.horizontalSliders.value()
        self.figure.clear()
        self.figure.canvas.mpl_connect('pick_event', self.pick_handler)
        self.figure.subplots_adjust(left=0.1, bottom=0.01, right=1, top=1, wspace=0.0 , hspace=0.0 )
        self.axes = self.figure.add_subplot(1, 1, 1)
        self.gain = float(self.parent.e_gain.text())
        self.win= float(self.parent.e_win.text())
        self.spacing = float(self.parent.e_spacing.text())
        self.linewidth = float(self.parent.e_linewidth.text())
        self.Ticksspace = float(self.parent.e_Ticks.text())
        ts = int(self.win*(win_num) * self.Fs)
        te = ts + int(self.win * self.Fs)
        if te > len(self.t):
            diff = te - len(self.t)
            ts = ts - diff
            if ts < 0:
                ts=0
            te = len(self.t)

        decimate =  len(self.t[ts:te])//10000 + 1
        self.Lines = []
        for i, key in enumerate(self.LFP_Names):
            line, = self.axes.plot(self.t[ts:te:decimate], self.gain * (self.Sigs_dict[key][ts:te:decimate]) + i * self.spacing, color=self.Sigs_Color[i], linewidth=self.linewidth, label=key)
            line.set_picker(2)
            self.Lines.append(line)
            if not self.parent.LFP_to_show[i]:
                self.Lines[i].set_visible(False)




        self.axes.set_yticks(np.arange(len(self.LFP_Names[:]))*self.spacing)
        self.axes.set_yticklabels(self.LFP_Names)
        for label in self.axes.get_yticklabels():  # make the xtick labels pickable
            label.set_picker(True)
        minorLocator = MultipleLocator(self.Ticksspace)
        minorLocator.MAXTICKS = 10000
        self.axes.xaxis.set_minor_locator(minorLocator)
        majorLocator = MultipleLocator(999999999)
        self.axes.xaxis.set_major_locator(majorLocator)
        self.axes.xaxis.grid(which='both', color='#B0B0B0', linestyle='-', linewidth=0.5)

        self.axes.autoscale(enable=True, axis='both', tight=True)
        self.axes.set_ylim((-self.spacing,(len(self.LFP_Names[:])+1)*self.spacing))
        self.axes.set_xlim((self.t[ts], self.t[te-1]))




        # self.figure.set_size_inches(len(LFP_Names[:])*10, self.parent.width()/8)
        self.canvas.setGeometry(0, 0, self.parent.width()-100, int((self.parent.height()-100)*self.spacing))
        self.canvas.draw_idle()
        # self.canvas.show()


class Sig_Managment(QDialog):
    def __init__(self, parent=None):
        super(Sig_Managment, self).__init__()
        self.parent = parent

        self.Names = self.parent.LFP_Names
        self.LFP_to_show = self.parent.LFP_to_show

        self.layoutparam = QVBoxLayout()

        self.layoutparam_man = QHBoxLayout()
        Label = QLabel('Select signals to show')
        self.layoutparam.addWidget(Label)

        layout_range = QVBoxLayout()
        grid = QGridLayout()
        layout_range.addLayout(grid)

        # nvariable setting
        # nvariable setting
        N = len(self.Names)
        sqrt_N = int(np.sqrt(N))
        nb_line = sqrt_N * 2
        nb_column = int(np.ceil(N / nb_line))

        # N = len(self.Names)
        # sqrt_N = int(np.sqrt(N))
        # if N<=100:
        #     nb_column = 10
        # else:
        #     nb_column = 20
        # nb_line = int(np.ceil(N / nb_column))

        self.CBs = []
        self.col_but = []
        grid_ligne_ind = 0
        grid_col_ind = 0
        for ind, n in enumerate(self.Names):
            Label = QLabel(n)
            CB = QCheckBox()
            if self.LFP_to_show[ind]:
                CB.setChecked(True)
            else:
                CB.setChecked(False)
            self.CBs.append(CB)
            grid.addWidget(CB, grid_ligne_ind, grid_col_ind)
            grid.addWidget(Label, grid_ligne_ind, grid_col_ind + 2)

            # grid_col_ind += 3
            # if grid_col_ind >= nb_column*3:
            #     grid_col_ind = 0
            #     grid_ligne_ind += 1

            grid_ligne_ind += 1
            if grid_ligne_ind >= nb_line:
                grid_ligne_ind = 0
                grid_col_ind += 3
        self.layoutparam_man.addLayout(layout_range)

        self.actionbutton_l = QVBoxLayout()
        self.actionbutton_l.setAlignment(Qt.AlignTop)
        self.selectall = QPushButton('Select all')
        self.cleartall = QPushButton('Clear all')
        self.selectall.clicked.connect(self.selectALL)
        self.cleartall.clicked.connect(self.clearALL)
        self.actionbutton_l.addWidget(self.selectall)
        self.actionbutton_l.addWidget(self.cleartall)



        self.layoutparam_man.addLayout(self.actionbutton_l)

        self.layoutparam.addLayout(self.layoutparam_man)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Horizontal, self)
        self.buttons.accepted.connect(self.myaccept)
        self.buttons.rejected.connect(self.reject)

        self.layoutparam.addWidget(self.buttons)

        self.setLayout(self.layoutparam)

    def apply_fun(self):
        Colors = self.colorbutton.palette().button().color().name()
        for i, cb in enumerate(self.CBs):
            if cb.isChecked():
                set_QPushButton_background_color(self.col_but[i], QColor(Colors))

    def selectALL(self):
        for cb in self.CBs:
            cb.setChecked(True)

    def clearALL(self):
        for cb in self.CBs:
            cb.setChecked(False)

    def myaccept(self):
        Colors = []
        for idx, cb in enumerate(self.CBs):
            self.LFP_to_show[idx]=cb.isChecked()
        self.parent.LFP_to_show = self.LFP_to_show
        self.accept()

def iir_band_filter_EEG(ite_data, fs, btype, ftype, order=None, Quality=None , window=None, lowcut=None, highcut=None, zerophase=None, rps=None):
    fe = fs / 2.0
    if not lowcut == ''  and not highcut == '':
        wn = [lowcut / fe, highcut / fe]
    elif not lowcut == '':
        wn = lowcut / fe
    elif not highcut == '':
        wn = highcut / fe

    if rps[0]=='':
        rps[0] =10
    if rps[1]=='':
        rps[1] =10

    if btype in ["butter","bessel","cheby1","cheby2","ellip"] :
        z, p, k = signal.iirfilter(order, wn, btype=ftype, ftype=btype, output="zpk", rp=rps[0], rs=rps[1])
        try:
            sos = signal.zpk2sos(z, p, k)
            ite_data = signal.sosfilt(sos, ite_data)
            if zerophase:
                ite_data = signal.sosfilt(sos, ite_data[::-1])[::-1]
        except:
            print('A filter had an issue: ','btype', btype,'ftype', ftype,'order', order ,'Quality', Quality , 'window',window ,'lowcut', lowcut ,'highcut', highcut  , 'rps',rps  )
    elif btype == "iirnotch":
        b, a = signal.iirnotch(lowcut, Quality, fs)
        y = signal.filtfilt(b, a, ite_data)
    elif btype == "Moving average":
        z2 = np.cumsum(np.pad(ite_data, ((window, 0) ), 'constant', constant_values=0), axis=0)
        z1 = np.cumsum(np.pad(ite_data, ((0, window) ), 'constant', constant_values=ite_data[-1]), axis=0)
        ite_data= (z1 - z2)[(window - 1):-1] / window
    elif btype == 'DC':
        ite_data = ite_data - np.median(ite_data)
    return ite_data

def signalfilterbandpass_EEG(Sigs_dict,fs,Filter_info):
    N = len(Sigs_dict[list(Sigs_dict.keys())[0]])
    btype = Filter_info[0]
    ftype = Filter_info[1]
    order = Filter_info[2]
    Quality = Filter_info[3]
    window= Filter_info[4]
    lowcut= Filter_info[5]
    highcut= Filter_info[6]
    rps = [Filter_info[7],Filter_info[8]]

    if not order == '':
        if order >10:
            order=10
        if order <0:
            order=0
    if not lowcut == '':
        if lowcut <=0:
            lowcut=1/fs
    if not highcut == '':
        if highcut >= fs/2:
            highcut = fs/2-1

    if not window == '':
        window = int(window*fs)
        if window <= 0:
            window = 1
        elif window > N:
            window = N
    for idx_lfp, key in enumerate(list(Sigs_dict.keys())):
        Sigs_dict[key] = iir_band_filter_EEG(Sigs_dict[key], fs, btype, ftype, order=order, Quality=Quality , window=window, lowcut=lowcut, highcut=highcut, zerophase=0, rps=rps)
    return Sigs_dict

class Filter_Management(QDialog):
    def __init__(self, parent=None):
        super(Filter_Management, self).__init__()
        self.parent = parent

        self.Filter_list = self.parent.Filter_list
        self.Fs = int(1/(self.parent.t[1]-self.parent.t[0]))

        self.layoutparam = QVBoxLayout()

        Label = QLabel('Choose a filter')
        self.layoutparam.addWidget(Label)

        layout_Filter = QHBoxLayout()
        grid = QGridLayout()
        layout_Filter.addLayout(grid)
        cnt =0
        self.CB_design = QComboBox()
        label = ['Moving average','butter','cheby1','cheby2','ellip','iirnotch','DC']
        self.CB_design.addItems(label)
        grid.addWidget(QLabel('Filter design'),0, cnt)
        grid.addWidget(self.CB_design,1, cnt)
        self.CB_design.currentIndexChanged.connect(self.fdesign_change)
        cnt += 1

        self.CB_type = QComboBox()
        label = ['lowpass', 'highpass', 'bandpass', 'bandstop']
        self.CB_type.addItems(label)
        grid.addWidget(QLabel('Filter type'),0, cnt)
        grid.addWidget(self.CB_type,1, cnt)
        self.CB_type.currentIndexChanged.connect(self.ftype_change)
        cnt += 1

        label = QLabel('order')
        self.e_order = LineEdit_Int('1')
        grid.addWidget(label,0, cnt)
        grid.addWidget(self.e_order,1, cnt)
        cnt += 1

        label = QLabel('Quality f (dB)')
        self.e_Quality = LineEdit('30')
        grid.addWidget(label,0, cnt)
        grid.addWidget(self.e_Quality,1, cnt)
        cnt += 1

        label = QLabel('Window (ms)')
        self.e_Window = LineEdit('1')
        grid.addWidget(label,0, cnt)
        grid.addWidget(self.e_Window,1, cnt)
        cnt += 1

        label = QLabel('Lowcut/fc (kHz)')
        self.e_Lowcut = LineEdit('1')
        grid.addWidget(label,0, cnt)
        grid.addWidget(self.e_Lowcut,1, cnt)
        cnt += 1

        label = QLabel('Highcut (kHz)')
        self.e_Highcut = LineEdit(str(self.Fs//2))
        grid.addWidget(label,0, cnt)
        grid.addWidget(self.e_Highcut,1, cnt)
        cnt += 1

        label = QLabel('Ripple min (dB)')
        self.e_rs = LineEdit('1')
        grid.addWidget(label,0, cnt)
        grid.addWidget(self.e_rs,1, cnt)
        cnt += 1

        label = QLabel('Ripple max (dB)')
        self.e_rp = LineEdit('2')
        grid.addWidget(label,0, cnt)
        grid.addWidget(self.e_rp,1, cnt)
        cnt += 1


        self.Add = QPushButton('Add')
        self.Add.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.Add.clicked.connect(self.Add_but)


        layout_Filter.addWidget(self.Add)

        self.layoutparam.addLayout(layout_Filter)

        self.tableWidget = QTableWidget()
        self.tableWidget.setMinimumHeight(300)
        self.tableWidget.setContextMenuPolicy(Qt.ActionsContextMenu)
        self.menu = QMenu(self)
        Reuse = QAction('Reuse', self)
        Reuse.triggered.connect(self.ReuseLine)
        Up = QAction('Up', self)
        Up.triggered.connect(self.UpLine)
        Down = QAction('Down', self)
        Down.triggered.connect(self.DownLine)
        REM = QAction('Remove Line', self)
        REM.triggered.connect(self.RemLine)
        self.tableWidget.addAction(Reuse)
        self.tableWidget.addAction(Up)
        self.tableWidget.addAction(Down)
        self.tableWidget.addAction(REM)
        self.update_Table()


        self.layoutparam.addWidget(self.tableWidget)

        self.buttons = QDialogButtonBox( QDialogButtonBox.Ok , Qt.Horizontal, self)
        self.buttons.accepted.connect(self.accept)
        self.layoutparam.addWidget(self.buttons)

        self.setLayout(self.layoutparam)

        self.CB_design.setCurrentIndex(1)

    def ReuseLine(self):
        raw = self.tableWidget.currentRow() - 1
        Filter = self.Filter_list[raw]
        self.CB_design.setCurrentIndex(self.CB_design.findText(Filter[0]))
        self.CB_type.setCurrentIndex(self.CB_type.findText(Filter[1]))
        self.e_order.setText(str(Filter[2]))
        self.e_Quality.setText(str(Filter[3]))
        self.e_Window.setText(str(Filter[4]))
        self.e_Lowcut.setText(str(Filter[5]))
        self.e_Highcut.setText(str(Filter[6]))
        self.e_rs.setText(str(Filter[7]))
        self.e_rp.setText(str(Filter[8]))


    def UpLine(self, event):
        raw = self.tableWidget.currentRow()-1
        print(raw,raw+1)
        try:
            if raw >=1:
                self.Filter_list[raw], self.Filter_list[raw-1] = self.Filter_list[raw-1], self.Filter_list[raw]
        except:
            pass
        self.update_Table()

    def DownLine(self, event):
        raw = self.tableWidget.currentRow()-1
        print(raw,raw+1)
        try:
            self.Filter_list[raw], self.Filter_list[raw+1] = self.Filter_list[raw+1], self.Filter_list[raw]
        except:
            pass
        self.update_Table()

    def RemLine(self, event):
        raw = self.tableWidget.currentRow()-1
        try:
            self.Filter_list.pop(raw)
        except:
            pass
        self.update_Table()


    def fdesign_change(self):
        text = self.CB_design.currentText()
        if text in ['Moving average']:
            self.CB_type.setEnabled(False)
            self.e_order.setEnabled(False)
            self.e_Quality.setEnabled(False)
            self.e_Window.setEnabled(True)
            self.e_Lowcut.setEnabled(False)
            self.e_Highcut.setEnabled(False)
            self.e_rs.setEnabled(False)
            self.e_rp.setEnabled(False)
        elif text in ['butter']:
            self.CB_type.setEnabled(True)
            self.e_order.setEnabled(True)
            self.e_Quality.setEnabled(False)
            self.e_Window.setEnabled(False)
            if self.CB_type.currentText() in ['lowpass']:
                self.e_Lowcut.setEnabled(True)
                self.e_Highcut.setEnabled(False)
            elif self.CB_type.currentText() in ['highpass']:
                self.e_Lowcut.setEnabled(False)
                self.e_Highcut.setEnabled(True)
            else:
                self.e_Lowcut.setEnabled(True)
                self.e_Highcut.setEnabled(True)
            self.e_rs.setEnabled(False)
            self.e_rp.setEnabled(False)
        elif text in ['cheby1']:
            self.CB_type.setEnabled(True)
            self.e_order.setEnabled(True)
            self.e_Quality.setEnabled(False)
            self.e_Window.setEnabled(False)
            if self.CB_type.currentText() in ['lowpass']:
                self.e_Lowcut.setEnabled(True)
                self.e_Highcut.setEnabled(False)
            elif self.CB_type.currentText() in ['highpass']:
                self.e_Lowcut.setEnabled(True)
                self.e_Highcut.setEnabled(True)
            elif self.CB_type.currentText() in ['bandpass']:
                self.e_Lowcut.setEnabled(True)
                self.e_Highcut.setEnabled(True)
            elif self.CB_type.currentText() in ['bandstop']:
                self.e_Lowcut.setEnabled(True)
                self.e_Highcut.setEnabled(True)
                self.e_rp.setEnabled(False)
            self.e_rs.setEnabled(True)
            self.e_rp.setEnabled(True)
        elif text in ['cheby2']:
            self.CB_type.setEnabled(True)
            self.e_order.setEnabled(True)
            self.e_Quality.setEnabled(False)
            self.e_Window.setEnabled(False)
            if self.CB_type.currentText() in ['lowpass']:
                self.e_Lowcut.setEnabled(True)
                self.e_Highcut.setEnabled(False)
            elif self.CB_type.currentText() in ['highpass']:
                self.e_Lowcut.setEnabled(True)
                self.e_Highcut.setEnabled(True)
            elif self.CB_type.currentText() in ['bandpass']:
                self.e_Lowcut.setEnabled(True)
                self.e_Highcut.setEnabled(True)
            elif self.CB_type.currentText() in ['bandstop']:
                self.e_Lowcut.setEnabled(True)
                self.e_Highcut.setEnabled(True)
            self.e_rs.setEnabled(True)
            self.e_rp.setEnabled(True)
        elif text in ['ellip']:
            self.CB_type.setEnabled(True)
            self.e_order.setEnabled(True)
            self.e_Quality.setEnabled(False)
            self.e_Window.setEnabled(False)
            if self.CB_type.currentText() in ['lowpass']:
                self.e_Lowcut.setEnabled(True)
                self.e_Highcut.setEnabled(False)
            elif self.CB_type.currentText() in ['highpass']:
                self.e_Lowcut.setEnabled(True)
                self.e_Highcut.setEnabled(True)
            elif self.CB_type.currentText() in ['bandpass']:
                self.e_Lowcut.setEnabled(True)
                self.e_Highcut.setEnabled(True)
            elif self.CB_type.currentText() in ['bandstop']:
                self.e_Lowcut.setEnabled(True)
                self.e_Highcut.setEnabled(True)
            self.e_rs.setEnabled(True)
            self.e_rp.setEnabled(True)
        elif text in ['iirnotch']:
            self.CB_type.setEnabled(False)
            self.e_order.setEnabled(False)
            self.e_Quality.setEnabled(True)
            self.e_Window.setEnabled(False)
            self.e_Lowcut.setEnabled(True)
            self.e_Highcut.setEnabled(False)
            self.e_rs.setEnabled(False)
            self.e_rp.setEnabled(False)
        elif text in ['DC']:
            self.CB_type.setEnabled(False)
            self.e_order.setEnabled(False)
            self.e_Quality.setEnabled(False)
            self.e_Window.setEnabled(False)
            self.e_Lowcut.setEnabled(False)
            self.e_Highcut.setEnabled(False)
            self.e_rs.setEnabled(False)
            self.e_rp.setEnabled(False)

    def ftype_change(self):
        text = self.CB_type.currentText()
        if text in ['lowpass']:
            self.e_Lowcut.setEnabled(True)
            self.e_Highcut.setEnabled(False)
            self.e_rp.setEnabled(False)
            self.e_rs.setEnabled(False)
        elif text in ['highpass']:
            self.e_Lowcut.setEnabled(False)
            self.e_Highcut.setEnabled(True)
            self.e_rp.setEnabled(False)
            self.e_rs.setEnabled(False)
        elif text in ['bandpass']:
            self.e_Lowcut.setEnabled(True)
            self.e_Highcut.setEnabled(True)
            if self.CB_design.currentText() in ['cheby1','cheby2','ellip']:
                self.e_rp.setEnabled(True)
                self.e_rs.setEnabled(True)
        elif text in ['bandstop']:
            self.e_Lowcut.setEnabled(True)
            self.e_Highcut.setEnabled(True)
            if self.CB_design.currentText() in ['cheby1','cheby2','ellip']:
                self.e_rp.setEnabled(True)
                self.e_rs.setEnabled(True)

    def Add_but(self):
        design = self.CB_design.currentText()
        if self.CB_type.isEnabled():
            type = self.CB_type.currentText()
        else:
            type = ''
        if self.e_order.isEnabled():
            order = int(self.e_order.text())
        else:
            order = ''
        if self.e_Quality.isEnabled():
            Quality = float(self.e_Quality.text())
        else:
            Quality = ''
        if self.e_Window.isEnabled():
            Window = float(self.e_Window.text())
        else:
            Window = ''
        if self.e_Lowcut.isEnabled():
            Lowcut = float(self.e_Lowcut.text())
        else:
            Lowcut = ''
        if self.e_Highcut.isEnabled():
            Highcut = float(self.e_Highcut.text())
        else:
            Highcut = ''
        if self.e_rp.isEnabled():
            rp = float(self.e_rp.text())
        else:
            rp = ''
        if self.e_rs.isEnabled():
            rs = float(self.e_rs.text())
        else:
            rs = ''

        self.Filter_list.append([design, type, order, Quality, Window, Lowcut, Highcut, rp, rs])
        self.update_Table()

    def update_Table(self):
        numberline = len(self.Filter_list)
        if numberline<100:
            numberline=100
        self.tableWidget.setRowCount(0)
        self.tableWidget.setRowCount(numberline+1)
        self.tableWidget.setColumnCount(9)
        [self.tableWidget.setColumnWidth(i,100) for i in range(9)]
        self.tableWidget.setItem(0,0, QTableWidgetItem("Design"))
        self.tableWidget.setItem(0,1, QTableWidgetItem("Type"))
        self.tableWidget.setItem(0,2, QTableWidgetItem("Order"))
        self.tableWidget.setItem(0,3, QTableWidgetItem("Quality F"))
        self.tableWidget.setItem(0,4, QTableWidgetItem("Window"))
        self.tableWidget.setItem(0,5, QTableWidgetItem("Lowcut/fc"))
        self.tableWidget.setItem(0,6, QTableWidgetItem("Highcut"))
        self.tableWidget.setItem(0,7, QTableWidgetItem("Ripple min"))
        self.tableWidget.setItem(0,8, QTableWidgetItem("Ripple max"))


        for i,line in enumerate(self.Filter_list):
            for j,val in enumerate(line):
                self.tableWidget.setItem(i+1,j, QTableWidgetItem(str(val)))



    def apply_fun(self):
        Colors = self.colorbutton.palette().button().color().name()
        for i, cb in enumerate(self.CBs):
            if cb.isChecked():
                set_QPushButton_background_color(self.col_but[i], QColor(Colors))

    def selectALL(self):
        for cb in self.CBs:
            cb.setChecked(True)

    def clearALL(self):
        for cb in self.CBs:
            cb.setChecked(False)


class MySlider(QScrollBar):
    text_pre =''
    text_post =''
    text_slider =''

    def paintEvent(self, event):
        # call the base class paintEvent, which will draw the scrollbar
        super().paintEvent(event)

        # create a suitable styleoption and "init" it to this instance
        option = QStyleOptionSlider()
        self.initStyleOption(option)

        painter = QPainter(self)

        # get the "subPage" rectangle and draw the text
        subPageRect = self.style().subControlRect(QStyle.CC_ScrollBar, option, QStyle.SC_ScrollBarSubPage, self)
        painter.drawText(subPageRect, Qt.AlignLeft|Qt.AlignVCenter, self.text_pre)

        # get the "addPage" rectangle and draw its text
        addPageRect = self.style().subControlRect(QStyle.CC_ScrollBar, option, QStyle.SC_ScrollBarAddPage, self)
        painter.drawText(addPageRect, Qt.AlignRight|Qt.AlignVCenter, self.text_post)

        # the same, for the slider
        sliderRect = self.style().subControlRect(QStyle.CC_ScrollBar, option, QStyle.SC_ScrollBarSlider, self)
        painter.drawText(sliderRect, Qt.AlignCenter, self.text_slider)

class TextScrollBarStyle(QProxyStyle):
    def drawComplexControl(self, control, option, painter, widget):
        # call the base implementation which will draw anything Qt will ask
        super().drawComplexControl(control, option, painter, widget)
        # check if the control type is that of a scroll bar and its orientation matches
        if control == QStyle.CC_ScrollBar and option.orientation == Qt.Horizontal:
            # the option is already provided by the widget's internal paintEvent;
            # from this point on, it's almost the same as explained above, but
            # setting the pen might be required for some styles
            painter.setPen(widget.palette().color(QPalette.WindowText))
            margin = self.frameMargin(widget)
            subPageRect = self.subControlRect(control, option, QStyle.SC_ScrollBarSubPage, widget)
            painter.drawText(subPageRect.adjusted(margin, 0, 0, 0), Qt.AlignLeft|Qt.AlignVCenter, widget.preText)

            addPageRect = self.subControlRect(control, option, QStyle.SC_ScrollBarAddPage, widget)
            painter.drawText(addPageRect.adjusted(0, 0, -margin, 0), Qt.AlignRight|Qt.AlignVCenter, widget.postText)

            sliderRect = self.subControlRect(control, option, QStyle.SC_ScrollBarSlider, widget)
            painter.drawText(sliderRect, Qt.AlignCenter, widget.sliderText)

    def frameMargin(self, widget):
        # a helper function to get the default frame margin which is usually added
        # to widgets and sub widgets that might look like a frame, which usually
        # includes the slider of a scrollbar
        option = QStyleOptionFrame()
        option.initFrom(widget)
        return self.pixelMetric(QStyle.PM_DefaultFrameWidth, option, widget)

    def subControlRect(self, control, option, subControl, widget):
        rect = super().subControlRect(control, option, subControl, widget)
        if (control == QStyle.CC_ScrollBar and isinstance(widget, StyledTextScrollBar) and
            option.orientation == Qt.Horizontal and subControl == QStyle.SC_ScrollBarSlider):
                # get the *default* groove rectangle (the space in which the slider can move)
                grooveRect = super().subControlRect(control, option, QStyle.SC_ScrollBarGroove, widget)
                # if the slider has text, ensure that the width is large enough to show it
                width = max(rect.width(), widget.sliderWidth + self.frameMargin(widget))
                # compute the position of the slider according to the current value
                pos = self.sliderPositionFromValue(widget.minimum(), widget.maximum(),
                    widget.sliderPosition(), grooveRect.width() - width)
                # return the new rectangle
                return QRect(grooveRect.x() + pos, rect.y(), width, rect.height())
        return rect

    def hitTestComplexControl(self, control, option, pos, widget):
        if control == QStyle.CC_ScrollBar:
            sliderRect = self.subControlRect(control, option, QStyle.SC_ScrollBarSlider, widget)
            if pos in sliderRect:
                return QStyle.SC_ScrollBarSlider
        return super().hitTestComplexControl(control, option, pos, widget)

class StyledTextScrollBar(QScrollBar):
    def __init__(self, sliderText='', preText='', postText=''):
        super().__init__(Qt.Horizontal)
        self.setStyle(TextScrollBarStyle())
        self.preText = preText
        self.postText = postText
        self.sliderText = sliderText
        self.sliderTextMargin = 2
        self.sliderWidth = self.fontMetrics().width(sliderText) + self.sliderTextMargin + 2

    def setPreText(self, text):
        self.preText = text
        self.update()

    def setPostText(self, text):
        self.postText = text
        self.update()

    def setSliderText(self, text):
        self.sliderText = text
        self.sliderWidth = self.fontMetrics().width(text) + self.sliderTextMargin + 2

    def setSliderTextMargin(self, margin):
        self.sliderTextMargin = margin
        self.sliderWidth = self.fontMetrics().width(self.sliderText) + margin + 2
