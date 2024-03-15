__author__ = 'Maxime'
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

def Layout_groupbox_Label_Edit(labelGroup ='None', label = ['None'],edit =['None'] , name =None, color= None, width = 100 , height_add =0, height_per_line=20 ):
    layout = QGroupBox(labelGroup)
    if not width == None:
        layout.setFixedWidth(width)
    layout.setAlignment( Qt.AlignTop)
    layout_range = QVBoxLayout()
    # layout.setFixedHeight(  height_per_line* len(label))
    grid = QGridLayout()
    grid.setAlignment(Qt.AlignTop)
    layout_range.addLayout(grid)
    Edit_List =[]

    Label = QLabel('Name')
    Label2 = QLineEdit(name)
    colorbutton = QPushButton('')
    set_QPushButton_background_color(colorbutton, QColor(color))
    colorbutton.clicked.connect(lambda state, x=colorbutton: label_color_clicked(state, x))
    grid.addWidget(Label, 0, 0,1,2)
    grid.addWidget(Label2, 0, 2,1,2)
    grid.addWidget(colorbutton, 0, 4,1,2)
    Edit_List.append(Label2)
    Edit_List.append(colorbutton)
    for idx in range(len(label)):
        Label = QLabel(label[idx])
        Label.setFixedHeight(height_per_line)
        Edit = QLineEdit(edit[idx])
        Edit.setFixedHeight(height_per_line)
        grid.addWidget(Label, idx+2, 0, 1, 3)
        grid.addWidget(Edit, idx+2, 3, 1, 3)
        Edit_List.append(Edit)
    layout.setLayout(layout_range)
    return layout, Edit_List

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


class Modify_1_NMM(QMainWindow):
    Mod_OBJ = pyqtSignal(list,list,list)
    Close_OBJ = pyqtSignal()
    def __init__(self, parent=None,Dict_Param = [],List_Names=[],List_Color=[]):
        super(Modify_1_NMM, self).__init__()
        self.isclosed = False
        self.Dict_Param = Dict_Param
        self.List_Color = List_Color
        self.List_Names = List_Names

        ############variable utiles###########
        self.height_per_line = 20
        self.height_add = 30
        self.width_per_col =155
        self.width_label =100
        self.Heigtheach =  600
        #######################################
        self.centralWidget = QWidget()
        self.setCentralWidget(self.centralWidget)
        self.mainHBOX_param_scene = QHBoxLayout()
        self.setMinimumHeight(700)
        self.setMinimumWidth(800)

        # set Tabs
        self.set_param()
        self.mainHBOX_param_scene.addLayout(self.layoutmain)
        self.centralWidget.setLayout(self.mainHBOX_param_scene)

    def set_param(self):
        self.layoutmain = QVBoxLayout()

        #scrolling
        self.groupscrollmodelselectGB = QHBoxLayout()
        self.groupscrollmodelselect = QGroupBox('Populations')



        self.praram = QGroupBox('Populations')
        self.praram.setFixedHeight(self.Heigtheach)


        self.layout_setup = QHBoxLayout()
        self.edit_model =[]
        for idx_l, l in enumerate(self.Dict_Param):
            for idx_p, p in enumerate(l):
                edit =[]
                list_variable = []
                for key, value in p.items():
                    edit.append(str(value))
                    list_variable.append(key)
                layout_NMM_var, Edit_List_NMM_var = Layout_groupbox_Label_Edit(labelGroup ='', label = list_variable,edit =edit, name=self.List_Names[idx_l][idx_p], color=self.List_Color[idx_l][idx_p], width = 150)
                layout_NMM_var.setAlignment(Qt.AlignTop)
                self.edit_model.append(Edit_List_NMM_var)
                self.layout_setup.addWidget(layout_NMM_var)


        scroll = QScrollArea()
        widget = QWidget(self)
        widget.setLayout(QHBoxLayout())
        widget.layout().addWidget(self.groupscrollmodelselect)
        scroll.setWidget(widget)
        scroll.setWidgetResizable(True)
        self.groupscrollmodelselect.setLayout(self.layout_setup)
        self.groupscrollmodelselectGB.addWidget(scroll)

        # self.praram.setLayout(self.scroll)

        #action
        layout_Actions = QVBoxLayout()
        self.Apply = QPushButton('Apply')
        self.Apply.setFixedWidth(self.width_label)
        self.Apply.clicked.connect(self.Applyclick)
        layout_Actions.addWidget(self.Apply)
        layout_Actions.setAlignment(Qt.AlignRight)

        self.layoutmain.addLayout(self.groupscrollmodelselectGB)
        self.layoutmain.addLayout(layout_Actions)


    def Applyclick(self):
        popName = []
        popColor = []
        for idx, p in enumerate(self.Dict_Param):
            popName.append(self.edit_model[idx][0].text())
            popColor.append(self.edit_model[idx][1].palette().button().color().name())
            for idx_v,key in enumerate(self.Dict_Param[idx].keys()):
                self.Dict_Param[idx][key]=float(self.edit_model[idx][idx_v+2].text())
        self.Mod_OBJ.emit(self.Dict_Param,popName,popColor)

    def closeEvent(self, event):
        self.Close_OBJ.emit()
        self.isclosed = True
        self.close()



#
# def main():
#     app = QApplication(sys.argv)
#     pop = []
#     for idx in range(10):
#         pop.append(Model_Siouar.pop_Siouar())
#         pop[idx].random_seeded(10)
#
#     list_variable = Model_Siouar.get_Variable_Names()
#
#     ex = Modify_1_NMM(app,pop=pop,list_variable=list_variable)
#     ex.setWindowTitle('Create Connectivity Matrix')
#     # ex.showMaximized()
#     ex.show()
#     #ex.move(0, 0)
#     sys.exit(app.exec_( ))
#
#
# if __name__ == '__main__':
#     main()