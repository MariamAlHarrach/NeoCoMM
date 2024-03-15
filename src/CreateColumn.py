__author__ = 'Mariam, Maxime'
#This function creates the cortical column cell network by positionning the cells inside the defined volume using the best candidate algorithm

import numpy as np
import math
from scipy.spatial import distance



def PlaceCell_func(L,Layer_d,D,Layer_nbCells, type='Cylinder', seed = 0, C=2000): 
    if not seed == 0:
        np.random.seed(seed)
    def pol2cart(rho, phi):
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return (x, y)

    def polar2cart3D(r, theta, phi):
        return [
            r * math.sin(theta) * math.cos(phi),
            r * math.sin(theta) * math.sin(phi),
            r * math.cos(theta)
        ]

    def asCartesian(r, theta, phi):
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return [x, y, z]
    Cellpositionall =[]

    Layer_d_cumsum = np.cumsum(Layer_d)
    Layer_d_cumsum = np.hstack((0,Layer_d_cumsum))
    print('Place cells....')
    if type == 'Cylinder':
        for l in range(len(Layer_nbCells)):
            CellPosition=[]

            module = float(np.random.uniform(low=0, high=1, size=1))
            phi = float(np.random.uniform(low=0, high=2*np. pi, size=1))
            x0, y0 = pol2cart(module*D/2, phi)
            z0 = float(np.random.uniform(low=L - Layer_d_cumsum[l], high=L - Layer_d_cumsum[l+1], size=1))
            CellPosition.append(np.array([x0,y0,z0] ))
            # print(CellPosition)
            for nb in range(int(Layer_nbCells[l])-1):
                candidate=[]
                for k_i in range(20):
                    module = float(np.random.uniform(low=0, high=1, size=1))
                    phi = float(np.random.uniform(low=0, high=2 * np.pi, size=1))
                    x0, y0 = pol2cart(module*D /2, phi )
                    z0 = float(np.random.uniform(low=L - Layer_d_cumsum[l], high=L - Layer_d_cumsum[l + 1], size=1))
                    candidate.append(np.array([x0, y0, z0]))
                candidate = np.array(candidate)

                # print(candidate)
                CellDistances = distance.cdist(CellPosition, candidate, 'euclidean')
                # print(CellDistances)
                argmin = np.argmin(CellDistances, axis=0)
                valmin = [CellDistances[k, j] for j, k in enumerate(argmin)]
                argmax = np.argmax(valmin)
                CellPosition.append(candidate[argmax, :])

            CellPosition=np.array(CellPosition)
            marange = np.arange(CellPosition.shape[0])
            np.random.shuffle(marange)
            CellPosition2 = CellPosition[marange]
            Cellpositionall.append(CellPosition2)
    elif type == 'Cylinder with curvature':
        CurvatureD = C
        for l in range(len(Layer_nbCells)):
            CellPosition=[]
            PYRCellPosition=[]
            IntCellPosition=[]
            module = float(np.random.uniform(low=0, high=1, size=1))
            phi = float(np.random.uniform(low=0, high=2*np.pi , size=1))
            x0, y0 = pol2cart(module*D /2, phi )

            zL = CurvatureD + L
            c = np.sqrt(x0 * x0 + y0 * y0)
            thetaX= np.arctan(x0/zL)
            thetaY= np.arctan(y0/zL)

            zmin = CurvatureD + Layer_d_cumsum[-1] - Layer_d_cumsum[l+1]
            zmax = CurvatureD + Layer_d_cumsum[-1] - Layer_d_cumsum[l]
            module = float(np.random.uniform(low=zmin, high=zmax, size=1))


            x0, y0, z0 = asCartesian(module, thetaX, float(np.random.uniform(low=0, high=2*np.pi , size=1)))
            z0 = z0 - CurvatureD

            CellPosition.append(np.array([x0,y0,z0] ))
            # print(CellPosition)
            for nb in range(int(Layer_nbCells[l])-1):

                candidate=[]
                for k_i in range(20):
                    module = float(np.random.uniform(low=0, high=1, size=1))
                    phi = float(np.random.uniform(low=0, high=2 * np.pi, size=1))
                    x0, y0 = pol2cart(module * D / 2, phi)

                    zL = CurvatureD + L
                    c = np.sqrt(x0 * x0 + y0 * y0)
                    thetaX = np.arctan(x0 / zL)
                    thetaY = np.arctan(y0 / zL)

                    zmin = CurvatureD + Layer_d_cumsum[-1] - Layer_d_cumsum[l + 1]
                    zmax = CurvatureD + Layer_d_cumsum[-1] - Layer_d_cumsum[l]
                    module = float(np.random.uniform(low=zmin, high=zmax, size=1))


                    x0, y0, z0 = asCartesian(module, thetaX, float(np.random.uniform(low=0, high=2*np.pi , size=1)))
                    z0 = z0 - CurvatureD

                    candidate.append(np.array([x0, y0, z0]))
                candidate = np.array(candidate)

                # print(candidate)
                CellDistances = distance.cdist(CellPosition, candidate, 'euclidean')
                # print(CellDistances)
                argmin = np.argmin(CellDistances, axis=0)
                valmin = [CellDistances[k, j] for j, k in enumerate(argmin)]
                argmax = np.argmax(valmin)
                CellPosition.append(candidate[argmax, :])
                # if nb<self. NB_PYR[l]-1:
                #     PYRCellPosition.append(candidate[argmax, :])
                # else:
                #     IntCellPosition.append(candidate[argmax, :])
            CellPosition=np.array(CellPosition)
            marange = np.arange(CellPosition.shape[0])
            np.random.shuffle(marange)
            CellPosition2 = CellPosition[marange]
            Cellpositionall.append(CellPosition2)
    elif type == 'Square':
        for l in range(len(Layer_nbCells)):
            CellPosition=[]
            PYRCellPosition=[]
            IntCellPosition=[]

            # phi = float(np.random.uniform(low=0, high=2*np.pi, size=1))
            x0 = float(np.random.uniform(low=-D/2, high=D/2, size=1))
            y0 = float(np.random.uniform(low=-D/2, high=D/2, size=1))
            z0 = float(np.random.uniform(low=L - Layer_d_cumsum[l], high=L - Layer_d_cumsum[l+1], size=1))
            CellPosition.append(np.array([x0,y0,z0] ))
            for nb in range(int(Layer_nbCells[l])-1):
                candidate=[]
                for k_i in range(20):
                    module = float(np.random.uniform(low=0, high=1, size=1))
                    phi = float(np.random.uniform(low=0, high=2 * np.pi, size=1))

                    x0 = float(np.random.uniform(low=-D/2, high=D/2, size=1))
                    y0 = float(np.random.uniform(low=-D/2, high=D/2, size=1))
                    z0 = float(np.random.uniform(low=L - Layer_d_cumsum[l], high=L - Layer_d_cumsum[l + 1], size=1))
                    candidate.append(np.array([x0, y0, z0]))
                candidate = np.array(candidate)

                # print(candidate)
                CellDistances = distance.cdist(CellPosition, candidate, 'euclidean')
                # print(CellDistances)
                argmin = np.argmin(CellDistances, axis=0)
                valmin = [CellDistances[k, j] for j, k in enumerate(argmin)]
                argmax = np.argmax(valmin)
                CellPosition.append(candidate[argmax, :])

            CellPosition=np.array(CellPosition)
            marange = np.arange(CellPosition.shape[0])
            np.random.shuffle(marange)
            CellPosition2 = CellPosition[marange]

            Cellpositionall.append(CellPosition2)
    elif type == 'Square with curvature':

        CurvatureD = C
        for l in range(len(Layer_nbCells)):
            CellPosition=[]
            PYRCellPosition=[]
            IntCellPosition=[]

            x0 = float(np.random.uniform(low=-D/2, high=D/2, size=1))
            y0 = float(np.random.uniform(low=-D/2, high=D/2, size=1))

            # zL = CurvatureD + L
            zmin = CurvatureD + Layer_d_cumsum[-1] - Layer_d_cumsum[l+1]
            zmax = CurvatureD + Layer_d_cumsum[-1] - Layer_d_cumsum[l]
            c = np.sqrt(x0 * x0 + y0 * y0)
            thetamin = np.arctan(c/zmin)
            thetamax = np.arctan(c/zmax)

            Zmin = (CurvatureD + Layer_d_cumsum[-1] - Layer_d_cumsum[l+1]) * np.cos(thetamin)
            Zmax = (CurvatureD + Layer_d_cumsum[-1] - Layer_d_cumsum[l]) * np.cos(thetamax)

            z0 = float(np.random.uniform(low=Zmin-CurvatureD, high=Zmax-CurvatureD, size=1))
            CellPosition.append(np.array([x0,y0,z0]) )
            for nb in range(int(Layer_nbCells[l])-1):
                candidate=[]
                for k_i in range(20):
                    module = float(np.random.uniform(low=0, high=1, size=1))
                    phi = float(np.random.uniform(low=0, high=2 * np.pi, size=1))

                    x0 = float(np.random.uniform(low=-D/2, high =D/2, size =1))
                    y0 = float(np.random.uniform(low=-D/2, high =D/2, size =1))
                    zmin = CurvatureD + Layer_d_cumsum[-1] - Layer_d_cumsum[l + 1]
                    zmax = CurvatureD + Layer_d_cumsum[-1] - Layer_d_cumsum[l]
                    c = np.sqrt(x0 * x0 + y0 * y0)
                    thetamin = np.arctan(c / zmin)
                    thetamax = np.arctan(c / zmax)

                    Zmin = (CurvatureD + Layer_d_cumsum[-1] - Layer_d_cumsum[l + 1]) * np.cos(thetamin)
                    Zmax = (CurvatureD + Layer_d_cumsum[-1] - Layer_d_cumsum[l]) * np.cos(thetamax)

                    z0 = float(np.random.uniform(low=Zmin - CurvatureD, high=Zmax - CurvatureD, size=1))
                    candidate.append(np.array([x0, y0, z0]))
                candidate = np.array(candidate)

                # print(candidate)
                CellDistances = distance.cdist(CellPosition, candidate, 'euclidean')
                # print(CellDistances)
                argmin = np.argmin(CellDistances, axis=0)
                valmin = [CellDistances[k, j] for j, k in enumerate(argmin)]
                argmax = np.argmax(valmin)
                CellPosition.append(candidate[argmax, :])

            CellPosition=np.array(CellPosition)
            marange = np.arange(CellPosition.shape[0])
            np.random.shuffle(marange)
            CellPosition2 = CellPosition[marange]

            Cellpositionall.append(CellPosition2)

    elif type == 'Rectange':
        for l in range(len(Layer_nbCells)):
            CellPosition=[]
            PYRCellPosition=[]
            IntCellPosition=[]

            x0 = float(np.random.uniform(low=-D/2, high =D/2, size =1))
            y0 = float(np.random.uniform(low=-L/2, high =L/2, size =1))
            z0 = float(np.random.uniform(low=L - Layer_d_cumsum[l], high=L - Layer_d_cumsum[l+1], size=1))
            CellPosition.append(np.array([x0,y0,z0]) )
            for nb in range(int(Layer_nbCells[l])-1):
                candidate=[]
                for k_i in range(20):
                    module = float(np.random.uniform(low=0, high=1, size=1))
                    phi = float(np.random.uniform(low=0, high=2 * np.pi, size=1))

                    x0 = float(np.random.uniform(low=-D/2, high =D/2, size =1))
                    y0 = float(np.random.uniform(low=-L/2, high =L/2, size =1))
                    z0 = float(np.random.uniform(low=L - Layer_d_cumsum[l], high=L - Layer_d_cumsum[l + 1], size=1))
                    candidate.append(np.array([x0, y0, z0]))
                candidate = np.array(candidate)

                CellDistances = distance.cdist(CellPosition, candidate, 'euclidean')
                argmin = np.argmin(CellDistances, axis=0)
                valmin = [CellDistances[k, j] for j, k in enumerate(argmin)]
                argmax = np.argmax(valmin)
                CellPosition.append(candidate[argmax, :])

            CellPosition=np.array(CellPosition)
            marange = np.arange(CellPosition.shape[0])
            np.random.shuffle(marange)
            CellPosition2 = CellPosition[marange]
            Cellpositionall.append(CellPosition2)
    elif type == 'Rectange with curvature':

        CurvatureD = C
        for l in range(len(Layer_nbCells)):
            CellPosition=[]
            PYRCellPosition=[]
            IntCellPosition=[]

            x0 = float(np.random.uniform(low=-D/2, high =D/2, size =1))
            y0 = float(np.random.uniform(low=-L/2, high =L/2, size =1))
            zmin = CurvatureD + Layer_d_cumsum[-1] - Layer_d_cumsum[l+1]
            zmax = CurvatureD + Layer_d_cumsum[-1] - Layer_d_cumsum[l]
            c = np.sqrt(x0 * x0 + y0 * y0)
            thetamin = np.arctan(c/zmin)
            thetamax = np.arctan(c/zmax)

            Zmin = (zmin) * np.cos(thetamin)
            Zmax = (zmax) * np.cos(thetamax)

            z0 = float(np.random.uniform(low=Zmin-CurvatureD, high=Zmax-CurvatureD, size=1))

            CellPosition.append(np.array([x0,y0,z0]) )
            for nb in range(int(Layer_nbCells[l])-1):
                candidate=[]
                for k_i in range(20):
                    module = float(np.random.uniform(low=0, high=1, size=1))
                    phi = float(np.random.uniform(low=0, high=2 * np.pi, size=1))

                    x0 = float(np.random.uniform(low=-D/2, high =L/2, size =1))
                    y0 = float(np.random.uniform(low=-L/2, high =L/2, size =1))

                    # zL = CurvatureD + L
                    zmin = CurvatureD + Layer_d_cumsum[-1] - Layer_d_cumsum[l + 1]
                    zmax = CurvatureD + Layer_d_cumsum[-1] - Layer_d_cumsum[l]
                    c = np.sqrt(x0 * x0 + y0 * y0)
                    thetamin = np.arctan(c / zmin)
                    thetamax = np.arctan(c / zmax)

                    Zmin = (zmin) * np.cos(thetamin)
                    Zmax = (zmax) * np.cos(thetamax)

                    z0 = float(np.random.uniform(low=Zmin - CurvatureD, high=Zmax - CurvatureD, size=1))
                    candidate.append(np.array([x0, y0, z0]))
                candidate = np.array(candidate)

                CellDistances = distance.cdist(CellPosition, candidate, 'euclidean')
                argmin = np.argmin(CellDistances, axis=0)
                valmin = [CellDistances[k, j] for j, k in enumerate(argmin)]
                argmax = np.argmax(valmin)
                CellPosition.append(candidate[argmax, :])

            CellPosition=np.array(CellPosition)
            marange = np.arange(CellPosition.shape[0])
            np.random.shuffle(marange)
            CellPosition2 = CellPosition[marange]

            Cellpositionall.append(CellPosition2)
    Cellpos=np.array(Cellpositionall)
    return Cellpos
