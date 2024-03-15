__author__ = 'Mariam, Maxime'

import numpy as np
import math
import copy
from numba import vectorize, float64
import Cell_morphology


@vectorize([float64(float64, float64, float64, float64)])
def MeanMax(S_Z0, T_Z0_n, S_Z1, T_Z1_n):
    if S_Z0 >= T_Z0_n:
        Z0_m = S_Z0
    else:
        Z0_m =T_Z0_n

    if S_Z1 >= T_Z1_n:
        Z1_m = T_Z1_n
    else:
        Z1_m = S_Z1

    return (Z0_m + Z1_m) /2


def IsConnected(Source, Sourcepos, Target, Targetpos, d,Col_L, Layer_d):

    L=Col_L #column length
    step = 10.  # micrometeres
    connected = 0
    MeanPos = 0
    overlap = 0.0
    ConnPos = 0 #position of overlapping center
    ConnPos_nb = 0 #number of overlapping volumes
    ConnPossans = 0
    ConnPossans_nb = 0

    if Source.type in [0, 3, 4]:  # PC/ VIP or RLN
        S_Z0 = Sourcepos[2] + Source.AX_down  # lower threshold
        S_Z1 = Sourcepos[2] + Source.AX_up  # upper threshold

        if S_Z0 < 0:  # kepp axon in the cortical columnlength
            S_Z0 = 0.
        if S_Z1 > L:
            S_Z1 = L + 0.

        # if PC ---> PC ----> #dendrite targeting
        if Target.type == 0:  # PC
            if Target.subtype in [1, 2]:  # IPC or UPC
                # first cylinder
                T_Z0_n = Targetpos[2] + Target.c1_down
                T_Z1_n = Targetpos[2] + Target.c1_up
                o = find_intersection_high_vectorize(S_Z0, S_Z1, T_Z0_n, T_Z1_n, Source.AX_w, Target.r1, d)
                overlap += o
                if o > 0:
                    ConnPos += MeanMax(S_Z0, T_Z0_n, S_Z1, T_Z1_n)
                    ConnPos_nb += 1
                else:
                    ConnPossans += MeanMax(S_Z0, T_Z0_n, S_Z1, T_Z1_n)
                    ConnPossans_nb += 1
                #  second cylinder
                T_Z0_n = Targetpos[2] + Target.c2_down
                T_Z1_n = Targetpos[2] + Target.c2_up
                o = find_intersection_high_vectorize(S_Z0, S_Z1, T_Z0_n, T_Z1_n, Source.AX_w, Target.r2, d)
                overlap += o
                if o > 0:
                    ConnPos += MeanMax(S_Z0, T_Z0_n, S_Z1, T_Z1_n)
                    ConnPos_nb += 1
                else:
                    ConnPossans += MeanMax(S_Z0, T_Z0_n, S_Z1, T_Z1_n)
                    ConnPossans_nb += 1






            elif Target.subtype == 3:  # BPC
                # cylinder
                T_Z0_n = Targetpos[2] + Target.c1_down
                T_Z1_n = Targetpos[2] + Target.c1_up
                o = find_intersection_high_vectorize(S_Z0, S_Z1, T_Z0_n, T_Z1_n, Source.AX_w, Target.Adend_w, d)
                overlap += o
                if o > 0:
                    ConnPos += MeanMax(S_Z0, T_Z0_n, S_Z1, T_Z1_n)
                    ConnPos_nb += 1
                else:
                    ConnPossans += MeanMax(S_Z0, T_Z0_n, S_Z1, T_Z1_n)
                    ConnPossans_nb += 1

                # cylinder haut
                T_Z0_n = Targetpos[2] + Target.c2_down
                T_Z1_n = Targetpos[2] + Target.c2_up
                o = find_intersection_high_vectorize(S_Z0, S_Z1, T_Z0_n, T_Z1_n, Source.AX_w, 10, d)
                overlap += o
                if o > 0:
                    ConnPos += MeanMax(S_Z0, T_Z0_n, S_Z1, T_Z1_n)
                    ConnPos_nb += 1
                else:
                    ConnPossans += MeanMax(S_Z0, T_Z0_n, S_Z1, T_Z1_n)
                    ConnPossans_nb += 1

                # cylinder bas
                T_Z0_n = Targetpos[2] + Target.c3_down
                T_Z1_n = Targetpos[2] + Target.c3_up
                o = find_intersection_high_vectorize(S_Z0, S_Z1, T_Z0_n, T_Z1_n, Source.AX_w, 10, d)
                overlap += o
                if o > 0:
                    ConnPos += MeanMax(S_Z0, T_Z0_n, S_Z1, T_Z1_n)
                    ConnPos_nb += 1
                else:
                    ConnPossans += MeanMax(S_Z0, T_Z0_n, S_Z1, T_Z1_n)
                    ConnPossans_nb += 1

            elif Target.subtype == 4:  # SSC
                T_Z0_n = Targetpos[2] + Target.c1_down
                T_Z1_n = Targetpos[2] + Target.c1_up
                o = find_intersection_high_vectorize(S_Z0, S_Z1, T_Z0_n, T_Z1_n, Source.AX_w, Target.Bdend_w, d)
                overlap += o
                if o > 0:
                    ConnPos += MeanMax(S_Z0, T_Z0_n, S_Z1, T_Z1_n)
                    ConnPos_nb += 1
                else:
                    ConnPossans += MeanMax(S_Z0, T_Z0_n, S_Z1, T_Z1_n)
                    ConnPossans_nb += 1


            elif Target.subtype == 0:  # TPC
                # first cylinder
                T_Z0_n = Targetpos[2] + Target.c1_down
                T_Z1_n = Targetpos[2] + Target.c1_up
                o = find_intersection_high_vectorize(S_Z0, S_Z1, T_Z0_n, T_Z1_n, Source.AX_w, Target.r1, d)
                overlap += o
                if o > 0:
                    ConnPos += MeanMax(S_Z0, T_Z0_n, S_Z1, T_Z1_n)
                    ConnPos_nb += 1
                else:
                    ConnPossans += MeanMax(S_Z0, T_Z0_n, S_Z1, T_Z1_n)
                    ConnPossans_nb += 1

                #  second cylinder
                T_Z0_n = Targetpos[2] + Target.c2_down
                T_Z1_n = Targetpos[2] + Target.c2_up
                o = find_intersection_high_vectorize(S_Z0, S_Z1, T_Z0_n, T_Z1_n, Source.AX_w, Target.r2, d)
                overlap += o
                if o > 0:
                    ConnPos += MeanMax(S_Z0, T_Z0_n, S_Z1, T_Z1_n)
                    ConnPos_nb += 1
                else:
                    ConnPossans += MeanMax(S_Z0, T_Z0_n, S_Z1, T_Z1_n)
                    ConnPossans_nb += 1

                #  cone
                T_Z0_n = Targetpos[2] + Target.c3_down
                T_Z1_n = Targetpos[2] + Target.c3_up
                o = find_conic_intersection_high_vectorize(S_Z0, S_Z1, T_Z0_n, T_Z1_n, Source.AX_w / 2, Target.r3,
                                                                d, step, 0)
                overlap += o
                if o > 0:
                    ConnPos += MeanMax(S_Z0, T_Z0_n, S_Z1, T_Z1_n)
                    ConnPos_nb += 1
                else:
                    ConnPossans += MeanMax(S_Z0, T_Z0_n, S_Z1, T_Z1_n)
                    ConnPossans_nb += 1


        elif Target.type in [1, 2, 3, 4]:  # PV, SST, VIP or RLN
            T_Z0_n = Targetpos[2] + Target.c1_down
            T_Z1_n = Targetpos[2] + Target.c1_up
            overlap += find_intersection_high_vectorize(S_Z0, S_Z1, T_Z0_n, T_Z1_n, Source.AX_w, Target.Bdend_w, d)
            if overlap > 0:
                ConnPos += MeanMax(S_Z0, T_Z0_n, S_Z1, T_Z1_n)
                ConnPos_nb += 1
            else:
                ConnPossans += MeanMax(S_Z0, T_Z0_n, S_Z1, T_Z1_n)
                ConnPossans_nb += 1



    ###interneurons
    ###PV
    elif Source.type == 1:  # PV

        if Source.subtype == 0:  # BC  only inside the layer
            S_Z0 = Sourcepos[2] + Source.AX_down
            S_Z1 = Sourcepos[2] + Source.AX_up
            if S_Z0 < 0:  # kepp axon in the cortical columnlength
                S_Z0 = 0.
            if S_Z1 > L:
                S_Z1 = L + 0.

            if Target.type == 0:  # PV basket -->PC targetssoma
                if Target.layer == 2 and Target.subtype == 4:  # Si layeur IV et stellate
                    T_Z0_n = Targetpos[2] - Target.Bdend_l / 4
                    T_Z1_n = Targetpos[2] + Target.Bdend_l / 4
                    overlap += find_intersection_high_vectorize(S_Z0, S_Z1, T_Z0_n, T_Z1_n, Source.AX_w / 2,
                                                                     Target.Bdend_w / 4, d)
                    if overlap > 0:
                        ConnPos += MeanMax(S_Z0, T_Z0_n, S_Z1, T_Z1_n)
                        ConnPos_nb += 1
                    else:
                        ConnPossans += MeanMax(S_Z0, T_Z0_n, S_Z1, T_Z1_n)
                        ConnPossans_nb += 1


                else:  #
                    T_Z0_n = Targetpos[2] - Target.Bdend_l / 2
                    T_Z1_n = Targetpos[2] + Target.hsoma
                    overlap += find_intersection_high_vectorize(S_Z0, S_Z1, T_Z0_n, T_Z1_n, Source.AX_w,
                                                                     Target.dsoma, d)
                    if overlap > 0:
                        ConnPos += MeanMax(S_Z0, T_Z0_n, S_Z1, T_Z1_n)
                        ConnPos_nb += 1
                    else:
                        ConnPossans += MeanMax(S_Z0, T_Z0_n, S_Z1, T_Z1_n)
                        ConnPossans_nb += 1


            elif Target.type in [1, 2, 3, 4]:  # PV-->PV/SST/VIP/RLN
                T_Z0_n = Targetpos[2] + Target.c1_down
                T_Z1_n = Targetpos[2] + Target.c1_up
                overlap += find_intersection_high_vectorize(S_Z0, S_Z1, T_Z0_n, T_Z1_n, Source.AX_w,
                                                                 Target.Bdend_w, d)
                if overlap > 0:
                    ConnPos += MeanMax(S_Z0, T_Z0_n, S_Z1, T_Z1_n)
                    ConnPos_nb += 1
                else:
                    ConnPossans += MeanMax(S_Z0, T_Z0_n, S_Z1, T_Z1_n)
                    ConnPossans_nb += 1



        elif Source.subtype == 1:  # PV Chandelier -> PC
            S_Z0 = Sourcepos[2] + Source.AX_down
            S_Z1 = Sourcepos[2] + Source.AX_up

            if Target.type == 0:  # PV Chandelier -->PC targetssoma
                T_Z0_n = Targetpos[2] - Target.hsoma
                T_Z1_n = Targetpos[2]
                overlap += find_intersection_high_vectorize(S_Z0, S_Z1, T_Z0_n, T_Z1_n, Source.AX_w / 2,
                                                                 Target.dsoma / 2, d)
                if overlap > 0:
                    ConnPos += MeanMax(S_Z0, T_Z0_n, S_Z1, T_Z1_n)
                    ConnPos_nb += 1
                else:
                    ConnPossans += MeanMax(S_Z0, T_Z0_n, S_Z1, T_Z1_n)
                    ConnPossans_nb += 1


            elif Target.type == 1 in [1, 2, 3, 4]:  # PV-->PV/SST/VIP/RLN
                T_Z0_n = Targetpos[2] + Target.c1_down
                T_Z1_n = Targetpos[2] + Target.c1_up
                overlap += find_intersection_high_vectorize(S_Z0, S_Z1, T_Z0_n, T_Z1_n, Source.AX_w,
                                                                 Target.Bdend_w, d)
                if overlap > 0:
                    ConnPos += MeanMax(S_Z0, T_Z0_n, S_Z1, T_Z1_n)
                    ConnPos_nb += 1
                else:
                    ConnPossans += MeanMax(S_Z0, T_Z0_n, S_Z1, T_Z1_n)
                    ConnPossans_nb += 1



    ####
    # SST
    elif Source.type == 2:  # SST
        Sbot_Z0 = Sourcepos[2] + Source.c1_down
        Sbot_Z1 = L - Layer_d[0]
        if Sbot_Z0 < 0:  # kepp axon in the cortical columnlength
            Sbot_Z0 = 0.
        if Sbot_Z1 > L:
            Sbot_Z1 = L + 0.
        Stop_Z0 = Sbot_Z1
        Stop_Z1 = L
        if Sbot_Z0 < 0:  # kepp axon in the cortical columnlength
            Sbot_Z0 = 0.
        if Sbot_Z1 > L:
            Sbot_Z1 = L + 0.

        if Target.type == 0:  # SST -> PC
            if Target.subtype in [1, 2]:  # IPC or UPC
                # first cylinder
                T_Z0_n = Targetpos[2] + Target.c1_down
                T_Z1_n = Targetpos[2] + Target.c1_up

                o = find_intersection_high_vectorize(Sbot_Z0, Sbot_Z1, T_Z0_n, T_Z1_n, Source.AX_w, Target.r1, d)
                overlap += o
                if o > 0:
                    ConnPos += MeanMax(Sbot_Z0, T_Z0_n, Sbot_Z1, T_Z1_n)
                    ConnPos_nb += 1
                else:
                    ConnPossans += MeanMax(Sbot_Z0, T_Z0_n, Sbot_Z1, T_Z1_n)
                    ConnPossans_nb += 1
                o = find_intersection_high_vectorize(Stop_Z0, Stop_Z1, T_Z0_n, T_Z1_n, Source.AX_w2, Target.r1, d)
                overlap += o
                if o > 0:
                    ConnPos += MeanMax(Stop_Z0, T_Z0_n, Stop_Z1, T_Z1_n)
                    ConnPos_nb += 1
                else:
                    ConnPossans += MeanMax(Stop_Z0, T_Z0_n, Stop_Z1, T_Z1_n)
                    ConnPossans_nb += 1

                # cylinder
                T_Z0_n = Targetpos[2] + Target.c2_down
                T_Z1_n = Targetpos[2] + Target.c2_up
                # first axon
                o = find_intersection_high_vectorize(Sbot_Z0, Sbot_Z1, T_Z0_n, T_Z1_n, Source.AX_w, Target.r2, d)
                overlap += o
                if o > 0:
                    ConnPos += MeanMax(Sbot_Z0, T_Z0_n, Sbot_Z1, T_Z1_n)
                    ConnPos_nb += 1
                else:
                    ConnPossans += MeanMax(Sbot_Z0, T_Z0_n, Sbot_Z1, T_Z1_n)
                    ConnPossans_nb += 1

                # second axon
                o = find_intersection_high_vectorize(Stop_Z0, Stop_Z1, T_Z0_n, T_Z1_n, Source.AX_w2, Target.r2, d)
                overlap += o
                if o > 0:
                    ConnPos += MeanMax(Stop_Z0, T_Z0_n, Stop_Z1, T_Z1_n)
                    ConnPos_nb += 1
                else:
                    ConnPossans += MeanMax(Stop_Z0, T_Z0_n, Stop_Z1, T_Z1_n)
                    ConnPossans_nb += 1



            elif Target.subtype == 3:  # BPC
                # cylinder
                T_Z0_n = Targetpos[2] + Target.c1_down
                T_Z1_n = Targetpos[2] + Target.c1_up
                # first axon
                o = find_intersection_high_vectorize(Sbot_Z0, Sbot_Z1, T_Z0_n, T_Z1_n, Source.AX_w, Target.Adend_w,
                                                          d)
                overlap += o
                if o > 0:
                    ConnPos += MeanMax(Sbot_Z0, T_Z0_n, Sbot_Z1, T_Z1_n)
                    ConnPos_nb += 1
                else:
                    ConnPossans += MeanMax(Sbot_Z0, T_Z0_n, Sbot_Z1, T_Z1_n)
                    ConnPossans_nb += 1

                # second axon
                o = find_intersection_high_vectorize(Stop_Z0, Stop_Z1, T_Z0_n, T_Z1_n, Source.AX_w2,
                                                          Target.Adend_w, d)
                overlap += o
                if o > 0:
                    ConnPos += MeanMax(Stop_Z0, T_Z0_n, Stop_Z1, T_Z1_n)
                    ConnPos_nb += 1
                else:
                    ConnPossans += MeanMax(Stop_Z0, T_Z0_n, Stop_Z1, T_Z1_n)
                    ConnPossans_nb += 1

                # cylinder bas
                T_Z0_n = Targetpos[2] + Target.c2_down
                T_Z1_n = Targetpos[2] + Target.c2_up
                o = find_intersection_high_vectorize(Sbot_Z0, Sbot_Z1, T_Z0_n, T_Z1_n, Source.AX_w, Target.r2, d)
                overlap += o
                if o > 0:
                    ConnPos += MeanMax(Sbot_Z0, T_Z0_n, Sbot_Z1, T_Z1_n)
                    ConnPos_nb += 1
                else:
                    ConnPossans += MeanMax(Sbot_Z0, T_Z0_n, Sbot_Z1, T_Z1_n)
                    ConnPossans_nb += 1

                o = find_intersection_high_vectorize(Stop_Z0, Stop_Z1, T_Z0_n, T_Z1_n, Source.AX_w2, Target.r2, d)
                overlap += o
                if o > 0:
                    ConnPos += MeanMax(Stop_Z0, T_Z0_n, Stop_Z1, T_Z1_n)
                    ConnPos_nb += 1
                else:
                    ConnPossans += MeanMax(Stop_Z0, T_Z0_n, Stop_Z1, T_Z1_n)
                    ConnPossans_nb += 1

                # cylinder haut
                T_Z0_n = Targetpos[2] + Target.c3_down
                T_Z1_n = Targetpos[2] + Target.c3_up

                o = find_intersection_high_vectorize(Sbot_Z0, Sbot_Z1, T_Z0_n, T_Z1_n, Source.AX_w, Target.r2, d)
                overlap += o
                if o > 0:
                    ConnPos += MeanMax(Sbot_Z0, T_Z0_n, Sbot_Z1, T_Z1_n)
                    ConnPos_nb += 1
                else:
                    ConnPossans += MeanMax(Sbot_Z0, T_Z0_n, Sbot_Z1, T_Z1_n)
                    ConnPossans_nb += 1

                o = find_intersection_high_vectorize(Stop_Z0, Stop_Z1, T_Z0_n, T_Z1_n, Source.AX_w2, Target.r2, d)
                overlap += o
                if o > 0:
                    ConnPos += MeanMax(Stop_Z0, T_Z0_n, Stop_Z1, T_Z1_n)
                    ConnPos_nb += 1
                else:
                    ConnPossans += MeanMax(Stop_Z0, T_Z0_n, Stop_Z1, T_Z1_n)
                    ConnPossans_nb += 1



            elif Target.subtype == 4:  # SSC
                T_Z0_n = Targetpos[2] + Target.c1_down
                T_Z1_n = Targetpos[2] + Target.c1_up

                o = find_intersection_high_vectorize(Sbot_Z0, Sbot_Z1, T_Z0_n, T_Z1_n, Source.AX_w, Target.Bdend_w,
                                                          d)
                overlap += o
                if o > 0:
                    ConnPos += MeanMax(Sbot_Z0, T_Z0_n, Sbot_Z1, T_Z1_n)
                    ConnPos_nb += 1
                else:
                    ConnPossans += MeanMax(Sbot_Z0, T_Z0_n, Sbot_Z1, T_Z1_n)
                    ConnPossans_nb += 1

                o = find_intersection_high_vectorize(Stop_Z0, Stop_Z1, T_Z0_n, T_Z1_n, Source.AX_w2,
                                                          Target.Bdend_w, d)
                overlap += o
                if o > 0:
                    ConnPos += MeanMax(Stop_Z0, T_Z0_n, Stop_Z1, T_Z1_n)
                    ConnPos_nb += 1
                else:
                    ConnPossans += MeanMax(Stop_Z0, T_Z0_n, Stop_Z1, T_Z1_n)
                    ConnPossans_nb += 1



            elif Target.subtype == 0:  # TPC
                # first cylinder
                T_Z0_n = Targetpos[2] + Target.c1_down
                T_Z1_n = Targetpos[2] + Target.c1_up
                o = find_intersection_high_vectorize(Sbot_Z0, Sbot_Z1, T_Z0_n, T_Z1_n, Source.AX_w, Target.r1, d)
                overlap += o
                if o > 0:
                    ConnPos += MeanMax(Sbot_Z0, T_Z0_n, Sbot_Z1, T_Z1_n)
                    ConnPos_nb += 1
                else:
                    ConnPossans += MeanMax(Sbot_Z0, T_Z0_n, Sbot_Z1, T_Z1_n)
                    ConnPossans_nb += 1

                o = find_intersection_high_vectorize(Stop_Z0, Stop_Z1, T_Z0_n, T_Z1_n, Source.AX_w2, Target.r1, d)
                overlap += o
                if o > 0:
                    ConnPos += MeanMax(Stop_Z0, T_Z0_n, Stop_Z1, T_Z1_n)
                    ConnPos_nb += 1
                else:
                    ConnPossans += MeanMax(Stop_Z0, T_Z0_n, Stop_Z1, T_Z1_n)
                    ConnPossans_nb += 1

                # cylinder
                T_Z0_n = Targetpos[2] + Target.c2_down
                T_Z1_n = Targetpos[2] + Target.c2_up
                o = find_intersection_high_vectorize(Sbot_Z0, Sbot_Z1, T_Z0_n, T_Z1_n, Source.AX_w, Target.r2, d)
                overlap += o
                if o > 0:
                    ConnPos += MeanMax(Sbot_Z0, T_Z0_n, Sbot_Z1, T_Z1_n)
                    ConnPos_nb += 1
                else:
                    ConnPossans += MeanMax(Sbot_Z0, T_Z0_n, Sbot_Z1, T_Z1_n)
                    ConnPossans_nb += 1

                o = find_intersection_high_vectorize(Stop_Z0, Stop_Z1, T_Z0_n, T_Z1_n, Source.AX_w2, Target.r2, d)
                overlap += o
                if o > 0:
                    ConnPos += MeanMax(Stop_Z0, T_Z0_n, Stop_Z1, T_Z1_n)
                    ConnPos_nb += 1
                else:
                    ConnPossans += MeanMax(Stop_Z0, T_Z0_n, Stop_Z1, T_Z1_n)
                    ConnPossans_nb += 1

                # cone
                T_Z0_n = Targetpos[2] + Target.c3_down
                T_Z1_n = Targetpos[2] + Target.c3_up
                o = find_conic_intersection_high_vectorize(Sbot_Z0, Sbot_Z1, T_Z0_n, T_Z1_n, Source.AX_w,
                                                                Target.r3, d, step, 0)
                overlap += o
                if o > 0:
                    ConnPos += MeanMax(Sbot_Z0, T_Z0_n, Sbot_Z1, T_Z1_n)
                    ConnPos_nb += 1
                else:
                    ConnPossans += MeanMax(Sbot_Z0, T_Z0_n, Sbot_Z1, T_Z1_n)
                    ConnPossans_nb += 1

                o = find_conic_intersection_high_vectorize(Stop_Z0, Stop_Z1, T_Z0_n, T_Z1_n, Source.AX_w2,
                                                                Target.r3, d, step, 0)
                overlap += o
                if o > 0:
                    ConnPos += MeanMax(Stop_Z0, T_Z0_n, Stop_Z1, T_Z1_n)
                    ConnPos_nb += 1
                else:
                    ConnPossans += MeanMax(Stop_Z0, T_Z0_n, Stop_Z1, T_Z1_n)
                    ConnPossans_nb += 1


        #     # if SST--->PV
        elif Target.type in [1, 2, 3, 4]:  # PV insidelayer
            T_Z0_n = Targetpos[2] + Target.c1_down
            T_Z1_n = Targetpos[2] + Target.c1_up
            overlap += find_intersection_high_vectorize(Sbot_Z0, Sbot_Z1, T_Z0_n, T_Z1_n, Source.AX_w,
                                                             Target.Bdend_w, d)
            ConnPossans += MeanMax(Sbot_Z0, T_Z0_n, Sbot_Z1, T_Z1_n)
            ConnPossans_nb += 1
            overlap += find_intersection_high_vectorize(Stop_Z0, Stop_Z1, T_Z0_n, T_Z1_n, Source.AX_w2,
                                                             Target.Bdend_w, d)
            if overlap > 0:
                ConnPos += MeanMax(Stop_Z0, T_Z0_n, Stop_Z1, T_Z1_n)
                ConnPos_nb += 1
            else:
                ConnPossans += MeanMax(Stop_Z0, T_Z0_n, Stop_Z1, T_Z1_n)
                ConnPossans_nb += 1

    if overlap > 0:
        connected = 1
        if ConnPos_nb == 0:
            if ConnPossans_nb == 0:
                MeanPos = 0
            else:
                MeanPos = ConnPossans / ConnPossans_nb
        else:
            MeanPos = ConnPos / ConnPos_nb
    else:
        if ConnPossans_nb == 0:
            MeanPos = 0
        else:
            MeanPos = ConnPossans / ConnPossans_nb

    ################
    #        print(connected)
    #        print(overlap)
    return connected, overlap, MeanPos


def Create_Connectivity_Matrix(C,inputpercent,NB_DPYR,NB_Th,Cellpos,seed=0,func=None):
    #C is column morphology

    if not seed == 0:
        np.random.seed(seed)
    #output dict
    Conx={}
    print('Create_Connectivity_Matrices........')

    connectivitymatrix = []
    connectivityweight = []
    connectivityZpos = []
    PreSynaptic_Cell_AMPA = []
    PreSynaptic_Cell_GABA = []
    PreSynaptic_Soma_AMPA = []
    PreSynaptic_Soma_GABA_d = []
    PreSynaptic_Soma_GABA_s = []
    PreSynaptic_Soma_GABA_a = []
    ExternalPreSynaptic_Cell_AMPA_DPYR = []
    ExternalPreSynaptic_Cell_AMPA_Th = []
    PreSynapticWeight_AMPA = []
    PreSynapticWeight_GABA = []
    PreSynapticPos_AMPA = []
    PreSynapticPos_GABA = []

    Cellposflat = np.vstack((Cellpos[0], Cellpos[1], Cellpos[2], Cellpos[3], Cellpos[4]))
    Layertop_pos = np.cumsum(C.L_th[::-1])

    target = Cell_morphology.Neuron(0, 1, 0)
    Neighbor = Cell_morphology.Neuron(0, 1, 0)
    nbcells = np.sum(C.Layer_nbCells)
    nbcellscum = np.cumsum(C.Layer_nbCells)
    nbcellscum = np.append(0, nbcellscum)
    nbcellscum_mat = np.cumsum(np.array([nbcellscum[:-1], C.NB_PYR, C.NB_PV, C.NB_SST, C.NB_VIP, C.NB_RLN]), axis=0)

    PCsubtypes_Per = np.cumsum(C.PCsubtypes_Per, axis=1)
    indexval = -1

    cm = np.zeros(nbcells, dtype=int)  # []
    dist = np.zeros(nbcells, dtype=float)  # []
    Weight = np.zeros(nbcells, dtype=float)  # []
    Zpos = np.zeros(nbcells, dtype=np.int32)  # []
    Layerconn_l = np.zeros(nbcells, dtype=np.int32)  # []

    AMPAcells = np.zeros(nbcells, dtype=int) - 1  # []
    GABAcells = np.zeros(nbcells, dtype=int) - 1  # []

    SLayer_nbCells = np.cumsum(np.hstack((0, C.Layer_nbCells)))
    # List_celltypes = copy.deepcopy(C.List_celltypes)
    List_cellsubtypes = copy.deepcopy(C.List_cellsubtypes)
    List_cellsubtypes_flatten = [item for sublist in List_cellsubtypes for item in sublist]
    # t0 = time.time()
    # ik = 1
    curcellnb = 0
    for l in range(len(C.Layer_nbCells)):
        for cell in range(int(C.Layer_nbCells[l])):
            if np.mod(cell, 20) == 0:
                print('cell', cell)
            curcellnb +=1
            if np.mod(cell, 20) == 0 and not func is None:
                func('current layer: ' + str(l+1) +' \ 5\n' +
                     'current layer: ' + str(cell+1) +' \ ' +str(int(C.Layer_nbCells[l]))+'\n' +
                     'placed number of cell: ' + format(curcellnb/nbcells, '.2f') )
            # if indexval ==80:
            #     a=1
            # self.updateCell.something_happened.emit(indexval)
            indexval += 1
            cm = cm * 0
            dist = dist * 0
            Weight = Weight * 0 - 1
            Zpos = Zpos * 0  # The Z position of the synaptic connexion (Zpos=0 no connection, Zpos=-1 ==> not enough cells connected)
            Layerconn_l = Layerconn_l * 0

            AMPAcells = AMPAcells * 0 - 1
            AMPAcells_sparse = []
            Weight_AMPA = []
            Pos_AMPA = []
            GABAcells = GABAcells * 0 - 1  # []
            GABAcells_sparse = []
            GABASoma =  (np.zeros(nbcells).astype(int)).astype(str)
            GABASoma_d_sparse = []
            GABASoma_s_sparse = []
            GABASoma_a_sparse = []
            Weight_GABA = []
            Pos_GABA = []

            subtype = List_cellsubtypes[l][cell]

            # # if Principal cell check subtypes
            # if (C.List_celltypes[l][cell] == 0):  # getsubtype
            #     if cell < PCsubtypes_Per[l][0]:
            #         subtype = 0  # TPC
            #     elif (cell >= PCsubtypes_Per[l][0]) and (cell < PCsubtypes_Per[l][1]):
            #         subtype = 1  # UPC
            #     elif (cell >= PCsubtypes_Per[l][1]) and (cell < PCsubtypes_Per[l][2]):
            #         subtype = 2  # IPC
            #     elif (cell >= PCsubtypes_Per[l][2]) and (cell < PCsubtypes_Per[l][3]):
            #         subtype = 3  # BPC
            #     elif (cell >= PCsubtypes_Per[l][3]) and (cell < PCsubtypes_Per[l][4]):
            #         subtype = 4  # SSC
            #
            #     # print(l,cell,subtype,PCsubtypes_Per)
            #     # print(List_cellsubtypes[l][cell])
            #     List_cellsubtypes[l][cell] = subtype
            #
            # # if PV check is chandeliers or Basket
            # elif (C.List_celltypes[l][cell] == 1):  # PV get subtype
            #     if (cell - C.NB_PYR[l]) < C.NB_PV_BC[l]:
            #         subtype = 0  # BC
            #     else:
            #         subtype = 1  # Chandelier
            #     List_cellsubtypes[l][cell] = subtype
            # else:
            #     List_cellsubtypes[l][cell] = -1

            target.update_type(type=C.List_celltypes[l][cell], layer=l, subtype=subtype)
            index = 0

            d = np.linalg.norm(Cellposflat[:, :2] - Cellpos[l][cell][:2], axis=1)
            indflat = -1
            for subl in range(len(C.Layer_nbCells)):
                for v in range(0, int(C.Layer_nbCells[subl])):
                    indflat += 1

                    if inputpercent[int(C.List_celltypes[subl][v] + 5 * subl), int(target.type + 5 * l)] == 0:
                        pass
                        # cm[index] = 0
                        # Weight[index] = -1
                        # AMPAcells[index] = -1
                        # GABAcells[index] = -1
                        # GABASoma[index] = '0'
                        # Zpos[index] = 0


                    else:
                        # remove auto connections except for PV [Deleuze et al. 2019,plos Bio]
                        if ((v == cell) and (subl == l) and (C.List_celltypes[l][cell] != 1)):
                            pass
                            # cm[index] = 0
                            # Weight[index] = -1
                            # AMPAcells[index] = -1
                            # GABAcells[index] = -1
                            # GABASoma[index] = '0'
                            # Zpos[index] = 0

                        else:
                            subtype = List_cellsubtypes[subl][v]
                            # if (C.List_celltypes[subl][v] == 0):  # getsubtype
                            #     if v < PCsubtypes_Per[subl][0]:
                            #         subtype = 0  # TPC
                            #     elif (v >= PCsubtypes_Per[subl][0]) and (v < PCsubtypes_Per[subl][1]):
                            #         subtype = 1  # UPC
                            #     elif (v >= PCsubtypes_Per[subl][1]) and (v < PCsubtypes_Per[subl][2]):
                            #         subtype = 2  # IPC
                            #     elif (v >= PCsubtypes_Per[subl][2]) and (v < PCsubtypes_Per[subl][3]):
                            #         subtype = 3  # BPC
                            #     elif (v >= PCsubtypes_Per[subl][3]) and (v < PCsubtypes_Per[subl][4]):
                            #         subtype = 4  # SSC
                            #
                            # if (C.List_celltypes[subl][v] == 1):  # PV get subtype
                            #     if (v - C.NB_PYR[subl]) < C.NB_PV_BC[subl]:
                            #         subtype = 0  # BC
                            #     else:
                            #         subtype = 1  # Chandelier
                            if not Neighbor.is_Same(C.List_celltypes[subl][v],subl,subtype):
                                Neighbor.update_type(type=C.List_celltypes[subl][v], layer=subl, subtype=subtype)

                            isconnected, overlap, ConnPos = IsConnected(Neighbor, Cellpos[subl][v], target,
                                                                        Cellpos[l][cell],
                                                                        d[indflat],C.L,C.L_th)  # Neighbor to target)
                            # find at which layer the dendritic connection is
                            if ConnPos == 0:  # not connected
                                Layerconn = 0
                            else:
                                # Layerconn = np.argwhere(np.sort(np.concatenate((Layertop_pos, np.array([ConnPos])),
                                #                                                axis=0)) == ConnPos)  # 1--> layer 6,5 layer I
                                # Layerconn = Layerconn[0][0] + 1
                                if ConnPos <= Layertop_pos[0]:
                                    Layerconn = 1
                                elif ConnPos <= Layertop_pos[1] and ConnPos >Layertop_pos[0]:
                                    Layerconn = 2
                                elif ConnPos <= Layertop_pos[2] and ConnPos >Layertop_pos[1]:
                                    Layerconn = 3
                                elif ConnPos <= Layertop_pos[3] and ConnPos >Layertop_pos[2]:
                                    Layerconn = 4
                                else:
                                    Layerconn = 5

                                # if ConnPos < self.Cellpos[l][cell][2]: #if connection below soma center
                                #     Layerconn=-1*Layerconn
                            Layerconn_l[index] = Layerconn
                            if isconnected == 1:
                                cm[index] = 1
                                Weight[index] = overlap
                                Zpos[index] = Layerconn

                                ####Fill presynatptic cell
                                if Neighbor.type == 0:  # excitatory Input
                                    AMPAcells[index] = v + SLayer_nbCells[subl]#np.sum(SLayer_nbCells[subl])
                                    # GABAcells[index] = -1
                                    GABASoma[index] = '0'
                                else:
                                    # inhibitory input
                                    GABAcells[index] = v + SLayer_nbCells[subl]#np.sum(SLayer_nbCells[subl])
                                    # AMPAcells[index] = -1

                                    if target.type in [0]: #to PC
                                        if Neighbor.type == 1:  # from, PV
                                            if Neighbor.subtype == 0:  # if from Basket cell
                                                GABASoma[index] = 's'
                                            elif Neighbor.subtype == 1:  # if from  chandelier cell
                                                GABASoma[index] = 'a'
                                        else:
                                            GABASoma[index] = 'd'
                                    elif target.type in [1, 2, 3, 4]:
                                        GABASoma[index] = 's'
                            # else:
                            #     cm[index] = 0
                            #     Weight[index] = -1
                            #     AMPAcells[index] = -1
                            #     GABAcells[index] = -1
                            #     GABASoma[index] = '0'
                            #     Zpos[index] = 0

                    index += 1
            #####Check afferences:##############
            Afferences = inputpercent[:, int(target.type + 5 * l)]



            if np.sum(cm[0:C.NB_RLN[0]]) > Afferences[4]:
                NBrange = np.array(range(0, C.NB_RLN[0]))
                th = np.argsort([Weight[j] for j in np.array(NBrange)])[::-1]
                Weight[NBrange[th[int(Afferences[4]):]]] = 0
                Zpos[NBrange[th[int(Afferences[4]):]]] = 0
                cm[NBrange[th[int(Afferences[4]):]]] = 0
                AMPAcells[NBrange[th[int(Afferences[4]):]]] = -1
                GABAcells[NBrange[th[int(Afferences[4]):]]] = -1
                GABASoma[NBrange[th[int(Afferences[4]):]]] = '0'

            elif np.sum(cm[0:C.NB_RLN[0]]) < Afferences[4]:
                nb = Afferences[4] - np.sum(cm[0:C.NB_RLN[0]])
                indice_ = np.random.randint(0, C.NB_RLN[0], size=nb)

                # getminweight
                Weight2 = Weight[: C.NB_RLN[0]]
                Weight2 = Weight2[Weight2 > 0]
                if Weight2.size == 0:
                    weigthmini = 1
                    # print('RLN',l,cell,weigthmini)
                else:
                    weigthmini = np.min(Weight2)
                # weigthmini = np.min(Weight)
                # if weigthmini < 1:
                #     weigthmini = 1

                for ind in range(nb):
                    pos = indice_[ind]
                    if not pos == indexval:
                        Weight[pos] = weigthmini
                        cm[pos] = 1
                        AMPAcells[pos] = -1
                        GABAcells[pos] = pos
                        GABASoma[pos] = 's'
                        Zpos[pos] = 5  # from layer 1

            for ll in range(1, 5):
                for type in range(5):
                    if not int(Afferences[type + 5 * ll]) == 0:

                        # NBrange = []
                        # if type == 0:  # PC
                        #     NBrange = np.array(range(nbcellscum[ll], nbcellscum[ll] + C.NB_PYR[ll]))
                        # elif type == 1:  # PV
                        #     NBrange = np.array(range(nbcellscum[ll] + C.NB_PYR[ll],
                        #                              nbcellscum[ll] + C.NB_PYR[ll] + C.NB_PV[ll]))
                        # elif type == 2:  # SST
                        #     NBrange = np.array(range(nbcellscum[ll] + C.NB_PYR[ll] + C.NB_PV[ll],
                        #                              nbcellscum[ll] + C.NB_PYR[ll] + C.NB_PV[ll] + C.NB_SST[
                        #                                  ll]))
                        # elif type == 3:  # VIP
                        #     NBrange = np.array(
                        #         range(nbcellscum[ll] + C.NB_PYR[ll] + C.NB_PV[ll] + C.NB_SST[ll],
                        #               nbcellscum[ll] + C.NB_PYR[ll] + C.NB_PV[ll] + C.NB_SST[ll] + C.NB_VIP[
                        #                   ll]))
                        # elif type == 4:  # RLN
                        #     NBrange = np.array(range(
                        #         nbcellscum[ll] + C.NB_PYR[ll] + C.NB_PV[ll] + C.NB_SST[ll] + C.NB_VIP[ll],
                        #         nbcellscum[ll] + C.NB_PYR[ll] + C.NB_PV[ll] + C.NB_SST[ll] +
                        #         C.NB_VIP[ll] + C.NB_RLN[ll]))
                        NBrange = np.array(range(nbcellscum_mat[type,ll], nbcellscum_mat[type+1,ll]))

                        if len(NBrange) > 0:
                            somme_cm = np.sum(cm[NBrange])
                            if somme_cm > int(Afferences[
                                                  type + 5 * ll]):  # np.sum([cm[j] for j in NBrange]) > int(Afferences[type+5*ll]):
                                th = np.argsort(Weight[NBrange])[::-1]
                                ind_th = NBrange[th[int(Afferences[type + 5 * ll]):]]
                                Weight[ind_th] = 0
                                Zpos[ind_th] = 0
                                cm[ind_th] = 0
                                AMPAcells[ind_th] = -1
                                GABAcells[ind_th] = -1
                                GABASoma[ind_th] = '0'

                            elif somme_cm < int(Afferences[type + 5 * ll]):
                                Weight2 = Weight[NBrange]
                                Weight2 = Weight2[Weight2 > 0]
                                if Weight2.size == 0:
                                    weigthmini = 1
                                    # print(ll,type,l,cell,weigthmini)
                                else:
                                    weigthmini = np.min(Weight2)

                                # print('not enough)')
                                nb = int(Afferences[type + 5 * ll]) - somme_cm

                                indice_ = np.random.randint(0, len(NBrange), size=nb)
                                for ind in range(nb):
                                    pos = NBrange[indice_[ind]]
                                    # subtp = List_cellsubtypes[ll][pos-nbcellscum_mat[type,ll]]
                                    if not pos == indexval:
                                        cm[pos] = 1
                                        Weight[pos] = weigthmini
                                        Zpos[pos] = Layerconn_l[pos]# 5 - ll  # add the layer of presynaptic cell
                                        if type == 0: # From PC
                                            AMPAcells[pos] = pos
                                            GABAcells[pos] = -1
                                            GABASoma[pos] = '0'
                                        else:
                                            #From from GABA
                                            GABAcells[pos] = pos
                                            AMPAcells[pos] = -1
                                            if target.type in [0]: #GABA to PC
                                                if type == 1: #PV --> PC
                                                    if List_cellsubtypes_flatten[pos]  == 0:
                                                        GABASoma[pos] = 's'
                                                    else:
                                                        GABASoma[pos] = 'a'
                                                else:
                                                    GABASoma[pos] = 'd'
                                            elif target.type in [1, 2, 3, 4]:
                                                GABASoma[pos] = 's'

            # create sparse arrays
            for i in range(len(AMPAcells)):
                if AMPAcells[i] != -1:
                    AMPAcells_sparse.append(AMPAcells[i])
                    Weight_AMPA.append(Weight[i])
                    Pos_AMPA.append(Zpos[i])

            for i in range(len(GABAcells)):
                if GABAcells[i] != -1:
                    GABAcells_sparse.append(GABAcells[i])
                    Weight_GABA.append(Weight[i])
                    Pos_GABA.append(Zpos[i])

                    if GABASoma[i] == 'd':
                        GABASoma_d_sparse.append(1)  # 1   for dent
                        GABASoma_s_sparse.append(0)  # 1   for soma
                        GABASoma_a_sparse.append(0)  # 1   for ais
                    elif GABASoma[i] == 's':
                        GABASoma_d_sparse.append(0)  # 1   for dent
                        GABASoma_s_sparse.append(1)  # 1   for soma
                        GABASoma_a_sparse.append(0)  # 1   for ais
                    elif GABASoma[i] == 'a':
                        GABASoma_d_sparse.append(0)  # 1   for dent
                        GABASoma_s_sparse.append(0)  # 1   for soma
                        GABASoma_a_sparse.append(1)  # 1   for ais
                    else:
                        GABASoma_d_sparse.append(0)  # 1   for dent
                        GABASoma_s_sparse.append(0)  # 1   for soma
                        GABASoma_a_sparse.append(0)  # 1   for ais



            PreSynaptic_Cell_AMPA.append(np.asarray(AMPAcells_sparse))
            PreSynaptic_Cell_GABA.append(np.asarray(GABAcells_sparse))
            PreSynaptic_Soma_AMPA.append(np.ones(len(AMPAcells_sparse), dtype=int))
            PreSynaptic_Soma_GABA_d.append(np.asarray(GABASoma_d_sparse))
            PreSynaptic_Soma_GABA_s.append(np.asarray(GABASoma_s_sparse))
            PreSynaptic_Soma_GABA_a.append(np.asarray(GABASoma_a_sparse))
            # print(np.sum(PreSynaptic_Soma_GABA_d[-1]),np.sum(PreSynaptic_Soma_GABA_s[-1]),np.sum(PreSynaptic_Soma_GABA_a[-1]))
            PreSynapticWeight_AMPA.append(np.asarray(Weight_AMPA))
            PreSynapticWeight_GABA.append(np.asarray(Weight_GABA))
            PreSynapticPos_AMPA.append(np.asarray(Pos_AMPA))
            PreSynapticPos_GABA.append(np.asarray(Pos_GABA))

            connectivitymatrix.append(np.where(cm == 1)[0])
            connectivityweight.append(Weight[np.where(cm == 1)])
            connectivityZpos.append(Zpos[np.where(cm == 1)])
            # create external synaptic input
            nbstim = int(Afferences[26])
            nbth = int(Afferences[25])

            if (nbstim != 0):
                x0 = [np.random.randint(NB_DPYR) for i in range(np.min((int(nbstim), NB_DPYR)))]
                ExternalPreSynaptic_Cell_AMPA_DPYR.append(np.asarray(x0))
            else:
                ExternalPreSynaptic_Cell_AMPA_DPYR.append(np.asarray([]))

            if (nbth != 0):
                x1 = [np.random.randint(NB_Th) for i in range(np.min((int(nbth), NB_Th)))]
                ExternalPreSynaptic_Cell_AMPA_Th.append(np.asarray(x1))
            else:
                ExternalPreSynaptic_Cell_AMPA_Th.append(np.asarray([]))

            # print('z',(time.time()-t0)/ik)
            # ik +=1

    Conx['PreSynaptic_Cell_AMPA']=PreSynaptic_Cell_AMPA
    Conx['PreSynaptic_Cell_GABA']=PreSynaptic_Cell_GABA
    Conx['PreSynaptic_Soma_AMPA']=PreSynaptic_Soma_AMPA
    Conx['PreSynaptic_Soma_GABA_d']=PreSynaptic_Soma_GABA_d
    Conx['PreSynaptic_Soma_GABA_s']=PreSynaptic_Soma_GABA_s
    Conx['PreSynaptic_Soma_GABA_a']=PreSynaptic_Soma_GABA_a
    Conx['PreSynapticWeight_AMPA']=PreSynapticWeight_AMPA
    Conx['PreSynapticWeight_GABA']=PreSynapticWeight_GABA
    Conx['PreSynapticPos_AMPA']=PreSynapticPos_AMPA
    Conx['PreSynapticPos_GABA']=PreSynapticPos_GABA
    Conx['ExternalPreSynaptic_Cell_AMPA_DPYR']=ExternalPreSynaptic_Cell_AMPA_DPYR
    Conx['ExternalPreSynaptic_Cell_AMPA_Th']=ExternalPreSynaptic_Cell_AMPA_Th
    Conx['connectivitymatrix']=connectivitymatrix
    Conx['connectivityweight']=connectivityweight

    return Conx


@vectorize([float64(float64, float64, float64, float64, float64, float64, float64)])
def find_intersection_high_vectorize(S_z0, S_z1, T_z0, T_z1, r1, r2, d):
    # h = np.minimum(S_z1, T_z1) - np.maximum(S_z0, T_z0)
    if S_z1 <= T_z1:
        h = S_z1
    else:
        h = T_z1
    if S_z0 >= T_z0:
        h -= S_z0
    else:
        h -= T_z0
    if h > 0:
        if r1 + r2 < d:
            return 0.
        if d == 0.:
            r3 = np.minimum(r1, r2)
            return math.pi * r3 * r3
        rad1sqr = r1 * r1
        rad2sqr = r2 * r2
        d2 = d * d
        angle1 = (rad1sqr + d2 - rad2sqr) / (2 * r1 * d)
        angle2 = (rad2sqr + d2 - rad1sqr) / (2 * r2 * d)
        if (-1 <= angle1 < 1) or (-1 <= angle2 < 1):
            theta1 = math.acos(angle1) * 2
            theta2 = math.acos(angle2) * 2
            area1 = (0.5 * theta2 * rad2sqr) - (0.5 * rad2sqr * math.sin(theta2))
            area2 = (0.5 * theta1 * rad1sqr) - (0.5 * rad1sqr * math.sin(theta1))
            return area1 + area2
        elif angle1 < -1 or angle2 < -1:
            r3 = np.minimum(r1, r2)
            return math.pi * r3 * r3
        return 0.
    else:
        return 0.

@vectorize([float64(float64, float64, float64, float64, float64, float64, float64, float64, float64)])
def find_conic_intersection_high_vectorize(S_z0, S_z1, T_z0, T_z1, r1, r2, d, step, cone_sens):
    # h = np.minimum(S_z1, T_z1) - np.maximum(S_z0, T_z0)
    if S_z1 <= T_z1:
        h = S_z1
    else:
        h = T_z1
    if S_z0 >= T_z0:
        h -= S_z0
    else:
        h -= T_z0
    if h > 0:
        heith = T_z1 - T_z0
        r = np.arange(step, heith, step) / heith * r2
        h2 = np.arange(step, heith, step)
        if cone_sens == 0:
            d2 = T_z0 + h2
            r = r[np.bitwise_and(d2 >= S_z0, d2 <= S_z1)]
        elif cone_sens == 1:
            d2 = T_z1 - h2
            r = r[np.bitwise_and(d2 >= S_z0, d2 <= S_z1)]
        o = 0.
        for ids in range(len(r)):
            r3 = r[ids]
            if r1 + r3 < d:
                continue
            if d == 0:
                r4 = min(r1, r3)
                o += math.pi * r4 * r4
                continue
            rad1sqr = r1 * r1
            rad2sqr = r3 * r3
            d2 = d * d
            angle1 = (rad1sqr + d2 - rad2sqr) / (2 * r1 * d)
            angle2 = (rad2sqr + d2 - rad1sqr) / (2 * r3 * d)
            if (-1 <= angle1 < 1) or (-1 <= angle2 < 1):
                theta1 = math.acos(angle1) * 2
                theta2 = math.acos(angle2) * 2
                area1 = (0.5 * theta2 * rad2sqr) - (0.5 * rad2sqr * math.sin(theta2))
                area2 = (0.5 * theta1 * rad1sqr) - (0.5 * rad1sqr * math.sin(theta1))
                o += area1 + area2
                continue
            elif angle1 < -1 or angle2 < -1:
                r4 = min(r1, r3)
                o += math.pi * r4 * r4
                continue
        return o
    else:
        return 0.


