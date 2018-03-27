# from unicon import initialization, Element, Monoatom_species, Gas_species, Solid_species,\
# Endmember_species, solid_solution_species
import unicon
import pandas as pd
from g_mix import *

def add_species(name, formula = {}, tran_T = -1, s_type = "gas", solution_type = "ideal solution"):
    assert type(name) is str
    df = pd.read_excel("thermodynamics.xlsx", name)
    df = df[["T","G_f"]].dropna(0)
    if s_type == "gas":
        unicon.Gas_species(name, formula, df.values, transition_temp = tran_T)
    elif s_type == "monoatom":       
        unicon.Monoatom_species(name, df.values, transition_temp = tran_T)
    elif s_type == "solid":       
        unicon.Solid_species(name, formula, df.values, transition_temp = tran_T)
    elif s_type == "endmember":
        unicon.Endmember_species(name, formula, df.values, transition_temp = tran_T)


def add_everything(p_tot = 1e-3):
    unicon.initialization()

    A_H = 2.43e10
    A_Al = 8.41e4/A_H
    A_O = 1.413e7/A_H
    A_C = 7.079e6/A_H
    A_Si = 1e6/A_H
    A_Ca = 6.287e4/A_H
    A_Ti = 2422/A_H
    A_Mg = 1.020e6/A_H
    A_Fe = 8.380e5/A_H
    A_Ni = 4.780e4/A_H
    A_S = 4.449e5/A_H

    p_H2 = p_tot
    P_Al = 2*p_H2*A_Al;
    P_O = 2*p_H2*A_O;
    P_C = 2*p_H2*A_C;
    P_Si = 2*p_H2*A_Si
    P_Ca = 2*p_H2*A_Ca
    P_Mg = 2*p_H2*A_Mg
    P_Ti = 2*p_H2*A_Ti
    P_S = 2*p_H2*A_S


    unicon.Element("Al", P_Al, A_Al)
    unicon.Element("O", P_O, A_O)
    unicon.Element("H", 2*p_H2, 1)
    unicon.Element("C", P_C, A_C)
    unicon.Element("Si", P_Si, A_Si)
    unicon.Element("Ca", P_Ca, A_Ca)
    unicon.Element("Ti", P_Ti, A_Ti)
    unicon.Element("Mg", P_Mg, A_Mg)
    unicon.Element("S", P_S, A_S)

    add_species("H", s_type = "monoatom")
    add_species("H2", {"H":2}, s_type = "gas")
    add_species("H2O", {"H":2, "O":1}, s_type = "gas")


    add_species("Al", s_type = "monoatom")
    add_species("AlOH", {"Al": 1,"H":1, "O":1}, s_type = "gas")
    add_species("AlH", {"Al": 1,"H":1}, s_type = "gas")
    add_species("Al2O3", {"Al": 2,"O":3}, s_type = "solid")
    add_species("hibonite", {"Ca": 1,"Al": 12,"O":19}, s_type = "solid")
    add_species("Gehlenite", {"Ca": 2,"Al": 2, "Si": 1,"O":7}, s_type = "endmember")
    add_species("Akermanite", {"Ca": 2, "Mg":1, "Si":2,"O":7}, s_type = "endmember")
    add_species("spinel", {"Mg":1,"Al": 2,"O":4}, s_type = "solid")

    mel = unicon.solid_solution_species("mel", {"Gehlenite":0.99, "Akermanite":0.01}, 3, \
                           G_mix = G_melilite_mix, G_mix_prime = G_melilite_mix_prime)


    add_species("O", s_type = "monoatom")
    add_species("O2", {"O":2},s_type = "gas")

    add_species("C", s_type = "monoatom")
    add_species("CO", {"C":1, "O":1},s_type = "gas")
    add_species("CS", {"C":1, "S":1},s_type = "gas")
    add_species("HCO", {"H":1,"C":1, "O":1},s_type = "gas")

    add_species("Si", s_type = "monoatom")
    add_species("SiO", {"Si":1, "O":1},s_type = "gas")
    add_species("SiS", {"Si":1, "S":1},s_type = "gas")
    add_species("SiO2", {"Si":1, "O":2},s_type = "gas")


    add_species("Ca", tran_T = 1773,s_type = "monoatom")
    add_species("CaO", {"Ca":1, "O":1}, tran_T = 1773,s_type = "gas")
    add_species("diopside", {"Ca":1,"Mg":1,"Si": 2,"O":6}, s_type = "solid")

    add_species("Mg", tran_T = 1366,s_type = "monoatom")
    add_species("MgO", {"Mg":1, "O":1}, tran_T = 1366,s_type = "gas")
    add_species("MgH", {"Mg":1, "H":1}, tran_T = 1366,s_type = "gas")
    add_species("MgS", {"Mg":1, "S":1}, tran_T = 1366,s_type = "gas")
    add_species("MgOH", {"Mg":1, "O":1, "H":1}, tran_T = 1366,s_type = "gas")
    add_species("Mg(OH)2", {"Mg":1, "O":2, "H":2}, tran_T = 1366,s_type = "gas")

    add_species("Ti", s_type = "monoatom")
    add_species("TiO", {"Ti":1, "O":1},s_type = "gas")
    add_species("TiO2", {"Ti":1, "O":2},s_type = "gas")
    add_species("CaTiO3", {"Ca": 1,"Ti": 1,"O":3}, s_type = "solid")


    add_species("S", s_type = "monoatom")
    add_species("HS", {"S":1, "H":1},s_type = "gas")
    add_species("H2S", {"S":1, "H":2},s_type = "gas")
    add_species("SO", {"S":1, "O":1},s_type = "gas")