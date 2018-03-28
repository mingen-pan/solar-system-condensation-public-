import math
import types
import numpy as np
from scipy.optimize import curve_fit
from g_mix import *


def initialization(p_ref = 1e6/2.43e10*1e-3*2):
    global Elements, Gas_species_list, Monoatom_dict, Potential_solid_species_list, \
    Existing_solid_species_list, Solid_species_dict,Endmember_dict, Potential_solid_solution_species_list, \
    Existing_solid_solution_species_list, Solid_solution_species_dict, Isolation_dict
    global T
    Elements = {}
    Gas_species_list = []
    Monoatom_dict = {}
    Potential_solid_species_list = []
    Existing_solid_species_list = []
    Potential_solid_solution_species_list = []
    Existing_solid_solution_species_list = []
    Solid_species_dict = {}
    Solid_solution_species_dict = {}
    Endmember_dict = {}
    isolation_dict = {}
    global solid_p_ref
    solid_p_ref =  p_ref ## arbitrary set as the Si pressure

class Element():
    def __init__(self, name, pressure,solar_abundance):
        self.name = name
        self.p_tot = pressure
        self.abundance = solar_abundance
        self.gas_species_list = []
        #gas_species_list appends the tuple (species, # of atoms)
        self.solid_species_dict = {}
        self.solid_solution_species_dict = {}
        #the reason to use dict is that we can easily delete it
        Elements[name] = self
    def add_species(self, species):
        self.species_list.append(species)
    def mass_balance(self):
        total = 0
        for species in self.gas_species_list:
            total += species[0].pressure * species[1]
#             print(species[0].name, species[0].pressure, species[1]*species[0].pressure/self.p_tot)
        for species, atom_num in self.solid_species_dict.items():
            total += species.pressure * atom_num
        for species, atom_array in self.solid_solution_species_dict.items():
            total += np.sum(species.pressure * species.X * atom_array)
        return self.p_tot - total

class Monoatom_species():
    ## this is the mono-atomic gas in the system
    def __init__(self, name, G_vs_T, transition_temp = -1):
        assert type(G_vs_T) is np.ndarray
        self.log_pressure = 0
        self.pressure = 0
        self.name = name
        self.formula = {name: 1}
        self.transition_temp = transition_temp
        if transition_temp == -1:
            self.para,_ = curve_fit(Gibbs_energy_fit, G_vs_T[:,0],G_vs_T[:,1])
        else:
            self.G_vs_T = G_vs_T
        Monoatom_dict[name] = self
        Elements[name].gas_species_list.append((self, 1))
    def G(self):
        if self.transition_temp == -1:
            return Gibbs_energy_fit(T, *self.para)
        if self.transition_temp > 0:
            return np.interp(T, self.G_vs_T[:,0], self.G_vs_T[:,1])
    def update_log_pressure(self, log_pressure):
        self.log_pressure = log_pressure
        self.pressure = math.exp(log_pressure)
    def update_pressure(self, pressure):
        self.log_pressure = math.log(pressure)
        self.pressure = pressure
        
class Gas_species():
    def __init__(self, name, formula,G_vs_T, transition_temp = -1):
        assert type(G_vs_T) is np.ndarray
        assert type(formula) is dict
#         self.log_pressure = 0
        self.pressure = 0
        self.name = name
        self.formula = formula
        self.transition_temp = transition_temp
        if transition_temp == -1:
            self.para,_ = curve_fit(Gibbs_energy_fit, G_vs_T[:,0],G_vs_T[:,1])
        else:
            self.G_vs_T = G_vs_T
        Gas_species_list.append(self)
        for key, value in self.formula.items():
            Elements[key].gas_species_list.append((self, value))
    def G(self):
        if self.transition_temp == -1:
            return Gibbs_energy_fit(T, *self.para)
        if self.transition_temp > 0:
            return np.interp(T, self.G_vs_T[:,0], self.G_vs_T[:,1])
    def update_pressure(self):
        log_p = 0
        log_K = -self.G()
        for key, value in self.formula.items():
            # key is the element name and value is the number
            log_p += value * Monoatom_dict[key].log_pressure
            log_K += value * Monoatom_dict[key].G()
        log_K = log_K*1e3/(8.314*T)
        self.pressure = math.exp(log_K + log_p)
        
class Solid_species():
    def __init__(self, name, formula, G_vs_T, transition_temp = -1):
        assert type(G_vs_T) is np.ndarray
        assert type(formula) is dict
        self.pressure = 0
        self.name = name
        self.formula = formula
        self.transition_temp = transition_temp
#         p_max = [Elements[name].p_tot for name in formula.keys()]
#         self.p_ref = np.min(p_max)
        self.p_ref = solid_p_ref
        self.f_pressure = 0 # pressure = f_pressure * p_max
        if transition_temp == -1:
            self.para,_ = curve_fit(Gibbs_energy_fit, G_vs_T[:,0],G_vs_T[:,1])
        else:
            self.G_vs_T = G_vs_T
        Potential_solid_species_list.append(self)
        Solid_species_dict[name] = self
    def G(self):
        if self.transition_temp == -1:
            return Gibbs_energy_fit(T, *self.para)
        if self.transition_temp > 0:
            return np.interp(T, self.G_vs_T[:,0], self.G_vs_T[:,1])
    def equilibrium(self):
        log_p = 0
        log_K = -self.G()
        for key, value in self.formula.items():
            # key is the element name and value is the number
            log_p += value * Monoatom_dict[key].log_pressure
            log_K += value * Monoatom_dict[key].G()
        log_K = log_K*1e3/(8.314*T)
        return log_K + log_p
    def update_f_pressure(self, fp):
        self.f_pressure = fp
        self.pressure = fp * self.p_ref

class Endmember_species():
    def __init__(self, name, formula, G_vs_T, transition_temp = -1):
        assert type(G_vs_T) is np.ndarray
        assert type(formula) is dict
        self.pressure = 0
        self.name = name
        self.formula = formula
        self.transition_temp = transition_temp
        if transition_temp == -1:
            self.para,_ = curve_fit(Gibbs_energy_fit, G_vs_T[:,0],G_vs_T[:,1])
        else:
            self.G_vs_T = G_vs_T
        Endmember_dict[name] = self
    def G(self):
        if self.transition_temp == -1:
            return Gibbs_energy_fit(T, *self.para)
        if self.transition_temp > 0:
            return np.interp(T, self.G_vs_T[:,0], self.G_vs_T[:,1])
    def equilibrium(self):
        log_p = 0
        G = -self.G()
        for key, value in self.formula.items():
            # key is the element name and value is the number
            log_p += value * Monoatom_dict[key].log_pressure
            G += value * Monoatom_dict[key].G()
        G = G*1e3
        return G + log_p*8.314*T
            
class solid_solution_species():
    def __init__(self, name, endmember, solution_type, **kwargs):
        assert type(endmember) is dict
        self.name = name
        self.pressure = 0
        self.f_pressure = 0
        self.X = np.array(list(endmember.values()))
        self.endmember = endmember
        for num, key in enumerate(self.endmember.keys()):
            self.endmember[key] = num
        if "p_ref" in kwargs:
            self.p_ref = Elements[kwargs["p_ref"]].p_tot
        else:
            self.p_ref = solid_p_ref
        if solution_type == "ideal solution":
            solution_type = 1
        elif solution_type == "regular solution":
            solution_type = 2
        elif solution_type == "custom":
            solution_type = 3
            
        if solution_type == 1:
            self.G_mix = ideal_solution_G_mix
            self.G_mix_prime = ideal_solution_G_mix_prime
        elif solution_type == 2:
            self.W = kwargs["W"]
        elif solution_type == 3:
            assert type(kwargs["G_mix"]), type(kwargs["G_mix_prime"])  is types.FunctionType
            ## The G_mix should return an array or number
            self.G_mix = kwargs["G_mix"]
            self.G_mix_prime = kwargs["G_mix_prime"]
        self.solution_type = solution_type
        Potential_solid_solution_species_list.append(self)
        Solid_solution_species_dict[name] = self
    def G(self):
        G = 0
        for key, pos in self.endmember.items():
            G += Endmember_dict[key].G(T)* self.X[pos]
        G += self.G_mix(self.X, T)
        return G
    def update_f_pressure(self, fp):
        self.f_pressure = fp
        self.pressure = fp * self.p_ref
    def update_X(self, X):
        assert len(X) == len(self.endmember)
        if np.abs(np.sum(X) - 1) >= 1e-6:
            X = X/np.sum(X)
        self.X = X
    def update_bounded_X(self, X):
        assert len(X) == len(self.endmember) - 1, "the bounded X should have dim of N - 1"
        tot_X = np.sum(X)
        X = np.append(X, 1 - tot_X)
        self.X = X
    def equilibrium(self):
        eqn = 0
        #eqn = [G0-i - sum(G0_gas_i) - RT*sum(log_p_gas_i)] - [G0-n - sum(G0_gas_n) - RT*sum(log_p_gas_n)] + dG_mix/dX
        for num, key in enumerate(self.endmember.keys()):
            eqn -= Endmember_dict[key].equilibrium() * self.X[num]
        eqn += self.G_mix(self.X, T)
        return eqn
    def equilibrium_prime(self):
        eqn = np.zeros(len(self.endmember) - 1)
        #eqn = [G0-i - sum(G0_gas_i) - RT*sum(log_p_gas_i)] - [G0-n - sum(G0_gas_n) - RT*sum(log_p_gas_n)] + dG_mix/dX
        keys = list(self.endmember.keys())
        for num, key in enumerate(keys[:-1]):
            eqn[num] = Endmember_dict[keys[-1]].equilibrium() \
            - Endmember_dict[key].equilibrium()
        eqn += self.G_mix_prime(self.X, T)
        return eqn
    def _equilibrium(self, bounded_X):
        self.update_bounded_X(bounded_X)
        return self.equilibrium()
    def _equilibrium_prime(self, bounded_X):
        self.update_bounded_X(bounded_X)
        return self.equilibrium_prime()
        

