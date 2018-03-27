from scipy.optimize import fsolve, fmin_tnc
import numpy as np
from unicon import *
from pre_process import *

def add_solid(solid):
    unicon.Existing_solid_species_list.append(solid)
    unicon.Potential_solid_species_list.remove(solid)
    solid.update_f_pressure(1e-4) #arbitrary choose a constituent element
    for key, value in solid.formula.items():
        unicon.Elements[key].solid_species_dict[solid] = value
        
def del_solid(solid):
    unicon.Potential_solid_species_list.append(solid)
    unicon.Existing_solid_species_list.remove(solid)
    solid.pressure = 0
    for key in solid.formula.keys():
        del unicon.Elements[key].solid_species_dict[solid]
        
def add_solid_solution(solid):
    unicon.Existing_solid_solution_species_list.append(solid)
    unicon.Potential_solid_solution_species_list.remove(solid) 
    solid.update_f_pressure(1e-4) #arbitrary choose a constituent element
    element_dict = {}
    for endmember in solid.endmember.keys():
        for key in unicon.Endmember_dict[endmember].formula.keys():
            if key not in element_dict:
                element_dict[key] = []
    for endmember in solid.endmember.keys():
        for element in element_dict.keys():
            if element in unicon.Endmember_dict[endmember].formula:
                element_dict[element].append(unicon.Endmember_dict[endmember].formula[element])
            else:
                element_dict[element].append(0)
    for key in element_dict.keys():
        unicon.Elements[key].solid_solution_species_dict[solid] = np.array(element_dict[key])
        
def del_solid_solution(solid):
    unicon.Potential_solid_solution_species_list.append(solid)
    unicon.Existing_solid_solution_species_list.remove(solid)
    solid.pressure = 0
    element_list = []
    for endmember in solid.endmember.keys():
            for key in unicon.Endmember_dict[endmember].formula.keys():
                if key not in element_list:
                    element_list.append(key)
    for key in element_list:
        del unicon.Elements[key].solid_solution_species_dict[solid]


def check_potential_solid(T):
    flag = 0
    for solid in unicon.Potential_solid_species_list:
        if solid.equilibrium(unicon.Monoatom_dict, T) >= 1e-9:
            flag = 1
            add_solid(solid)
        
    for solid in unicon.Potential_solid_solution_species_list:
        X = solid.X
        X = X[:-1]
        X = fmin_tnc(solid._equilibrium, X, args = (unicon.Monoatom_dict, T), fprime = solid._equilibrium_prime, bounds = [(0.000000001,0.99999999999)])
        X = X[0]
        if solid._equilibrium(X, unicon.Monoatom_dict, T) <= -1e-9:
            flag = 1
            add_solid_solution(solid)                                     
    return flag
                
def check_existing_solid(T):
    flag = 0
    for solid in unicon.Existing_solid_species_list:
        if solid.pressure < 0:
            flag = 1
            del_solid(solid)
                    
    for solid in unicon.Existing_solid_solution_species_list:
        if solid.pressure < 0:
            flag = 1
            del_solid_solution(solid)
    return flag

def isolation(alpha):
    # alpha is the isolation rate
    assert alpha <= 1, "alpha is the fraction of isolation and should be smaller than one"
    for solid in unicon.Existing_solid_species_list:
        for key, value in solid.formula.items():
            unicon.Elements[key].p_tot -= solid.pressure * value * alpha
        if solid.name not in unicon.Isolation_dict:
            unicon.Isolation_dict[solid.name] = solid.pressure * alpha
        else:
            unicon.Isolation_dict[solid.name] += solid.pressure * alpha
        solid.f_pressure *= 1 - alpha
        solid.pressure *= 1 - alpha
    for solid in unicon.Existing_solid_solution_species_list:
        element_list = []
        for endmember in solid.endmember.keys():
            for key in unicon.Endmember_dict[endmember].formula.keys():
                if key not in element_list:
                    element_list.append(key)
        for key in element_list:
            atom_array = unicon.Elements[key].solid_solution_species_dict[solid]
            unicon.Elements[key].p_tot -= alpha * solid.pressure * solid.X  * atom_array
        for endmember, idx in solid.endmember.items():
            if solid.name not in unicon.Isolation_dict:
                unicon.Isolation_dict[endmember] = solid.pressure * solid.X[idx] * alpha
            else:
                unicon.Isolation_dict[endmember] += solid.pressure * solid.X[idx] * alpha
        solid.f_pressure *= 1 - alpha
        solid.pressure *= 1 - alpha
        
def change_p_tot(P):
    if type(P) == np.float64 or type(P) == float:
        # Here the P is defined by the P_H2
        P_H = 2*P
        for key in unicon.Elements.keys():
            # The abundance here is normalized to the H_abundance
            unicon.Elements[key].p_tot = unicon.Elements[key].abundance * P_H
    elif type(P) == list or type(P) == np.array:
        assert len(P) == len(unicon.Elements), "the input P has differnt length from Elements"
        for num, element in enumerate(unicon.Elements.values()):
            element.p_tot = P[num]
    
    
def state_change(T0, P_0, Tf, P_f, P_change_rate = 1.001):
	assert T0 == Tf, "the current version does not support T change, please change mannually"
	x = extract_x()
	if (Tf - T0) >= 0:
	    t_sign = 1 
	else:
	    t_sign = -1
	t = T0
	#Change the pressure
	if type(P_0) == float:
	    delta = np.abs(math.log(P_0) - math.log(P_f))
	    n = int(np.ceil(delta/math.log(P_change_rate)))
	    P_step = np.linspace(math.log(P_0),math.log(P_f),n)
	    P_step = np.exp(P_step)
	    for P in P_step:
	        change_p_tot(P)
	        x = update_state(x, t)
	        
	elif type(P_0) == list or type(P_0) == np.array:
	    log_P_f = np.log(P_f)
	    log_P_0 = np.log(P_0)
	    delta_max = np.amax(np.abs(log_P_f - log_P_0))
	    n = int(np.ceil(delta_max/math.log(P_change_rate)))
	    P_step = []
	    for i in range(len(log_P_0)):
	        P_step.append(np.linspace(log_P_0[i], log_P_f[i],n))
	    P_step = np.exp(P_step)

	    P_element = P_element_0
	    for P in P_step.T:
	        change_p_tot(P)
	        x = update_state(x, t)
	return x





def update_pressure(log_p, element_order):
    for num, name in enumerate(element_order):
        unicon.Monoatom_dict[name].update_log_pressure(log_p[num])
 
            
def extract_x():
    x = []
    for gas in unicon.Monoatom_dict.values():
        x.append(gas.log_pressure)
    for solid in unicon.Existing_solid_species_list:
        x.append(solid.f_pressure)
    for solid in unicon.Existing_solid_solution_species_list:
        x.append(solid.f_pressure)
        X = solid.X.tolist()
        x += X[:-1]
    return x

def input_x(x):
	for num, gas in enumerate(unicon.Monoatom_dict.values()):
	    gas.update_log_pressure(x[num])
	for solid in unicon.Existing_solid_species_list:
	    num += 1
	    solid.update_f_pressure(x[num])
	for solid in unicon.Existing_solid_solution_species_list:
	    num += 1
	    solid.update_f_pressure(x[num])
	    X_len = len(solid.X)
	    num += 1
	    solid.update_bounded_X(x[num: num + X_len - 1])
	    num += X_len - 1

def decode_extration():
    species = {}
    for key, gas in unicon.Monoatom_dict.items():
        species[key] = gas.pressure
    for gas in unicon.Gas_species_list:
        species[gas.name] = gas.pressure    
    for solid in unicon.Existing_solid_species_list:
        species[solid.name] = solid.pressure
    for solid in unicon.Existing_solid_solution_species_list:
        p_endmember = solid.pressure * solid.X
        for endmember, num in solid.endmember.items():
            species[endmember] = p_endmember[num]
    return species

def decode_x():
    x = []
    species = []
    for name in unicon.Monoatom_dict.keys():
        x.append(unicon.Monoatom_dict[name].log_pressure)
        species.append(name)
    for solid in unicon.Existing_solid_species_list:
        x.append(solid.f_pressure)
        species.append(solid.name)
    for solid in unicon.Existing_solid_solution_species_list:
        x.append(solid.f_pressure)
        X = solid.X.tolist()
        x += X[:-1]
        species.append(solid.name)
    return x, species

def encode_x(x, species):
	i = 0
	for name in species:
		if name in unicon.Monoatom_dict:
			unicon.Monoatom_dict[name].update_log_pressure(x[i])
		elif name in unicon.Solid_species_dict:
		    solid = unicon.Solid_species_dict[name]
		    add_solid(solid)
		#             solid.update_f_pressure(x[i])
		elif name in unicon.Solid_solution_species_dict:
		    solid = unicon.Solid_solution_species_dict[name]
		    add_solid_solution(solid)
		    solid.update_f_pressure(x[i])
		    X_len = len(solid.endmember)
		    bounded_X = x[i + 1:i + X_len]
		    solid.update_bounded_X(bounded_X)
		    i += X_len - 1  
		i += 1
    # for gas in unicon.Gas_species_list:
    #         gas.update_pressure(unicon.Monoatom_dict, T)

def mass_balance(x, T):
    input_x(x)
    eqn = []
    for gas in unicon.Gas_species_list:
        gas.update_pressure(unicon.Monoatom_dict, T)
    for element in unicon.Elements.values():
        eqn.append(element.mass_balance()/element.p_tot)
    for solid in unicon.Existing_solid_species_list:
        eqn.append(solid.equilibrium(unicon.Monoatom_dict, T))
    for solid in unicon.Existing_solid_solution_species_list:
        eqn.append(solid.equilibrium(unicon.Monoatom_dict, T))
        prime = solid.equilibrium_prime(unicon.Monoatom_dict, T)
        if type(prime) is np.array:
            eqn += prime.tolist()
        else:
            eqn.append(prime)
    return eqn

def update_state(x, t):
    if check_potential_solid(t) or check_existing_solid(t):
        x = extract_x()
    x = fsolve(mass_balance, x, args = (t))
    while check_potential_solid(t) or check_existing_solid(t):
        x = extract_x()
        x = fsolve(mass_balance, x, args = (t))
    return x



