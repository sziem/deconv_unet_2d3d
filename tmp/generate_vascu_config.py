#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 15:06:35 2018

@author: soenke
"""
import numpy as np
from warnings import warn

# utility functions:
def arr_to_str(a):
    res = str()
    final = len(a)-1
    for i, el in enumerate(a):
        res += str(el)
        if i == final:
            return res
        else:
            res += " "
    warn("The function arr_to_str returned nothing.")
    

def generate_config_files(n_samples, im_shape, n_term_nodes_max=1000):
    """
    im_shape in order x, y, z
    """
    # provide names of samples and where to find their configuration
    image_names = list()
    demand_files = list()
    param_files = list()
    
    for i in range(n_samples):
        image_names.append("image" + str(i))
        demand_files.append("demand" + str(i) + ".txt")
        param_files.append("p" + str(i) + ".txt")
     
    np.savetxt("image_names.txt", image_names, fmt='%s')
    np.savetxt("param_files.txt", param_files, fmt='%s')
    print("have written image_names.txt and image_param_files.txt")    

    # provide supply map (how existing supply impacts demand).
    # Since I am not sure how to use it, it is the same for all structures.
    supply_pairs = generate_supply_map(im_shape)
    write_supply_map("supply.txt", supply_pairs, im_shape)
    print("have written supply.txt") 

    # TODO: play with params to know effects before using random params
    # currently randomized perf_point, n_term_nodes and demand_map
    for i in range(n_samples):
        demand_pair = generate_random_demand_pairs(im_shape, box_size=20)
        write_demand_map(demand_files[i], demand_pair, im_shape)
        print("have written",  demand_files[i]) 
        write_param_file(param_files[i],
                         perf_point=(np.random.randint(0, im_shape[0]),
                                     np.random.randint(0, im_shape[1]),
                                     np.random.randint(0, im_shape[2])),
                         perf_pressure=2*133000,
                         term_pressure_divisor=1.6,
                         n_term_nodes=np.random.randint(0, n_term_nodes_max),
                         demand_file_name=demand_files[i],
                         random_seed=i+1,
                         supply_file_name="supply.txt")
        print("have written",  param_files[i])

# %% 
def generate_supply_map(im_shape):
    top_left, bottom_right = (0, 0, 0), tuple(np.array(im_shape)-1)  # x, y, z
    pair = dict(box=top_left+bottom_right, coeff=(0.65, 0.34, 0.01, 7))
    return list((pair,))

#def generate_params():
#    pass

def generate_random_demand_pairs(im_shape, box_size=1):
    """
    Generates array of random floats of shape im_shape drawn from uniform 
    distribution in the half-open interval 0...1 and returns it in the correct
    format for input into write_demand_pairs (a list of dicts).
    
    box_size (int or tuple):  Files generated from this may become quite large.  
    You can specify that not every pixel has a different value, but boxes 
    of box_size to reduce filesize.  If im_shape is not a multiple of
    box_size in some direction, then smaller boxes are used on the right, 
    bottom, and back.
    """
    if isinstance(box_size, int):
        box_size = (box_size, box_size, box_size)
    demand_map = np.random.random(im_shape).astype(np.float16)
    demand_pairs = list()
    # single box for every pixel  -> very expensive in memory!
    for i in range(0, im_shape[0], box_size[0]):
        for j in range(0, im_shape[1], box_size[1]):
            for k in range(0, im_shape[2], box_size[1]):
                if i+box_size[0]-1 < im_shape[0]:
                    box = (i,j,k,i+box_size[0]-1)
                else:
                    box = (i,j,k,im_shape[0]-1)
                if j+box_size[1]-1 < im_shape[1]:
                    box += (j+box_size[1]-1,)
                else:
                    box += (im_shape[1]-1,)
                if k+box_size[2]-1 < im_shape[2]:
                    box += (k+box_size[2]-1,)
                else:
                    box += (im_shape[2]-1,)
                demand = demand_map[i,j,k]
                demand_pairs.append(dict(box=box, demand=demand))
    
    return demand_pairs
                



# %% write configurations to files in the right way
# TODO: Does the arg order always have to be in this order?


    
    #TODO: change values bleow after understanding
def write_param_file(file_name,
                     perf_point=(0,0,0),
                     perf_pressure=133000,
                     term_pressure_divisor=1.6,
                     n_term_nodes=200,
                     ratio=2,
                     perf_flow=8.33,
                     viscosity=0.036,  #blood
                     gamma=3,
                     min_distance=1,
                     voxel_width=0.04,
                     closest_neighbors=5,
                     random_seed=0,
                     supply_file_name="O2_supply.txt", 
                     demand_file_name="O2_demand.txt"):
    """
    vary to simulate structures:
        perf_point (integer triplet): Location of root branch
            Default: (0,0,0)
        perf_pressure (float): pressure at the perf_point in µmHg
            High pressure leads to thinner branches
            Default: 133000 µmHg  (slightly heightened blood pressure)
        term_pressure_divisor->term_pressure (float >= 1): 
            pressure at the terminal nodes must be lower than perf_pressure 
            This parameter defines, how much the radii of the branches
            shrink from the root radius.  If term_pressure
            Default: 1.6 (roughly leads to term_pressure=83000 µmHg as given
            in the example param file)
        n_term_nodes->num_nodes (int): number of terminal nodes
            default: 200 (from paper)
        ratio->lambda,mu (float): lambda / mu
            ratio of exponents of the objective function
            loss = sum_over_segments(length_j**mu + radius_j**lambda)
            which is minimized in candidate node selection.
            Here: mu is always set to 1 and lambda is set to ratio.
            A higher ratio leads to longer, but thinner branches
            Default: 2 (minimize volumes of cylindrical segments)
        demand_file_name->oxygenation_map (string to txt-file): 
            high demand increases probability 
            of terminal node creation. zero demand forbids branch here
            Default: "O2_demand.txt"
    
    keep as provided below (unless you know what you are doing).  
    Effect of some parameters may correlate with effect of above parameters:
        perf_flow (float): flow at perf_point in m**3/(h*kg)
            flow, pressure and radius are related
            Default: 8.33 m**3/(h*kg)  
            # NOT SURE WHY THERE IS KG HERE
            # NOT SURE WHERE DEFAULT VALUE COMES FROM BUT OTHER PARAMS WERE REASONABLE
        viscosity->rho (float): viscosity of the fluid in Pa*s
            Default: 0.036 (blood)
        gamma (float): impacts radii of child branches at a bifurcation location
            as in r_parent**gam = r_child1**gam + r_child2**gam
            Default: ~3 (empirical). 
        random_seed (integer, optional): with the same random_seed, same params 
            will yield same structure.  If it is not set or zero, clock time
            is used as random_seed
            Default:  0   (i.e. clock time is used)
        min_distance (integer or float??): minimum distance in mm between a 
            bifurcation point and a candidate node.
            integer according to docs, but shouldn't it be float??
            Default:  1 mm
        voxel_width (float): width of a voxel in mm. Cuboid Voxels are assumed.
            The voxel width parameter of the call to vascu_synth may be 
            different to simulate false sampling.
            Default: 0.04 mm
        closest_neighbors->closest_neighbours (int): number of closest 
            segments considered in  candidate node selection (lower to reduce 
            complexity).
            Default:  5
        supply_file_name (string to txt-file): how the supply to one terminal node 
            impacts demand in surrounding nodes.  Not sure about details
            Default: (0.65, 0.34, 0.01, 7) in the entire image
            
    Note: most of them have reasonable defaults, but in the end i am just 
    interested in getting lots of data
    """
    
    contents = list()
    contents.append("SUPPLY_MAP: " + supply_file_name)  # separate file
    contents.append("OXYGENATION_MAP: " + demand_file_name)  # separate file
    contents.append("RANDOM_SEED: " + str(random_seed))  # optional.
    contents.append("PERF_POINT: " + arr_to_str(perf_point))  # loc of root
    contents.append("PERF_PRESSURE: " + str(perf_pressure)) # µmHg
    contents.append("TERM_PRESSURE: " + str(perf_pressure/term_pressure_divisor)) # µmHg, 
    contents.append("PERF_FLOW: " + str(perf_flow)) # m**3/(h*kg)
    contents.append("RHO: " + str(viscosity)) # viscosity in Pa*s 
    contents.append("GAMMA: " + str(gamma)) # 
    contents.append("LAMBDA: " + str(ratio)) # 
    contents.append("MU: 1") # see docstring for ratio for why this is reasonable
    contents.append("MIN_DISTANCE: " + str(min_distance)) # mm
    contents.append("NUM_NODES: " + str(n_term_nodes)) # no. terminal nodes
    contents.append("VOXEL_WIDTH: " + str(voxel_width)) # mm
    contents.append("CLOSEST_NEIGHBOURS: " + str(closest_neighbors)) #
    np.savetxt(file_name, contents, fmt="%s")    

# oxygen demand and supply impact
# TODO: could im_sh be different btw demand and supply map?
# TODO: do you need to set coefficients for the entire box in supply map?  
# what happens otherwise?  Default or exception?    

def write_demand_map(file_name, demand_pairs, im_sh, box_size=1):
    """
    writes 
    pairs: list of dicts containing "box" and "demand" fields.
         "box": indices of the top left (first 3 numbers)
             and bottom right (last 3 numbers)
             of a box using the same coefficients.
             (coordinates in the order x, y, z)
         "demand": number between 0...1, where 0 forbids a vascular structure
             and 1 is high demand.
    """
    # first line contains parsing info (max image indices)    
    demand_map = list()
    demand_map.append(arr_to_str(np.array(im_sh)-1))
    for pair in demand_pairs:
        demand_map.append(arr_to_str(pair["box"]))
        demand_map.append(str(pair["demand"]))
    np.savetxt(file_name, demand_map, fmt="%s")

def write_supply_map(file_name, supply_pairs, im_sh):
    """
    pairs: list of dicts containing "box" and "coeff" fields.
         All pairs must have the same number of coeffs.
         "box": indices of the top left (first 3 numbers)
             and bottom right (last 3 numbers)
             of a box using the same coefficients.
             (coordinates in the order x, y, z)
         "coeffs": The coeffs specify how existing supply impacts demand in box.
             Am not sure about the exact funcional form the coeffs represent.
    """
    # first line contains parsing info (max image indices and n_coeffs to use)
    supply_map = list()
    n_coeff = len(supply_pairs[0]["coeff"])  # init
    supply_map.append(arr_to_str(np.array(im_sh)-1) + " " + str(n_coeff))

    for pair in supply_pairs:
        box = pair["box"]
        coeff = pair["coeff"]
        if len(coeff) != n_coeff:
            msg = ("the same number of coefficients must be provided for " +
                   "all boxes in supply_map.")
            raise ValueError(msg)
        supply_map.append(arr_to_str(box))
        supply_map.append(arr_to_str(coeff))
    np.savetxt(file_name, supply_map, fmt="%s")

def main():
    # hyperparams:
    n_samples = 10
    im_sh = (200, 200, 200)  # x, y, z; shape of image, i.e indices 0...199
    generate_config_files(n_samples, im_sh)

if __name__ == '__main__':
    main()

# This took me too long already.  Will just create random map
#def generate_demand_pairs(n_maps, im_shape):    
#    # 0 is low, 1 is high. --> TODO: can i use e.g 0.5?
#    # values given later override those given before
#    top_left, bottom_right = (0, 0, 0), tuple(np.array(im_shape)-1)  # x, y, z
#    base = dict(box=top_left+bottom_right, demand=1)
#    
#    demand_pairs = list()
#    
#    #uniform demand
#    demand_pairs.append(list(base,))        
#    
#    #exclude left_top_front
#    top_left_sub = top_left
#    bottom_right_sub = tuple((np.array(im_shape)-1)//2
#    demand_pairs.append(list(base,dict(box=top_left+bottom_right, demand=0))  
#
#    demand_pairs.append()

#    if n_maps > len(demand_pairs):
#        msg = ("only " + str(len(demand_pairs)) + " demand maps are " + 
#               "available and you want " + str(n_maps) + ".")
#        raise ValueError(msg)
#
#    # this is not very efficient, but who cares ...
#    return demand_maps[:n_maps]    
    
#    box1 = (0, 0, 0, 99, 99, 99)  # indices of the top left (first 3 numbers)
#                                 # and bottom right (last 3 numbers)
#                                 # of a box with constant O2 demand.
#    demand1 = 1  # demand in box 1.  
#    # 0 is low, 1 is high. --> TODO: can i use e.g 0.5?

#    box2 = (0, 34, 34, 99, 44, 44)  # indices of the top left (first 3 numbers)
#                                 # and bottom right (last 3 numbers)
#                                 # of a box with constant O2 demand.
#    demand2 = 0  # 0 is low, 1 is high.  --> TODO: can i use e.g 0.5?


