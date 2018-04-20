import numpy as np
from matplotlib import pyplot as plt
from pystan.external.pymc import plots
import sys

if sys.version_info[0] == 3:
    def xrange(i):
        return range(i)

def vb_extract(fit):
    var_names = fit["sampler_param_names"]
    samples = np.array([x for x in fit["sampler_params"]])
    
    samples_dict = {}
    means_dict = {}
    for i in xrange(len(var_names)):
        samples_dict[var_names[i]] = samples[i,:]
        means_dict[var_names[i]] = fit["mean_pars"][i]
        
    return samples_dict, means_dict, var_names


def vb_extract_variable(fit, var_name, var_type="real", dims=None):
    if var_type == "real":
        return fit["mean_pars"][fit["sampler_param_names"].index(var_name)]
    elif var_type == "vector":
        vec = []
        for i in xrange(len(fit["sampler_param_names"])):
            if var_name+"." in fit["sampler_param_names"][i]:
                vec.append(fit["mean_pars"][i])
        return np.array(vec)
    elif var_type == "matrix":
        if dims == None:
            raise Exception("For matrix variables, you must specify a 'dims' parameter")
        C, D = dims
        mat = []
        for i in xrange(len(fit["sampler_param_names"])):
            if var_name+"." in fit["sampler_param_names"][i]:
                mat.append(fit["mean_pars"][i])
        mat = np.array(mat).reshape(C, D, order='F')
        return mat
    else:
        raise Exception("Unknown variable type: %s. Valid types are: real, vector and matrix" % (var_type,))


def vb_plot_variables(fit, var_names):
    samples, means, names = vb_extract(fit)

    if type(var_names) == str:
        var_names = [var_names]
    elif type(var_names) != list:
        raise Exception("Invalid argument type for var_names")

    to_plot = []
    for var in var_names:
        for i in xrange(len(fit["sampler_param_names"])):
            if var == fit["sampler_param_names"][i] or var+"." in fit["sampler_param_names"][i]: 
                to_plot.append(fit["sampler_param_names"][i])

    for var in to_plot:
        plots.kdeplot_op(plt, samples[var])
    plt.legend(to_plot)
    plt.show()


def report(fit, prefix=''):
    for param in fit['sampler_param_names']:
        if param.startswith(prefix):
            print(param, "=", vb_extract_variable(fit, var_name=param))
        
