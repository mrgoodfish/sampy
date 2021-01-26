from .topology import (TopologiesFromFiles,
                       IcosphereTopology,
                       SquareGridWithDiagTopology,
                       SquareGridTopology)
from .vertex_attributes import PeriodicAttributes
from .procedural import AttributeFrom2DGaussianNoise
from ..utils.decorators import sampy_class

import numpy as np
import pandas as pd

import os
import json


@sampy_class
class SquareGridWithDiag(SquareGridWithDiagTopology,
                         PeriodicAttributes,
                         AttributeFrom2DGaussianNoise):
    """
    todo
    """
    def __init__(self, **kwargs):
        pass


@sampy_class
class SquareGrid(SquareGridTopology,
                 PeriodicAttributes,
                 AttributeFrom2DGaussianNoise):
    def __init__(self, **kwargs):
        pass









# class BuiltIn2DGraphs(BuiltInTopologies, VertexAttributes):
#     """
#     Class corresponding to built-in graphs.
#     """
#     def add_procedural_attribute_gaussian_noise(self, attr_name, low_val, high_val, sigma, lvl=1, start_weight=0.5):
#         """
#         Create a new vertex attribute, and generate a random float valued distribution for the attribute. The random
#         distribution is obtained by doing a gaussian smoothing on a 2D white noise with bias (each pixel value of the
#         noise is obtained with the uniform distribution on [0, 1]). The obtained map is then linearly transformed so
#         that it min and max values are low_val and high_val respectively. The map is saved as a new attribute of the
#         graph: 'proc_map_' + attr_name
#
#         :param attr_name: string, name of the attribute
#         :param low_val: min value of the attribute. This value is reached on at least one vertex of the graph.
#         :param high_val: max value of the attribute. This value is reached on at least one vertex of the graph.
#         :param sigma: couple of float, used as a diagonal covariance for the gaussian kernel smoothing.
#         :param lvl: optional, integer, default 1. Number of proc map that are superposed to generate the final map.
#                     Each new map is added to the previous one with an exponentially decreasing weight and covariance.
#                     These kind of techniques are used in video game to generate procedural map (even thought generally
#                     not with gaussian smoothing).
#         :param start_weight: optional, float, default 0.5. Base weight used when lvl variable is different from 1.
#                              The weight of the i-th proc map (starting counting at 0) added to the final map is in this
#                              case: (start_weight)**i.
#         """
#         proc_map = create_random_2d_height_map(self.shape, low_val, high_val, sigma, lvl=lvl, start_weight=start_weight)
#         setattr(self, 'proc_map_' + attr_name, proc_map)
#         attr_array = np.zeros((self.connections.shape[0],))
#         for i in range(self.shape[0]):
#             for j in range(self.shape[1]):
#                 attr_array[i*self.shape[1] + j] = proc_map[i][j]
#         self.create_new_vertex_attribute(attr_name)
#         self.populate_vertex_attribute_with_array(attr_name, attr_array)
#
#     def save_as_repository(self, path_to_rep, name_rep, sep=';'):
#         """
#         Save the graph as a repository containing some csv and json.
#         :param path_to_rep: String, path to the repository where the graph should be saved
#         :param name_rep: String, name of the save file that will be created.
#         :param sep: optional, used as separator in csv. Default ';'
#         """
#         path_dir = path_to_rep + '/' + name_rep
#         os.makedirs(path_dir)
#         # save attributes
#         self.df_attributes.to_csv(path_dir + '/attributes.csv', index=False, sep=sep)
#         # save connections
#         df_con = pd.DataFrame(self.connections)
#         df_con.to_csv(path_dir + '/connections.csv', header=False, index=False, sep=sep)
#         # save weights
#         df_weight = pd.DataFrame(self.weights)
#         df_weight.to_csv(path_dir + '/weights.csv', header=False, index=False, sep=sep)
#         # save some of the metadatas (only shape for the moment)
#         with open(path_dir + '/meta.json', 'w') as f:
#             f.write(json.dumps({'shape': self.shape}))
#
#     @classmethod
#     def from_repository(cls, path_to_rep, sep=';'):
#         """
#         Load the graph from a repository (repository obtained with 'save_as_repository' method
#         :param path_to_rep: String, path to the repository where the graph is saved
#         :param sep: optional, expected separator in csv. Default ';'
#         """
#         # create the returned graph
#         r_graph = cls()
#         # load the attributes
#         r_graph.df_attributes = pd.read_csv(path_to_rep + '/attributes.csv', sep=sep)
#         # load the connections
#         with open(path_to_rep + '/connections.csv', 'r') as f:
#             r_graph.connections = np.array([list(map(int, line.split(sep))) for _, line in enumerate(f)])
#         # load the weights
#         with open(path_to_rep + '/weights.csv', 'r') as f:
#             r_graph.weights = np.array([list(map(float, line.split(sep))) for _, line in enumerate(f)])
#         # load the meta data
#         with open(path_to_rep + '/meta.json', 'r') as f:
#             json_meta = json.load(f)
#             r_graph.shape = tuple(json_meta['shape'])
#             for key, value in json_meta.items():
#                 if key in ['shape']:
#                     continue
#                 setattr(r_graph, key, value)
#         return r_graph


# class IcosphereGraph(IcosphereTopology, VertexAttributes):
#     def __init__(self):
#         super().__init__()
#
#     def save_as_repository(self, path_to_rep, name_rep, sep=';'):
#         """
#         Save the graph as a repository containing some csv and json.
#         :param path_to_rep: String, path to the repository where the graph should be saved
#         :param name_rep: String, name of the save file that will be created.
#         :param sep: optional, used as separator in csv. Default ';'
#         """
#         path_dir = path_to_rep + '/' + name_rep
#         os.makedirs(path_dir)
#         # save attributes
#         self.df_attributes.to_csv(path_dir + '/attributes.csv', index=False, sep=sep)
#         # save connections
#         df_con = pd.DataFrame(self.connections)
#         df_con.to_csv(path_dir + '/connections.csv', header=False, index=False, sep=sep)
#         # save weights
#         df_weight = pd.DataFrame(self.weights)
#         df_weight.to_csv(path_dir + '/weights.csv', header=False, index=False, sep=sep)
#
#     @classmethod
#     def from_repository(cls, path_to_rep, sep=';'):
#         # create the returned graph
#         r_graph = cls()
#         # load the attributes
#         r_graph.df_attributes = pd.read_csv(path_to_rep + '/attributes.csv', sep=sep)
#         # load the connections
#         with open(path_to_rep + '/connections.csv', 'r') as f:
#             r_graph.connections = np.array([list(map(int, line.split(sep))) for _, line in enumerate(f)])
#         # load the weights
#         with open(path_to_rep + '/weights.csv', 'r') as f:
#             r_graph.weights = np.array([list(map(float, line.split(sep))) for _, line in enumerate(f)])
#         return r_graph
#
#
# class GraphFromORMXML(TopologiesFromFiles, VertexAttributes):
#     pass
