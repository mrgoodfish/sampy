from scipy.ndimage import gaussian_filter
import numpy as np
from ..pandas_xs.pandas_xs import DataFrameXS


def create_random_2d_height_map(shape, low_val, high_val, sigma, lvl=1, start_weight=0.5,
                                flatten=False, dict_coord_to_index=None):
    """
    Create a procedural 2D map with a random float valued distribution by pixel. The random distribution is obtained by
    doing a gaussian smoothing on a 2D white noise with bias (each pixel value of the noise is obtained with the uniform
    distribution on [0, 1]). The obtained map is then linearly transformed so that its min and max values are low_val
    and high_val respectively.
    :param shape: couple of integers, corresponds to the dimension of the 2D height map
    :param low_val: min value
    :param high_val: max value
    :param sigma: couple. Diagonal covariance of the gaussian kernel used to obtain the map
    :param lvl: optional, integer, default 1. Number of proc map that are superposed to generate the final map.
                Each new map is added to the previous one with an exponentially decreasing weight and covariance.
                These kind of techniques are used in video game to generate procedural map (even thought generally
                not with gaussian smoothing of white noise).
    :param start_weight: optional, float, default 0.5. Base weight used when lvl variable is different from 1.
                         The weight of the i-th proc map (starting counting at 0) added to the final map is in this
                         case: (start_weight)**i.
    :param flatten: optional, boolean, default False. If True, the function returns two value, the first one being the
                    height map and the second being this 2d map turned into a 1d array using the optional parameter
                    'dict_coord_to_index'. WARNING: an error will be raised if a value for 'dict_coord_to_index' is not
                    given while using this option.
    :param dict_coord_to_index: optional, dictionary, default None. Used only if the parameter 'flatten' is set to True
    """
    rv = np.zeros(shape)
    for i in range(lvl):
        weight = start_weight**i
        new_arr = np.random.uniform(0, 1, shape)
        new_arr = gaussian_filter(new_arr, sigma=(sigma[0]*weight, sigma[1]*weight))
        rv += weight*new_arr
    minimum = rv.min()
    maximum = rv.max()
    rv = (high_val - low_val)*((rv - minimum)/(maximum-minimum)) + low_val
    if flatten:
        if dict_coord_to_index is None:
            raise ValueError('A value for the parameter dict_coord_to_index has to be given if flatten is set to True')
        flat_rv = np.zeros(rv.flatten().shape, dtype=np.float)
        for coord, index in dict_coord_to_index.items():
            flat_rv[index] = rv[coord[0]][coord[1]]
        return rv, flat_rv
    return rv


class AttributeFrom2DGaussianNoise:
    def __init__(self, **kwargs):
        pass

    def _sampy_debug_populate_attribute_with_2d_gaussian_noise(self, name_attr, low_val, high_val, sigma, lvl=1,
                                                               start_weight=0.5, return_map=False):
        if not hasattr(self, 'df_attributes'):
            raise AttributeError("The graph does not have an attribute df_attributes.")
        # if not isinstance(self.df_attributes, DataFrameXS):
        #     raise AttributeError("'df_attributes' should be a DataFrameXS.")
        # if name_attr not in self.df_attributes.dict_colname_to_index:
        #     raise KeyError("The attribute " + name_attr + " is not a column of df_attribute DataFrameXS.")
        if not isinstance(lvl, int):
            raise ValueError("The parameter 'lvl' should be an integer")
        if not hasattr(self, 'shape'):
            raise AttributeError("The current graph object has no attribute 'shape'.")
        if len(self.shape) != 2:
            raise ValueError("The attribute 'shape' is not of length 2.")
        if not isinstance(self.shape[0], int) and not isinstance(self.shape[1], int):
            raise ValueError("The attribute 'shape' is not a couple of integer")

    def populate_attribute_with_2d_gaussian_noise(self, name_attr, low_val, high_val, sigma, lvl=1, start_weight=0.5,
                                                  return_map=False):
        """
        Create a procedural 2D map with a random float valued distribution by pixel. The random distribution is obtained
        by doing a gaussian smoothing on a 2D white noise with bias (each pixel value of the noise is obtained with the
        uniform distribution on [0, 1]). The obtained map is then linearly transformed so that its min and max values
        are low_val and high_val respectively. Finally, this 2d map is flattened and used to populate the chosen
        attribute of the graph.

        WARNING: this method expects to be called from an object having the following two attributes:
                - shape: tupple like object of length 2.
                - dict_cell_id_to_ind: a dictionary like object, whose keys are pairs of integers (a, b), and values
                                       are the indexes of the corresponding vertex.
                 This is checked in debug mode.

        :param name_attr: string, name of the attribute of the graph to populate.
        :param low_val: min value
        :param high_val: max value
        :param sigma: couple. Diagonal covariance of the gaussian kernel used to obtain the map
        :param lvl: optional, integer, default 1. Number of proc map that are superposed to generate the final map.
                    Each new map is added to the previous one with an exponentially decreasing weight and covariance.
                    These kind of techniques are used in video game to generate procedural map (even thought generally
                    not with gaussian smoothing of white noise).
        :param start_weight: optional, float, default 0.5. Base weight used when lvl variable is different from 1.
                             The weight of the i-th proc map (starting counting at 0) added to the final map is in this
                             case: (start_weight)**i.
        :param return_map: optional, boolean, default False. If True, the function returns the generated 2d map.

        :return: if return_map is True, returns the 2d map generated to populate the attribute. Otherwise, returns None.
        """
        proc_map = np.zeros(self.shape[0:2])
        for i in range(lvl):
            weight = start_weight ** i
            new_arr = np.random.uniform(0, 1, self.shape[0:2])
            new_arr = gaussian_filter(new_arr, sigma=(sigma[0] * weight, sigma[1] * weight))
            proc_map += weight * new_arr
        minimum = proc_map.min()
        maximum = proc_map.max()
        proc_map = (high_val - low_val) * ((proc_map - minimum) / (maximum - minimum)) + low_val
        attr_array = np.zeros((self.connections.shape[0],))
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                attr_array[i*self.shape[1] + j] = proc_map[i][j]
        self.populate_vertex_attribute_with_array(name_attr, attr_array)
        if return_map:
            return proc_map
