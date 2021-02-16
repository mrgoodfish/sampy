from .misc import (create_grid_hexagonal_cells,
                   create_grid_square_cells,
                   create_grid_square_with_diagonals,
                   SubdividedIcosahedron)
from sampy.utils.jit_compiled_functions import create_image_from_count_array_builtin_graph
from .jit_compiled_functions import (get_oriented_neighborhood_of_vertices,
                                     get_surface_array)
import xmltodict

import numpy as np
from math import sqrt, pi
import pandas as pd

from geopy import distance


class BaseTopology:
    def __init__(self, **kwargs):
        self.connections = None
        self.weights = None
        self.type = None
        self.dict_cell_id_to_ind = {}
        self.time = 0 # todo: it seems extremely strange for this attribute to be in 'topology'... Add a subpack 'time'

        self.on_ticker = ['increment_time']

    def increment_time(self):
        self.time += 1

    def tick(self):
        """
        execute the methods whose names are stored in the attribute on_ticker, in order.
        """
        for method in self.on_ticker:
            getattr(self, method)()

    def save_table_id_of_vertices_to_indices(self, path_to_csv, sep):
        """

        :param path_to_csv:
        :param sep:
        """
        with open(path_to_csv, 'a') as f_out:
            f_out.write("id_vertex" + sep + "index_vertex" + "\n")
            for id_vertex, index in self.dict_cell_id_to_ind.items():
                f_out.write(str(id_vertex) + sep + str(index) + '\n')
        return

    @property
    def number_vertices(self):
        return self.weights.shape[0]


class SquareGridWithDiagTopology(BaseTopology):
    def __init__(self, shape=None, **kwargs):
        if shape is None:
            raise ValueError("Kwarg 'shape' is missing while initializing the graph topology. 'shape' should be a "
                             "tuple like object of the form (a, b), where a and b are integers bigger than 1.")
        len_side_a = shape[0]
        len_side_b = shape[1]
        self.create_square_with_diag_grid(len_side_a, len_side_b)
        self.shape = (len_side_a, len_side_b)
        self.type = 'SquareGridWithDiag'

    def create_square_with_diag_grid(self, len_side_a, len_side_b):
        """
        Create a square grid with diagonals, where each vertex X[i][j] is linked to X[i-1][j-1], X[i][j-1], X[i+1][j-1],
        X[i+1][j], X[i+1][j+1], x[i][j+1], x[i-1][j+1] and x[i-1][j] if they exist. Note that the weights on the
        'diagonal connections' is reduced to take into account the fact that the vertices on the diagonal are 'further
        away' (i.e. using sqrt(2) as a distance instead of 1 in the weight computation).

        :param len_side_a: integer, x coordinate
        :param len_side_b: integer, y coordinate
        """
        if (len_side_a < 2) or (len_side_b < 2):
            raise ValueError('side length attributes for HexagonalCells should be at least 2.')

        self.connections, self.weights = create_grid_square_with_diagonals(len_side_a, len_side_b)

        # populate the dictionary from cell coordinates to cell indexes in arrays connection and weights
        for i in range(len_side_a):
            for j in range(len_side_b):
                self.dict_cell_id_to_ind[(i, j)] = j + i*len_side_b


class SquareGridTopology(BaseTopology):
    def __init__(self, shape=None, **kwargs):
        if shape is None:
            raise ValueError("Kwarg 'shape' is missing while initializing the graph topology. 'shape' should be a "
                             "tuple like object of the form (a, b), where a and b are integers bigger than 1.")
        len_side_a = shape[0]
        len_side_b = shape[1]
        self.create_square_grid(len_side_a, len_side_b)
        self.shape = (len_side_a, len_side_b)
        self.type = 'SquareGridWithDiag'

    def create_square_grid(self, len_side_a, len_side_b):
        """
        Create a square grid, where each vertex X[i][j] is linked to X[i-1][j], X[i][j-1], X[i+1][j], X[i][j+1] if they
        exist.

        :param len_side_a: integer, x coordinate
        :param len_side_b: integer, y coordinate
        """
        if (len_side_a < 2) or (len_side_b < 2):
            raise ValueError('side length attributes for HexagonalCells should be at least 2.')

        self.connections, self.weights = create_grid_square_cells(len_side_a, len_side_b)
        self.shape = (len_side_a, len_side_b)

        # populate the dictionary from cell coordinates to cell indexes in arrays connection and weights
        for i in range(len_side_a):
            for j in range(len_side_b):
                self.dict_cell_id_to_ind[(i, j)] = j + i*len_side_b


class IcosphereTopology(BaseTopology):
    def __init__(self):
        super().__init__()

    def create_topology(self, nb_sub):
        self.icosphere = SubdividedIcosahedron(nb_sub)
        self.connections = self.icosphere.connections
        self.weights = self.icosphere.weights
        self.df_attributes = pd.DataFrame()
        self.type = 'IcoSphere'

    def create_3d_coord_on_sphere(self, on_sphere=True):
        self.df_attributes['coord_x'] = self.icosphere.arr_coord[:, 0]
        self.df_attributes['coord_y'] = self.icosphere.arr_coord[:, 1]
        self.df_attributes['coord_z'] = self.icosphere.arr_coord[:, 2]
        if on_sphere:
            norm = np.sqrt(self.df_attributes['coord_x']**2 + self.df_attributes['coord_y']**2 +
                           self.df_attributes['coord_z']**2)
            self.df_attributes['coord_x'] = self.df_attributes['coord_x'] / norm
            self.df_attributes['coord_y'] = self.df_attributes['coord_y'] / norm
            self.df_attributes['coord_z'] = self.df_attributes['coord_z'] / norm

    def create_pseudo_epsg4326_coordinates(self):
        """
        This method approximate the shape of the earth using a sphere, which creates deformations.
        """
        self.create_3d_coord_on_sphere(on_sphere=True)
        self.df_attributes['lat'] = 180*(pi/2 - np.arccos(self.df_attributes['coord_z']))/pi
        self.df_attributes['lon'] = 180*np.arctan2(self.df_attributes['coord_y'], self.df_attributes['coord_x'])/pi

    def coords_on_spherical_earth(self):
        """
        Add new attributes, coord_x_earth, coord_y_earth and coord_z_earth. The unit is kilometer.
        This method approximate the shape of the earth using a sphere of radius 6371km, which creates deformations.
        """
        self.create_3d_coord_on_sphere(on_sphere=True)
        self.df_attributes['coord_x_earth'] = 6371.009 * self.df_attributes['coord_x']
        self.df_attributes['coord_y_earth'] = 6371.009 * self.df_attributes['coord_y']
        self.df_attributes['coord_z_earth'] = 6371.009 * self.df_attributes['coord_z']

    def compute_distance_matrix_on_earth(self, mode='great-circle'):
        """
        todo
        :param mode: string, optional. Chose the mode used to compute distance. Acceptable values:
                    - 'great-circle': compute the distance assuming the earth is a perfect sphere. uses Geopy.
                    - 'ellipsoid': compute the distance assuming the earth is an ellipsoid (model WGS-84), uses Geopy
        :return: Array of floats with the same shape as the array 'connections'
        """
        mode = str(mode)
        dist_matrix = np.full(self.connections.shape, -1, dtype=np.float32)
        if mode == 'ellipsoid':
            for i in range(self.connections.shape[0]):
                coord_p1 = (self.df_attributes['lat'][i], self.df_attributes['lon'][i])
                for j in range(self.connections.shape[1]):
                    ind_neighbour = self.connections[i][j]
                    if ind_neighbour == -1:
                        # we could break, but we do this to prevent potential future bugs.
                        continue
                    coord_p2 = (self.df_attributes['lat'][ind_neighbour],
                                self.df_attributes['lon'][ind_neighbour])
                    dist_matrix[i][j] = distance.distance(coord_p1, coord_p2).km
        elif mode == 'great-circle':
            for i in range(self.connections.shape[0]):
                coord_p1 = (self.df_attributes['lat'][i], self.df_attributes['lon'][i])
                for j in range(self.connections.shape[1]):
                    ind_neighbour = self.connections[i][j]
                    if ind_neighbour == -1:
                        # we could break, but we do this to prevent potential future bugs.
                        continue
                    coord_p2 = (self.df_attributes['lat'][ind_neighbour],
                                self.df_attributes['lon'][ind_neighbour])
                    dist_matrix[i][j] = distance.great_circle(coord_p1, coord_p2).km
        else:
            raise ValueError('unrecognized mode parameter: ' + str(mode))
        return dist_matrix

    def compute_surface_array(self, radius):
        """
        Return an array giving the surface of each cell, each cell being centered on the 
        :return: array of floats shape (nb_vertex,)
        """
        oriented_neigh_vert = get_oriented_neighborhood_of_vertices(self.connections)
        x = np.array(self.df_attributes['coord_x'])
        y = np.array(self.df_attributes['coord_y'])
        z = np.array(self.df_attributes['coord_z'])
        return get_surface_array(oriented_neigh_vert, x, y, z, radius)

    def create_and_save_surface_array_as_attribute(self, radius):
        arr_surface = self.compute_surface_array(radius)
        self.df_attributes['surface_cell'] = arr_surface


class TopologiesFromFiles(BaseTopology):
    def __init__(self):
        self.type = 'FromFile'
        self.has_coord = False
        self.coord_x = None
        self.coord_y = None
        super().__init__()

    def create_topology_from_orm_xml(self, path_to_xml, create_coord=True, ignored_cells=None):
        with open(path_to_xml, 'rb') as xml_file:
            dic_graph = xmltodict.parse(xml_file)

        list_cells = dic_graph['NewDataSet']['AllCellData']

        # we first give an ID to all the relevant cells
        current_ind = 0
        for i, cell in enumerate(list_cells):
            id_cell = cell['HEXID']
            if ignored_cells and (id_cell in ignored_cells):
                continue
            else:
                self.dict_cell_id_to_ind[id_cell] = current_ind
                current_ind += 1

        dict_id_neighbours = {}
        for i, cell in enumerate(list_cells):
            if cell['HEXID'] not in self.dict_cell_id_to_ind:
                continue
            ind_cell = self.dict_cell_id_to_ind[cell['HEXID']]

            dict_id_neighbours[ind_cell] = {}
            # deal with the neighbours
            for key in cell:
                if key in {'NE', 'NW', 'N', 'S', 'SW', 'SE'}:
                    # check if the cell already has an id
                    if cell[key] in self.dict_cell_id_to_ind:
                        dict_id_neighbours[ind_cell][key] = self.dict_cell_id_to_ind[cell[key]]

        # for key in dict_id_neighbours:
        #     if not dict_id_neighbours[key]:
        #         print([u for u in self.dict_cell_id_to_ind if key == self.dict_cell_id_to_ind[u]])

        max_nb_connection = max([len(neighb) for neighb in dict_id_neighbours.values()])
        self.weights = np.zeros((len(self.dict_cell_id_to_ind), max_nb_connection), dtype=np.float)
        self.connections = np.zeros((len(self.dict_cell_id_to_ind), max_nb_connection), dtype=np.int)

        for cell_ind, neighbours in dict_id_neighbours.items():
            nb_neigh = len(neighbours)
            for i, direction in enumerate(neighbours):
                self.weights[cell_ind][i] = (i+1)/nb_neigh
                self.connections[cell_ind][i] = neighbours[direction]

        # we create artificial coords of the cells, in order to draw som graphs
        # todo: replace this shit by a proper recursion.
        if create_coord:
            self.has_coord = True
            set_cell_having_coord = {0}
            self.coord_x = np.zeros((len(self.dict_cell_id_to_ind),), dtype=np.float)
            self.coord_y = np.zeros((len(self.dict_cell_id_to_ind),), dtype=np.float)
            while len(set_cell_having_coord) < len(self.dict_cell_id_to_ind):
                temp_set = set(set_cell_having_coord)
                for cell_in in set_cell_having_coord:
                    for direction, ind_neighbour in dict_id_neighbours[cell_in].items():
                        if ind_neighbour not in temp_set:
                            if direction == 'N':
                                self.coord_x[ind_neighbour] = self.coord_x[cell_in]
                                self.coord_y[ind_neighbour] = self.coord_y[cell_in] + 1.
                                temp_set.add(ind_neighbour)
                            if direction == 'NE':
                                self.coord_x[ind_neighbour] = self.coord_x[cell_in] + sqrt(3)/2
                                self.coord_y[ind_neighbour] = self.coord_y[cell_in] + 0.5
                                temp_set.add(ind_neighbour)
                            if direction == 'SE':
                                self.coord_x[ind_neighbour] = self.coord_x[cell_in] + sqrt(3)/2
                                self.coord_y[ind_neighbour] = self.coord_y[cell_in] - 0.5
                                temp_set.add(ind_neighbour)
                            if direction == 'S':
                                self.coord_x[ind_neighbour] = self.coord_x[cell_in]
                                self.coord_y[ind_neighbour] = self.coord_y[cell_in] - 1.
                                temp_set.add(ind_neighbour)
                            if direction == 'SW':
                                self.coord_x[ind_neighbour] = self.coord_x[cell_in] - sqrt(3)/2
                                self.coord_y[ind_neighbour] = self.coord_y[cell_in] - 0.5
                                temp_set.add(ind_neighbour)
                            if direction == 'NW':
                                self.coord_x[ind_neighbour] = self.coord_x[cell_in] - sqrt(3)/2
                                self.coord_y[ind_neighbour] = self.coord_y[cell_in] + 0.5
                                temp_set.add(ind_neighbour)
                set_cell_having_coord = set_cell_having_coord | temp_set
