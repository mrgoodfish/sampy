import numpy as np
from math import radians, cos, sin, asin, sqrt


def create_grid_hexagonal_cells(len_side_a, len_side_b):
    """
    Note : this function is kind of broken, in the sense that the obtained map is heavily deformed. Should be changed.
    :param len_side_a:
    :param len_side_b:
    :return: two numpy arrays of shape (len_side_a*len_side_b, 6)
    """
    connections = np.zeros((len_side_a * len_side_b, 6), dtype=np.int32)
    weights = np.zeros((len_side_a * len_side_b, 6), dtype=np.float64)

    # first 4 corner
    connections[0][0] = 1
    weights[0][0] = 0.5
    connections[0][1] = len_side_b
    weights[0][1] = 1.0

    connections[len_side_b - 1][0] = len_side_b - 2
    weights[len_side_b - 1][0] = 1 / 3
    connections[len_side_b - 1][1] = 2 * len_side_b - 2
    weights[len_side_b - 1][1] = 2 / 3
    connections[len_side_b - 1][2] = 2 * len_side_b - 1
    weights[len_side_b - 1][2] = 1.0

    connections[len_side_b * (len_side_a - 1)][0] = len_side_b * (len_side_a - 2)
    weights[len_side_b * (len_side_a - 1)][0] = 1 / 3
    connections[len_side_b * (len_side_a - 1)][1] = len_side_b * (len_side_a - 2) + 1
    weights[len_side_b * (len_side_a - 1)][1] = 2 / 3
    connections[len_side_b * (len_side_a - 1)][2] = len_side_b * (len_side_a - 1) + 1
    weights[len_side_b * (len_side_a - 1)][2] = 1.

    connections[len_side_a * len_side_b - 1][0] = (len_side_a - 1) * len_side_b - 1
    weights[len_side_a * len_side_b - 1][0] = 0.5
    connections[len_side_a * len_side_b - 1][1] = len_side_a * len_side_b - 2
    weights[len_side_a * len_side_b - 1][1] = 1.0

    # take care of the 4 borders
    for i in range(1, len_side_a - 1):
        ind_vert = len_side_b * i
        connections[ind_vert][0] = ind_vert - len_side_b
        weights[ind_vert][0] = 0.25
        connections[ind_vert][1] = ind_vert - len_side_b + 1
        weights[ind_vert][1] = 0.5
        connections[ind_vert][2] = ind_vert + 1
        weights[ind_vert][2] = 0.75
        connections[ind_vert][3] = ind_vert + len_side_b
        weights[ind_vert][3] = 1.0

    for i in range(1, len_side_a - 1):
        ind_vert = len_side_b * (i + 1) - 1
        connections[ind_vert][0] = ind_vert - len_side_b
        weights[ind_vert][0] = 0.25
        connections[ind_vert][1] = ind_vert - 1
        weights[ind_vert][1] = 0.5
        connections[ind_vert][2] = ind_vert + len_side_b - 1
        weights[ind_vert][2] = 0.75
        connections[ind_vert][3] = ind_vert + len_side_b
        weights[ind_vert][3] = 1.0

    for i in range(1, len_side_b - 1):
        ind_vert = i
        connections[ind_vert][0] = ind_vert - 1
        weights[ind_vert][0] = 0.25
        connections[ind_vert][1] = ind_vert + 1
        weights[ind_vert][1] = 0.5
        connections[ind_vert][2] = ind_vert + len_side_b - 1
        weights[ind_vert][2] = 0.75
        connections[ind_vert][3] = ind_vert + len_side_b
        weights[ind_vert][3] = 1.0

    for i in range(1, len_side_b - 1):
        ind_vert = (len_side_a - 1) * len_side_b + i
        connections[ind_vert][0] = ind_vert - len_side_b
        weights[ind_vert][0] = 0.25
        connections[ind_vert][1] = ind_vert - len_side_b + 1
        weights[ind_vert][1] = 0.5
        connections[ind_vert][2] = ind_vert - 1
        weights[ind_vert][2] = 0.75
        connections[ind_vert][3] = ind_vert + 1
        weights[ind_vert][3] = 1.0

    # making graph structure of the center of the graph
    for i in range(len_side_a - 2):
        for j in range(len_side_b - 2):
            ind_vert = len_side_b * (i + 1) + (j + 1)
            connections[ind_vert][0] = ind_vert - len_side_b
            weights[ind_vert][0] = 1 / 6
            connections[ind_vert][1] = ind_vert - len_side_b + 1
            weights[ind_vert][1] = 2 / 6
            connections[ind_vert][2] = ind_vert - 1
            weights[ind_vert][2] = 3 / 6
            connections[ind_vert][3] = ind_vert + 1
            weights[ind_vert][3] = 4 / 6
            connections[ind_vert][4] = ind_vert + len_side_b - 1
            weights[ind_vert][4] = 5 / 6
            connections[ind_vert][5] = ind_vert + len_side_b
            weights[ind_vert][5] = 1.0

    return connections, weights


def create_grid_square_cells(len_side_a, len_side_b):
    """
    todo
    :param len_side_a:
    :param len_side_b:
    :return: two numpy arrays of shape (len_side_a*len_side_b, 6)
    """
    connections = np.full((len_side_a * len_side_b, 4), -1, dtype=np.int32)
    weights = np.full((len_side_a * len_side_b, 4), -1., dtype=np.float64)

    # first 4 corner
    connections[0][0] = 1
    weights[0][0] = 0.5
    connections[0][1] = len_side_b
    weights[0][1] = 1.0

    connections[len_side_b - 1][0] = len_side_b - 2
    weights[len_side_b - 1][0] = 0.5
    connections[len_side_b - 1][1] = 2 * len_side_b - 1
    weights[len_side_b - 1][1] = 1

    connections[len_side_b * (len_side_a - 1)][0] = len_side_b * (len_side_a - 2)
    weights[len_side_b * (len_side_a - 1)][0] = 0.5
    connections[len_side_b * (len_side_a - 1)][1] = len_side_b * (len_side_a - 1) + 1
    weights[len_side_b * (len_side_a - 1)][1] = 1.

    connections[len_side_a * len_side_b - 1][0] = (len_side_a - 1) * len_side_b - 1
    weights[len_side_a * len_side_b - 1][0] = 0.5
    connections[len_side_a * len_side_b - 1][1] = len_side_a * len_side_b - 2
    weights[len_side_a * len_side_b - 1][1] = 1.0

    # take care of the 4 borders
    for i in range(1, len_side_a - 1):
        ind_vert = len_side_b * i
        connections[ind_vert][0] = ind_vert - len_side_b
        weights[ind_vert][0] = 1/3
        connections[ind_vert][1] = ind_vert + 1
        weights[ind_vert][1] = 2/3
        connections[ind_vert][2] = ind_vert + len_side_b
        weights[ind_vert][2] = 1.0

    for i in range(1, len_side_a - 1):
        ind_vert = len_side_b * (i + 1) - 1
        connections[ind_vert][0] = ind_vert - len_side_b
        weights[ind_vert][0] = 1/3
        connections[ind_vert][1] = ind_vert - 1
        weights[ind_vert][1] = 2/3
        connections[ind_vert][2] = ind_vert + len_side_b
        weights[ind_vert][2] = 1.0

    for i in range(1, len_side_b - 1):
        ind_vert = i
        connections[ind_vert][0] = ind_vert - 1
        weights[ind_vert][0] = 1/3
        connections[ind_vert][1] = ind_vert + 1
        weights[ind_vert][1] = 2/3
        connections[ind_vert][2] = ind_vert + len_side_b
        weights[ind_vert][2] = 1.0

    for i in range(1, len_side_b - 1):
        ind_vert = (len_side_a - 1) * len_side_b + i
        connections[ind_vert][0] = ind_vert - len_side_b
        weights[ind_vert][0] = 1/3
        connections[ind_vert][1] = ind_vert - 1
        weights[ind_vert][1] = 2/3
        connections[ind_vert][2] = ind_vert + 1
        weights[ind_vert][2] = 1.0

    # making graph structure of the center of the graph
    for i in range(len_side_a - 2):
        for j in range(len_side_b - 2):
            ind_vert = len_side_b * (i + 1) + (j + 1)
            connections[ind_vert][0] = ind_vert - len_side_b
            weights[ind_vert][0] = 0.25
            connections[ind_vert][1] = ind_vert - 1
            weights[ind_vert][1] = 0.5
            connections[ind_vert][2] = ind_vert + 1
            weights[ind_vert][2] = 0.75
            connections[ind_vert][3] = ind_vert + len_side_b
            weights[ind_vert][3] = 1.0

    return connections, weights


def create_grid_square_with_diagonals(len_side_a, len_side_b):
    connections = np.full((len_side_a * len_side_b, 8), -1, dtype=np.int32)
    weights = np.full((len_side_a * len_side_b, 8), -1., dtype=np.float64)

    diag_weight = 1/sqrt(2)

    # first 4 corner
    x = 2 + diag_weight
    connections[0][0] = 1
    weights[0][0] = 1/x
    connections[0][1] = len_side_b
    weights[0][1] = 2/x
    connections[0][2] = len_side_b + 1
    weights[0][2] = 1.0

    connections[len_side_b - 1][0] = len_side_b - 2
    weights[len_side_b - 1][0] = 1/x
    connections[len_side_b - 1][1] = 2 * len_side_b - 2
    weights[len_side_b - 1][1] = 2/x
    connections[len_side_b - 1][2] = 2 * len_side_b - 1
    weights[len_side_b - 1][2] = 1.

    connections[len_side_b * (len_side_a - 1)][0] = len_side_b * (len_side_a - 2)
    weights[len_side_b * (len_side_a - 1)][0] = 1/x
    connections[len_side_b * (len_side_a - 1)][1] = len_side_b * (len_side_a - 2) + 1
    weights[len_side_b * (len_side_a - 1)][1] = 2 / x
    connections[len_side_b * (len_side_a - 1)][2] = len_side_b * (len_side_a - 1) + 1
    weights[len_side_b * (len_side_a - 1)][2] = 1.

    connections[len_side_a * len_side_b - 1][0] = (len_side_a - 1) * len_side_b - 2
    weights[len_side_a * len_side_b - 1][0] = 1/x
    connections[len_side_a * len_side_b - 1][1] = (len_side_a - 1) * len_side_b - 1
    weights[len_side_a * len_side_b - 1][1] = 2/x
    connections[len_side_a * len_side_b - 1][2] = len_side_a * len_side_b - 2
    weights[len_side_a * len_side_b - 1][2] = 1.0

    # take care of the 4 borders
    x = 3 + 2*diag_weight
    for i in range(1, len_side_a - 1):
        ind_vert = len_side_b * i
        connections[ind_vert][0] = ind_vert - len_side_b
        weights[ind_vert][0] = 1/x
        connections[ind_vert][1] = ind_vert - len_side_b + 1
        weights[ind_vert][1] = (1 + diag_weight)/x
        connections[ind_vert][2] = ind_vert + 1
        weights[ind_vert][2] = (2 + diag_weight)/x
        connections[ind_vert][3] = ind_vert + len_side_b
        weights[ind_vert][3] = (3 + diag_weight)/x
        connections[ind_vert][4] = ind_vert + len_side_b + 1
        weights[ind_vert][4] = 1.0

    for i in range(1, len_side_a - 1):
        ind_vert = len_side_b * (i + 1) - 1
        connections[ind_vert][0] = ind_vert - len_side_b - 1
        weights[ind_vert][0] = diag_weight/x
        connections[ind_vert][1] = ind_vert - len_side_b
        weights[ind_vert][1] = (1 + diag_weight)/x
        connections[ind_vert][2] = ind_vert - 1
        weights[ind_vert][2] = (2 + diag_weight)/x
        connections[ind_vert][3] = ind_vert + len_side_b - 1
        weights[ind_vert][3] = (2 + 2*diag_weight)/x
        connections[ind_vert][4] = ind_vert + len_side_b
        weights[ind_vert][4] = 1.0

    for i in range(1, len_side_b - 1):
        ind_vert = i
        connections[ind_vert][0] = ind_vert - 1
        weights[ind_vert][0] = 1/x
        connections[ind_vert][1] = ind_vert + 1
        weights[ind_vert][1] = 2/x
        connections[ind_vert][2] = ind_vert + len_side_b - 1
        weights[ind_vert][2] = (2 + diag_weight)/x
        connections[ind_vert][3] = ind_vert + len_side_b
        weights[ind_vert][3] = (3 + diag_weight)/x
        connections[ind_vert][4] = ind_vert + len_side_b + 1
        weights[ind_vert][4] = 1.0

    for i in range(1, len_side_b - 1):
        ind_vert = (len_side_a - 1) * len_side_b + i
        connections[ind_vert][0] = ind_vert - len_side_b - 1
        weights[ind_vert][0] = diag_weight/x
        connections[ind_vert][1] = ind_vert - len_side_b
        weights[ind_vert][1] = (1 + diag_weight)/x
        connections[ind_vert][2] = ind_vert - len_side_b + 1
        weights[ind_vert][2] = (1 + 2*diag_weight)/x
        connections[ind_vert][3] = ind_vert - 1
        weights[ind_vert][3] = (2 + 2*diag_weight)/x
        connections[ind_vert][4] = ind_vert + 1
        weights[ind_vert][4] = 1.0

    # making graph structure of the center of the graph
    x = 4*(1 + diag_weight)
    for i in range(len_side_a - 2):
        for j in range(len_side_b - 2):
            ind_vert = len_side_b * (i + 1) + (j + 1)
            connections[ind_vert][0] = ind_vert - len_side_b - 1
            weights[ind_vert][0] = diag_weight/x
            connections[ind_vert][1] = ind_vert - len_side_b
            weights[ind_vert][1] = (1 + diag_weight)/x
            connections[ind_vert][2] = ind_vert - len_side_b + 1
            weights[ind_vert][2] = (1 + 2*diag_weight)/x
            connections[ind_vert][3] = ind_vert - 1
            weights[ind_vert][3] = (2 + 2*diag_weight)/x
            connections[ind_vert][4] = ind_vert + 1
            weights[ind_vert][4] = (3 + 2*diag_weight)/x
            connections[ind_vert][5] = ind_vert + len_side_b - 1
            weights[ind_vert][5] = (3 + 3*diag_weight)/x
            connections[ind_vert][6] = ind_vert + len_side_b
            weights[ind_vert][6] = (4 + 3*diag_weight)/x
            connections[ind_vert][7] = ind_vert + len_side_b + 1
            weights[ind_vert][7] = 1.0

    return connections, weights


class Icosahedron:
    """Class representing an icosahedron. The vertex coordinates as well as the face encoding based on the indices of
    vertices is taken from video games representation of icosahedron as a 3d mesh (for instance in unity3d, see the
    following link: https://wiki.unity3d.com/index.php/CreateIcoSphere ).

    It follows that the order of the vertices in each face has a clockwise orientation, if this is of any interest for
    the user."""
    def __init__(self):
        phi = (1 + sqrt(5)) / 2.
        self.vertices = [[-1.0, phi, 0.0],
                         [1.0, phi, 0.0],
                         [-1.0, -phi, 0.0],
                         [1.0, -phi, 0.0],
                         [0.0, -1.0, phi],
                         [0.0, 1.0, phi],
                         [0.0, -1.0, -phi],
                         [0.0, 1.0, -phi],
                         [phi, 0.0, -1.0],
                         [phi, 0.0, 1.0],
                         [-phi, 0.0, -1.0],
                         [-phi, 0.0, 1.0]
                         ]

        self.faces = \
            [[0, 11, 5],
             [0, 5, 1],
             [0, 1, 7],
             [0, 7, 10],
             [0, 10, 11],
             [1, 5, 9],
             [5, 11, 4],
             [11, 10, 2],
             [10, 7, 6],
             [7, 1, 8],
             [3, 9, 4],
             [3, 4, 2],
             [3, 2, 6],
             [3, 6, 8],
             [3, 8, 9],
             [5, 4, 9],
             [2, 4, 11],
             [6, 2, 10],
             [8, 6, 7],
             [9, 8, 1]
             ]

        self.edges = []
        for face in self.faces:
            if [face[0], face[1]] in self.edges or [face[1], face[0]] in self.edges:
                pass
            else:
                self.edges.append([face[0], face[1]])
            if [face[1], face[2]] in self.edges or [face[2], face[1]] in self.edges:
                pass
            else:
                self.edges.append([face[1], face[2]])
            if [face[0], face[2]] in self.edges or [face[2], face[0]] in self.edges:
                pass
            else:
                self.edges.append([face[0], face[2]])


class SubdividedIcosahedron:
    """
    Class creating a subdivided Icosahedron, that is an icosahedron where each triangular face is divided into
    equilateral triangles. Since the purpose of this class is essentially to be used to construct a "regular" graph on
    a sphere, all the connections between vertices are stored into the array 'connections".

    Finally, note that
    """
    def __init__(self, nb_sub_edges, ignore_warning=False):
        if nb_sub_edges < 2 and not ignore_warning:
            raise ValueError("in the current development stage we cannot guarantee that such a low resolution is " +
                             "reliable. If you so desire, set the kwarg ignore_warning to true and proceed.")
        # the icosahedron
        self.icosahedron = Icosahedron()

        # array used to reduce computation when creating connections
        self.arr_nb_connections = np.zeros((12 + 30 * nb_sub_edges + 10 * (nb_sub_edges - 1) * nb_sub_edges,),
                                           dtype=np.int32)

        # the graph structure
        self.arr_coord = np.zeros((12 + 30*nb_sub_edges + 10*(nb_sub_edges-1)*nb_sub_edges, 3), dtype=np.float32)
        for ii, vertex in enumerate(self.icosahedron.vertices):
            self.arr_coord[ii] = np.array(vertex)
        self.connections = np.full((12 + 30*nb_sub_edges + 10*(nb_sub_edges-1)*nb_sub_edges, 6), -1, dtype=np.int32)
        self.weights = np.full((12 + 30*nb_sub_edges + 10*(nb_sub_edges - 1)*nb_sub_edges, 6), -1, dtype=np.float32)


        # integer telling how much vertices have been created so far
        self.nb_vert_created = 12

        # technical data structure used for filling the faces with points
        self.dict_segments = {}
        for extremities in self.icosahedron.edges:
            self.dict_segments[tuple(extremities)] = self.fill_segment(nb_sub_edges, extremities)

        for face in self.icosahedron.faces:
            self.fill_face(face, nb_sub_edges)

        self.fill_weights()
        # print(self.connections)

    def make_connections(self, ind_v1, ind_v2):
        self.connections[ind_v1][self.arr_nb_connections[ind_v1]] = ind_v2
        self.arr_nb_connections[ind_v1] += 1
        self.connections[ind_v2][self.arr_nb_connections[ind_v2]] = ind_v1
        self.arr_nb_connections[ind_v2] += 1

    def fill_segment(self, nb_sub_edges, extremities):
        r_list = []
        n = nb_sub_edges + 1
        for iii in range(1, n):
            self.arr_coord[self.nb_vert_created] = (1-iii/n)*self.arr_coord[extremities[0]] + \
                                                   (iii/n)*self.arr_coord[extremities[1]]
            if iii == 1:
                self.make_connections(self.nb_vert_created, extremities[0])
            else:
                self.make_connections(self.nb_vert_created, self.nb_vert_created - 1)
                if iii == nb_sub_edges:
                    self.make_connections(self.nb_vert_created, extremities[1])

            r_list.append(self.nb_vert_created)
            self.nb_vert_created += 1
        return r_list

    def fill_face(self, face, nb_sub):
        if (face[0], face[1]) in self.dict_segments:
            seg1 = self.dict_segments[(face[0], face[1])]
        else:
            seg1 = list(reversed(self.dict_segments[(face[1], face[0])]))

        if (face[1], face[2]) in self.dict_segments:
            seg2 = self.dict_segments[(face[1], face[2])]
        else:
            seg2 = list(reversed(self.dict_segments[(face[2], face[1])]))

        if (face[0], face[2]) in self.dict_segments:
            seg3 = self.dict_segments[(face[0], face[2])]
        else:
            seg3 = list(reversed(self.dict_segments[(face[2], face[0])]))

        self.make_connections(seg1[0], seg3[0])
        self.make_connections(seg1[-1], seg2[0])
        self.make_connections(seg2[-1], seg3[-1])

        for i in range(1, nb_sub):
            for j in range(1, i + 1):
                self.arr_coord[self.nb_vert_created] = (1 - j/(i+1)) * self.arr_coord[seg1[i]] + \
                                                       (j / (i + 1)) * self.arr_coord[seg3[i]]
                # add the stuff to manage the connections here
                # deal with connection below
                if i == nb_sub - 1:
                    self.make_connections(self.nb_vert_created, seg2[j])
                    self.make_connections(self.nb_vert_created, seg2[j-1])
                else:
                    self.make_connections(self.nb_vert_created, self.nb_vert_created + i)
                    self.make_connections(self.nb_vert_created, self.nb_vert_created + i + 1)

                # deal with with side connections
                if j == 1:
                    self.make_connections(self.nb_vert_created, seg1[i-1])
                    self.make_connections(self.nb_vert_created, seg1[i])
                if j == i:
                    self.make_connections(self.nb_vert_created, seg3[i-1])
                    self.make_connections(self.nb_vert_created, seg3[i])
                if j > 1:
                    self.make_connections(self.nb_vert_created, self.nb_vert_created - 1)

                # increment the counter
                self.nb_vert_created += 1

    def fill_weights(self):
        nb_penta = 0
        for i, conn in enumerate(self.connections):
            nb_conn = (conn != -1).sum()
            if nb_conn == 5:
                nb_penta += 1
                for j in range(5):
                    self.weights[i][j] = (j+1)/5.
            elif nb_conn == 6:
                for j in range(6):
                    self.weights[i][j] = (j+1)/6.
            else:
                raise SystemError('something went horribly wrong with isosphere generation and a vertex has ' +
                                  str(nb_conn) + ' neighbours, which is impossible.')
        if nb_penta != 12:
            raise SystemError('something went horribly wrong with iso-sphere generation and there are ' +
                              str(nb_penta) + ' pentagonal vertices instead of 12.')


def dist_two_points_on_sphere(lat1, lon1, lat2, lon2):
    """
    Compute the distance between two points given their coord in lat-lon, using Haversine formula
    ( https://en.wikipedia.org/wiki/Haversine_formula ).
    :param lat1:
    :param lon1:
    :param lat2:
    :param lon2:
    :return: float, distance in kilometer
    """
    pass