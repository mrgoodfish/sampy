import pandas as pd
import numpy as np

import xmltodict

from .jit_compiled_functions import compute_sin_attr_with_condition
from ..pandas_xs.pandas_xs import DataFrameXS


class BaseVertexAttributes:
    def __init__(self, **kwargs):
        self.df_attributes = DataFrameXS()

    def create_vertex_attribute(self, attr_name, value):
        """
        Creates a new vertex attribute and populates its values. Accepted input for 'value' are:
                - None: in this case, the attribute column is set empty
                - A single value, in which case all vertexes will have the same attribute value
                - A 1D array, which will become the attribute column.

        Note that if you use a 1D array, then you are implicitly working with the indexes of the vertices, that is that
        the value at position 'i' in the array corresponds to the attribute value associated with the vertex whose index
        is 'i'. If you want to work with vertexes id instead, use the method 'create_vertex_attribute_from_dict'.

        :param attr_name: string, name of the attribute
        :param value: either None, a single value, or a 1D array.
        """
        self.df_attributes[attr_name] = value

    def create_vertex_attribute_from_dict(self, attr_name, dict_id_to_val, default_val=np.nan):
        """
        Creates a new vertex attribute and populates its values using a dictionary-like object, whose keys are id of
        vertices, and values the corresponding attribute values.

        :param attr_name: string, name of the attribute.
        :param dict_id_to_val: Dictionary like object, whose keys are id of vertices, and values the corresponding
                               attribute value.
        :param default_val: optional, default np.nan. Value used for the vertexes for which an attribute value is not
                            provided.
        """
        arr_attr = np.full((self.nb_vertex,), default_val)
        for key, val in dict_id_to_val.items():
            arr_attr[self.dict_cell_id_to_ind[key]] = val
        self.df_attributes[attr_name] = arr_attr

    def change_type_attribute(self, attr_name, str_type):
        """
        Change the dtype of the selected attribute. Note that the type should be supported by DataFrameXS
        :param attr_name: string, name of the attribute
        :param str_type: string, target dtype of the attribute
        """
        self.df_attributes.change_type(attr_name, str_type)


class PeriodicAttributes(BaseVertexAttributes):
    """
    Class that adds a method to define periodically varying arguments.
    """
    def __init__(self, **kwargs):
        pass

    def update_periodic_attribute(self, attr_name, amplitude, period, phase, intercept, time=None, condition=None):
        """
        Call this method to update the value of an attribute using the following formula.

            amplitude * np.sin(2 * math.pi * time / period + phase) + intercept

        Where time is either the value of the attribute 'time' of the graph, or if 'time' parameter is not None

        :param attr_name: string, name of the attribute
        :param amplitude: float, see formula above
        :param phase: float, see formula above
        :param period: float, see formula above
        :param intercept: float, see formula above
        :param time: optional, default None. If not None, then used as time parameter in the update formula.
        :param condition: optional, default None. Boolean Array saying for which cell to apply the sinusoidal variation.
            If None, this method behave like an array of True has been provided.
        """
        arr_attr = self.df_attributes[attr_name]
        if condition is None:
            condition = np.full(arr_attr.shape, True, dtype=np.bool_)
        if time is None:
            time = self.time
        compute_sin_attr_with_condition(arr_attr, condition, time, amplitude,
                                        period, phase, intercept)


# class VertexAttributes:
#     """
#     Class for management of vertices attributes in graphs
#     """
#
#     def populate_vertex_attribute_with_orm_xml(self, attr_name, attr_dtype, path_to_xml,
#                                                dict_cell_id_to_cell_index=None):
#         """
#         todo
#         :param attr_name:
#         :param attr_dtype:
#         :param path_to_xml:
#         :param dict_cell_id_to_cell_index:
#         """
#         if attr_name not in self.dict_tag_in_file:
#             raise ValueError('attribute should be created before being populated.')
#         if not dict_cell_id_to_cell_index:
#             if not hasattr(self, 'dict_cell_id_to_ind'):
#                 raise ValueError('no available dictionnary to translate cells id into their indexes')
#             else:
#                 dict_cell_id_to_cell_index = self.dict_cell_id_to_ind
#
#         with open(path_to_xml, 'rb') as xml_file:
#             dic_graph = xmltodict.parse(xml_file)
#
#         list_cells = dic_graph['NewDataSet']['AllCellData']
#         nb_cells = len([cell for cell in list_cells if cell['HEXID'] in dict_cell_id_to_cell_index])
#         if attr_dtype == str:
#             self.df_attributes['has_' + attr_name] = False
#             self.df_attributes[attr_name] = 'Null'
#             for cell in list_cells:
#                 id_cell = cell['HEXID']
#                 if id_cell in dict_cell_id_to_cell_index:
#                     ind_cell = dict_cell_id_to_cell_index[id_cell]
#                     if self.dict_tag_in_file[attr_name] in cell:
#                         attr_val = cell[self.dict_tag_in_file[attr_name]]
#                         self.df_attributes.at[ind_cell, attr_name] = attr_val
#                         self.df_attributes.at[ind_cell, 'has_' + attr_name] = True
#         else:
#             arr_attr_value = np.zeros((nb_cells,), dtype=attr_dtype)
#             arr_col_control = np.full((nb_cells,), False, dtype=bool)
#             for cell in list_cells:
#                 id_cell = cell['HEXID']
#                 if id_cell in dict_cell_id_to_cell_index:
#                     ind_cell = dict_cell_id_to_cell_index[id_cell]
#                     if self.dict_tag_in_file[attr_name] in cell:
#                         attr_val = cell[self.dict_tag_in_file[attr_name]]
#                         arr_attr_value[ind_cell] = attr_val
#                         arr_col_control[ind_cell] = True
#
#             self.df_attributes['has_' + attr_name] = arr_col_control
#             self.df_attributes[attr_name] = arr_attr_value
