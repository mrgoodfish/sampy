import numpy as np
from .jit_compiled_functions import (movement_change_territory_and_position,
                                     movement_change_territory_and_position_condition,
                                     movement_mov_around_territory_fill_bool_mov_using_condition,
                                     movement_mov_around_territory)
from ..pandas_xs.pandas_xs import DataFrameXS


class TerritorialMovementWithoutResistance:
    """

    """
    def __init__(self, **kwargs):
        if not hasattr(self, 'df_population'):
            self.df_population = DataFrameXS()

        self.df_population['territory'] = None
        self.df_population['position'] = None

    def change_territory(self,
                         condition=None,
                         territory_attribute='territory',
                         position_attribute='position'):
        """
        Change the territory and the position of the agents. If an agent is on an isolated vertex (a vertex without
        any neighbour), then the agent stays on the vertex.

        :param condition: optional, array of bool, default None. If not None, array telling which
        :param territory_attribute: optional, string, default 'territory'
        :param position_attribute: optional, string, default 'position'
        """
        if condition is not None:
            rand = np.random.uniform(0, 1, (condition.sum(),))
            movement_change_territory_and_position_condition(self.df_population[territory_attribute],
                                                             self.df_population[position_attribute],
                                                             condition, rand,
                                                             self.graph.connections, self.graph.weights)
        else:
            rand = np.random.uniform(0, 1, (self.df_population.shape[0],))
            movement_change_territory_and_position(self.df_population[territory_attribute],
                                                   self.df_population[position_attribute],
                                                   rand, self.graph.connections, self.graph.weights)

    def mov_around_territory(self,
                             proba_remain_on_territory,
                             condition=None,
                             territory_attribute='territory',
                             position_attribute='position'):
        """
        Update the average position of the agent around its territory during the current time step.

        :param proba_remain_on_territory: float, probability to stay on the territory
        :param condition: optional, array of bool, default None. Array of boolean such that the i-th value is
                          True if and only if the i-th agent (i.e. the agent at the line i of df_population) can
                          move.
        :param territory_attribute: optional, string, default 'territory'
        :param position_attribute: optional, string, default 'position'
        """
        if condition is not None:
            pre_bool_mov = np.random.uniform(0, 1, condition.sum()) > proba_remain_on_territory
            bool_mov = movement_mov_around_territory_fill_bool_mov_using_condition(pre_bool_mov, condition)
        else:
            bool_mov = np.random.uniform(0, 1, self.df_population.shape[0]) > proba_remain_on_territory
        rand = np.random.uniform(0, 1, bool_mov.sum())
        movement_mov_around_territory(self.df_population[territory_attribute], self.df_population[position_attribute],
                                      bool_mov, rand, self.graph.connections, self.graph.weights)

