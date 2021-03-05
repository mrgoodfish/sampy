from sampy.agent.base import BaseAgingAgent
from sampy.agent.mortality import NaturalMortalityOrmMethodology, OffspringDependantOnParents
from sampy.agent.reproduction import BasicReproductionTerritorialAgent
from sampy.agent.movement import TerritorialMovementWithoutResistance
from sampy.utils.decorators import sampy_class

from .jit_compiled_function import (orm_like_agent_orm_dispersion,
                                    orm_like_agents_mortality_from_v08_no_condition_no_alpha_beta)

import numpy as np


@sampy_class
class ORMLikeAgent(BaseAgingAgent,
                   NaturalMortalityOrmMethodology,
                   OffspringDependantOnParents,
                   BasicReproductionTerritorialAgent,
                   TerritorialMovementWithoutResistance):
    """
    Class developed for the need of the Leighton Lab. Add orm-like methods to the agents.

    IMPORTANT: for some methods, the underlying graph is assumed to come from an ORM xml.
    """
    def __init__(self, **kwargs):
        self.df_population['has_moved'] = False
        self.dict_default_val['has_moved'] = False

    def orm_dispersion(self, timestep, permissible_weeks, condition, arr_nb_steps, arr_prob_nb_steps,
                       position_attribute='position', territory_attribute='territory'):
        """

        :param timestep:
        :param permissible_weeks:
        :param condition:
        :param arr_nb_steps:
        :param arr_prob_nb_steps:
        :param position_attribute:
        :param territory_attribute:
        """
        if self.df_population.nb_rows == 0:
            return

        # we reinitialize the 'has_moved' status if first week of the year
        if timestep % 52 == 0:
            self.df_population['has_moved'] = False

        if timestep % 52 not in permissible_weeks:
            return

        can_move = condition & ~self.df_population['has_moved']
        will_move = np.random.uniform(0, 1, can_move.sum()) < \
                    (permissible_weeks.index(timestep % 52) + 1) / len(permissible_weeks)
        prob = arr_prob_nb_steps.astype('float64')
        prob = prob / prob.sum()
        rand_nb_steps = np.random.choice(arr_nb_steps, will_move.sum(), p=prob)
        rand_directions = np.random.uniform(0, 1, rand_nb_steps.sum())

        orm_like_agent_orm_dispersion(can_move, will_move, rand_nb_steps, rand_directions,
                                      self.df_population[position_attribute], self.df_population[territory_attribute],
                                      self.df_population['has_moved'], self.graph.connections)

    def mortality_from_v08(self, arr_annual_mortality, condition_count, alpha_beta=None, condition=None,
                           shuffle=True, age_attribute='age', position_attribute='position'):
        """
        This is an adaptation of the mortality method found in ARM v08 (file cFox.cs).

        WARNING: some elements of this method are odd, and there are no sources given in the ORM file. For instance, in
                 cFox.cs, a parameter computed using female mortality is used for males as well. Therefore, we recommand
                 the user to be cautious when using this method.

        :param arr_annual_mortality:
        :param condition_count:
        :param alpha_beta:
        :param condition:
        :param shuffle:
        :param age_attribute:
        :param position_attribute:
        """
        if self.df_population.nb_rows == 0:
            return

        if shuffle:
            permutation = self.df_population.scramble(return_permutation=True)
            if condition is not None:
                condition = condition[permutation]
            condition_count = condition_count[permutation]

        count_arr = self.count_pop_per_vertex(position_attribute=position_attribute, condition=condition_count)
        if alpha_beta is None:
            if condition is None:
                rand = np.random.uniform(0, 1, self.df_population.nb_rows)
                survive = orm_like_agents_mortality_from_v08_no_condition_no_alpha_beta(count_arr, condition_count,
                                                                                        self.df_population[position_attribute],
                                                                                        self.graph.df_attributes['K'],
                                                                                        arr_annual_mortality,
                                                                                        self.df_population[age_attribute],
                                                                                        rand)
        self.df_population = self.df_population[survive]
