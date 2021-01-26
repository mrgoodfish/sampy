import numpy as np
from .jit_compiled_functions import (transition_initialize_counters_of_newly_infected,
                                     transition_conditional_count_nb_agent_per_vertex,
                                     transition_falsify_when_condition
                                     )


class TransitionWithUniformProb:
    def __init__(self):
        super().__init__()

    def uniform_transition(self, proba_transition, initial_state, target_state):
        """
        Randomly perform the transition from an initial_state to a target_state. Very simply performed comparing the
        result of a uniform sample against the probability of transition for all individuals.
        :param proba_transition: float, probability of transition
        :param initial_state: string, in ['inf', 'con', 'imm']
        :param target_state: string, in ['inf', 'con', 'imm', 'death']
        """
        rand = np.random.uniform(0, 1, (self.host.df_population.shape[0],)) < proba_transition
        if target_state == 'death':
            self.host.df_population = self.host.df_population[~(rand & self.host.df_population[initial_state + '_' + self.disease_name])].copy()
        else:
            self.host.df_population[target_state + '_' + self.disease_name] = (rand & self.host.df_population[initial_state + '_' + self.disease_name]) | self.host.df_population[target_state + '_' + self.disease_name]
            if target_state == 'imm':
                bool_inf = np.array(self.host.df_population['inf_' + self.disease_name], dtype=bool)
                self.host.df_population['inf_' + self.disease_name] = falsify_when_condition(bool_inf, rand)
                bool_con = np.array(self.host.df_population['con_' + self.disease_name], dtype=bool)
                self.host.df_population['con_' + self.disease_name] = falsify_when_condition(bool_con, rand)


class TransitionDeterministicCounterWithPermanentImmunity:
    def __init__(self):
        self.host.df_population['cnt_inf_' + self.disease_name] = 0
        self.host.df_population['cnt_con_' + self.disease_name] = 0
        super().__init__(self)

    def increment_counters(self):
        """
        todo
        """
        self.host.df_population['cnt_inf_' + self.disease_name] += self.host.df_population['inf_' + self.disease_name]
        self.host.df_population['cnt_con_' + self.disease_name] += self.host.df_population['con_' + self.disease_name]

    def deterministic_transition_with_counter(self, limit, initial_state, target_state):
        """
        Deterministically perform the transition from an initial_state to a target_state.
        :param limit: integer, nb of week after which the transition should happen
        :param initial_state: string, in ['inf', 'con', 'imm']
        :param target_state: string, in ['inf', 'con', 'imm', 'death']
        """
        susceptible = (self.host.df_population['cnt_' + initial_state + '_' + self.disease_name] >= limit) & \
                      (self.host.df_population[initial_state + '_' + self.disease_name])
        if target_state == 'death':
            self.host.df_population = self.host.df_population[~susceptible].copy()
        else:
            self.host.df_population[target_state + '_' + self.disease_name] = susceptible | self.host.df_population[
                                                                                target_state + '_' + self.disease_name]
            if target_state == 'imm':
                bool_inf = np.array(self.host.df_population['inf_' + self.disease_name], dtype=bool)
                cnt_inf = np.array(self.host.df_population['cnt_inf_' + self.disease_name], dtype=np.int32)
                self.host.df_population['inf_' + self.disease_name] = falsify_when_condition(bool_inf, susceptible)
                self.host.df_population['cnt_inf_' + self.disease_name] = zero_when_condition(cnt_inf, susceptible)

                bool_con = np.array(self.host.df_population['con_' + self.disease_name], dtype=bool)
                cnt_con = np.array(self.host.df_population['cnt_con_' + self.disease_name], dtype=np.int32)
                self.host.df_population['con_' + self.disease_name] = falsify_when_condition(bool_con, susceptible)
                self.host.df_population['cnt_con_' + self.disease_name] = zero_when_condition(cnt_con, susceptible)


class TransitionPoissonLawWithPermanentImmunity:
    def __init__(self):
        self.host.df_population['cnt_inf_' + self.disease_name] = 0
        self.host.df_population['cnt_con_' + self.disease_name] = 0
        super().__init__(self)

    def _initialize_on_infection_poisson_law(self, arr_new_infected, lambda_poisson):
        """
        todo
        :param arr_new_infected: array of boolean saying which individuals are newly infected (and only those !)
        :param lambda_poisson:
        :return:
        """
        nb_new_infected = arr_new_infected.sum()
        arr_cnt = np.array(self.host.df_population['cnt_inf_' + self.disease_name], dtype=np.int32)
        poisson_sample = np.random.poisson(lambda_poisson, nb_new_infected)
        self.host.df_population['cnt_inf_' + self.disease_name] = fill_poisson_counter(arr_cnt,
                                                                                       arr_new_infected,
                                                                                       poisson_sample)

    def decrement_counter(self):
        """
        todo
        """
        self.host.df_population['cnt_inf_' + self.disease_name] -= self.host.df_population['inf_' + self.disease_name]
        self.host.df_population['cnt_con_' + self.disease_name] -= self.host.df_population['con_' + self.disease_name]

    def transition_with_decreasing_counter(self, initial_state, target_state, proba_death=None, lambda_next_state=None):
        """
        Deterministically perform the transition from an initial_state to a target_state.
        :param initial_state: string, in ['inf', 'con', 'imm']
        :param target_state: string, in ['con', 'imm', 'death']
        :param proba_death: float, optional. Give probability of death if target_state=death. Default value 1.0
        :param arr_nb_timestep: todo
        :param arr_prob_nb_timestep: todo
        """
        # bool array of all individuals that will make transition
        susceptible = np.array(self.host.df_population['cnt_' + initial_state + '_' + self.disease_name] == 0,
                               dtype=bool)
        susceptible = susceptible & np.array(self.host.df_population[initial_state + '_' + self.disease_name],
                                             dtype=bool)
        # in case of death
        if target_state == 'death':
            # there might be a probability of dying of the disease, which is taken into account now
            if proba_death is not None:
                susceptible = susceptible & \
                              (np.random.uniform(0, 1, (self.host.df_population.shape[0],)) < proba_death)
            # killing
            self.host.df_population = self.host.df_population[~susceptible].copy()

        # all the rest corresponds to transition between stages of the disease
        else:
            self.host.df_population[target_state + '_' + self.disease_name] = susceptible | self.host.df_population[
                                                                                target_state + '_' + self.disease_name]
            if target_state == 'imm':
                bool_inf = np.array(self.host.df_population['inf_' + self.disease_name], dtype=bool)
                self.host.df_population['inf_' + self.disease_name] = falsify_when_condition(bool_inf, susceptible)

                bool_con = np.array(self.host.df_population['con_' + self.disease_name], dtype=bool)
                self.host.df_population['con_' + self.disease_name] = falsify_when_condition(bool_con, susceptible)

            else:
                if lambda_next_state is None:
                    raise ValueError("No value given for parameter of poison law used in disease transition")

                bool_ini_state = np.array(self.host.df_population[initial_state + '_' + self.disease_name], dtype=bool)
                self.host.df_population[initial_state + '_' + self.disease_name] = \
                    falsify_when_condition(bool_ini_state, susceptible)
                nb_new_infected = susceptible.sum()
                arr_cnt = np.array(self.host.df_population['cnt_' + target_state + '_' + self.disease_name], dtype=np.int32)
                poisson_sample = np.random.poisson(lambda_next_state, nb_new_infected)
                self.host.df_population['cnt_' + target_state + '_' + self.disease_name] = \
                    fill_poisson_counter(arr_cnt, susceptible, poisson_sample)


class TransitionCustomProbPermanentImmunity:
    """
    This class introduce transitions between disease states based on probabilities given by the user.

    WARNING: be aware that the time each agent spend in each status of the disease is kept in memory using counters.
             Those counters HAVE TO be initialized for newly infected individuals throught the use of the method
             'initialize_counters_of_newly_infected'. This is a current problem of Sampy that the user has to
             explicitly call a method that should be automatically called in the background. This issue will be
             adressed in the future once a satisfactory design solution has been found (here, satisfactory means
             'that doesn't create too much special cases that developers working on Sampy have to keep in mind').
    """
    def __init__(self, **kwargs):
        self.host.df_population['cnt_inf_' + self.disease_name] = 0
        self.host.dict_default_val['cnt_inf_' + self.disease_name] = 0

        self.host.df_population['cnt_con_' + self.disease_name] = 0
        self.host.dict_default_val['cnt_con_' + self.disease_name] = 0

    def initialize_counters_of_newly_infected(self, arr_new_infected, arr_nb_timestep, arr_prob):
        """
        Method that HAS TO be called each time new individuals get infected

        :param arr_new_infected: 1d array of bool, saying which agent are newly infected and should have their
                                 'infectious status counter' initialized.
        :param arr_nb_timestep: 1d array of int.
        :param arr_prob: 1d array of non negative floats, will be normalized to 1.

        """
        prob = arr_prob.astype('float64')
        prob = prob/prob.sum()
        arr_cnt = np.random.choice(arr_nb_timestep, arr_new_infected.sum(), p=prob)

        transition_initialize_counters_of_newly_infected(arr_new_infected,
                                                         self.host.df_population['cnt_inf_' + self.disease_name],
                                                         arr_cnt)

    def decrement_counter(self):
        """
        Reduce by one the counters of the agents in a given disease state. Note that this method will only decrease
        positive counters, so that if a negative counter was to appear, which shouldn't, this should be caused by
        something else.
        """
        self.host.df_population['cnt_inf_' + self.disease_name] -= self.host.df_population['inf_' + self.disease_name] & \
                                                        (self.host.df_population['cnt_inf_' + self.disease_name] > 0)
        self.host.df_population['cnt_con_' + self.disease_name] -= self.host.df_population['con_' + self.disease_name] & \
                                                        (self.host.df_population['cnt_con_' + self.disease_name] > 0)

    def transition_between_states(self, initial_state, target_state, proba_death=1.,
                                  arr_nb_timestep=None, arr_prob_nb_timestep=None, return_transition_count=False,
                                  position_attribute='position'):
        """
        Performs the transition from an initial_state to a target_state of the disease, where 'death' is a possible
        target_state. When performing the transition, each agent for which 'initial_state' is True and the associated
        counter is 0 makes the transition (except if the target state is death proba_death is smaller than 1.).

        WARNING: note that an agent can only be in a SINGLE state. That is, an agent cannot be simultaneously 'infected'
                 and 'contagious'. So if the user wants, for instance, to count all agents carrying the disease, then
                 both 'inf' and 'con' states have to be considered.

        :param initial_state: string, in ['inf', 'con', 'imm']
        :param target_state: string, in ['con', 'imm', 'death']
        :param proba_death: optional, float, default 1.0. Probability of death if target_state=death.
        :param arr_nb_timestep: optional, 1d array of int, default None.
        :param arr_prob_nb_timestep: optional, 1d array of float, default None.
        :param return_transition_count: optional, bool, default False. If True, returns a 1D array of integer counting
                                        how many agents did the transition per cell.
        :param position_attribute: optional, string, default 'position'. Name of the position attribute used for counting
                                   if 'return_transition_count' is set to True.

        :returns: if return_transition_count is True, returns a 1d array of int. Else, returns None.
        """

        # bool array of all individuals that will make transition
        susceptible = self.host.df_population['cnt_' + initial_state + '_' + self.disease_name] == 0
        susceptible = susceptible & self.host.df_population[initial_state + '_' + self.disease_name]

        count_arr = None

        # in case of death
        if target_state == 'death':
            # there might be a probability of dying of the disease, which is taken into account now
            if proba_death < 1.:
                susceptible = susceptible & \
                              (np.random.uniform(0, 1, (self.host.df_population.shape[0],)) < proba_death)

            if return_transition_count:
                count_arr = transition_conditional_count_nb_agent_per_vertex(susceptible,
                                                                             self.host.df_population[
                                                                                 position_attribute],
                                                                             self.host.graph.weights.shape[0])

            # killing
            self.host.df_population = self.host.df_population[~susceptible]

        # all the rest corresponds to transition between stages of the disease
        else:
            if return_transition_count:
                count_arr = transition_conditional_count_nb_agent_per_vertex(susceptible,
                                                                             self.host.df_population[
                                                                                 position_attribute],
                                                                             self.host.graph.weights.shape[0])

            self.host.df_population[target_state + '_' + self.disease_name] = susceptible | self.host.df_population[
                                                                                target_state + '_' + self.disease_name]
            if target_state == 'imm':
                transition_falsify_when_condition(self.host.df_population['inf_' + self.disease_name], susceptible)
                transition_falsify_when_condition(self.host.df_population['con_' + self.disease_name], susceptible)

            else:
                transition_falsify_when_condition(self.host.df_population[initial_state + '_' + self.disease_name],
                                                  susceptible)

                prob = arr_prob_nb_timestep.astype('float64')
                prob = prob / prob.sum()
                arr_cnt = np.random.choice(arr_nb_timestep, susceptible.sum(), p=prob)

                transition_initialize_counters_of_newly_infected(susceptible,
                                                                 self.host.df_population[
                                                                     'cnt_' + target_state + self.disease_name],
                                                                 arr_cnt)
        return count_arr
