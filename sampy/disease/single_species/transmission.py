import numpy as np
from .jit_compiled_functions import conditional_count_return_full_array


class TransmissionByContact:
    def __init__(self):
        super().__init__()

    def contact_contagion(self, contact_rate, position_attribute='position',
                          condition=None,
                          return_arr_new_infected=True):
        """
        Propagate the disease by direct contact using the following methodology. For any vertex X of the graph, we
        count the number of contagious agents N_c, then each non immuned agent on the cell X has a probability of
        contact_rate * N_c to become infected.

        :param contact_rate: Float value. used to determine the probability of becoming infected
        :param position_attribute: optional, string, default 'position'. Name of the agent attribute used as position.
        :param condition: optional, array of bool, default None. Array of boolean such that the i-th value is
                          True if and only if the i-th agent (i.e. the agent at the line i of df_population) can be
                          infected. All the agent having their corresponding value to False are protected from
                          infection.
        :param return_arr_new_infected: optional, bool, default True
        """
        col_pos = self.host.df_population[position_attribute], dtype=np.int32
        col_con = self.host.df_population['con_' + self.disease_name], dtype=bool
        nb_vertex = self.host.graph.connections.shape[0]

        # return the array counting the number of contagious agents
        count_con = conditional_count_return_full_array(nb_vertex, col_pos, col_con)

        # make the array of newly infected individuals. Note that for the moment the computation is not optimized,
        # and the random contamination is computed for EVERY agent, and then we exclude the not susceptible ones.
        new_infected = np.random.uniform(0, 1, (count_con.shape[0],)) < 1 - (1 - contact_rate) ** count_con

        # exclusion of the immuned and already infected agents
        new_infected = new_infected & ~(self.host.df_population['inf_' + self.disease_name]) \
                                    & ~(self.host.df_population['con_' + self.disease_name]) \
                                    & ~(self.host.df_population['imm_' + self.disease_name])

        if condition is not None:
            new_infected = new_infected & condition

        self.host.df_population['inf_' + self.disease_name] = self.host.df_population['inf_' + self.disease_name] | new_infected

        if return_arr_new_infected:
            return new_infected
