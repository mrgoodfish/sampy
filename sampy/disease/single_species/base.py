import numpy as np
from .jit_compiled_functions import base_conditional_count_nb_agent_per_vertex


class BaseSingleSpeciesDisease:
    def __init__(self, disease_name=None, host=None, **kwargs):
        # check values have been given
        if host is None:
            raise ValueError("No host given for the disease. Use the kwarg 'host'.")
        if disease_name is None:
            raise ValueError("No name given to the disease. Use the kwarg 'disease_name'.")

        self.host = host
        self.disease_name = disease_name

        self.host.df_population['inf_' + disease_name] = False
        self.host.df_population['con_' + disease_name] = False
        self.host.df_population['imm_' + disease_name] = False

        self.host.dict_default_val['inf_' + disease_name] = False
        self.host.dict_default_val['con_' + disease_name] = False
        self.host.dict_default_val['imm_' + disease_name] = False

    def count_nb_status_per_vertex(self, target_status, attribute_position='position'):
        """
        Count the number of agent having the targeted status in each vertex. The status can either be 'inf', 'con' and
        'imm', which respectively corresponds to infected, contagious and immuned agents.

        :param target_status: string in ['inf', 'con', 'imm']
        :param attribute_position: optional, string.

        :return: array counting the number of agent having the target status in each vertex
        """
        return base_conditional_count_nb_agent_per_vertex(self.host.df_population[target_status + '_' +
                                                                                  self.disease_name],
                                                          self.host.df_population[attribute_position],
                                                          self.host.graph.weights.shape[0])
