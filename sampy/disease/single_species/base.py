import numpy as np
from .jit_compiled_functions import base_conditional_count_nb_agent_per_vertex
from ...utils.errors_shortcut import check_col_exists_good_type


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

        if not hasattr(self, 'list_disease_status'):
            self.set_disease_status = {'inf', 'con', 'imm'}
        else:
            self.set_disease_status.update(['inf', 'con', 'imm'])

    def _sampy_debug_count_nb_status_per_vertex(self, target_status, attribute_position='position'):
        if self.host.df_population.nb_rows == 0:
            return
        check_col_exists_good_type(self.host.df_population, attribute_position, 'attribute_position',
                                   prefix_dtype='int', reject_none=True)
        check_col_exists_good_type(self.host.df_population, target_status + '_' + self.disease_name,
                                   'target_status', prefix_dtype='bool', reject_none=True)

    def count_nb_status_per_vertex(self, target_status, position_attribute='position'):
        """
        Count the number of agent having the targeted status in each vertex. The status can either be 'inf', 'con' and
        'imm', which respectively corresponds to infected, contagious and immuned agents.

        :param target_status: string in ['inf', 'con', 'imm']
        :param position_attribute: optional, string.

        :return: array counting the number of agent having the target status in each vertex
        """
        if self.host.df_population.nb_rows == 0:
            return np.full((self.host.graph.number_vertices,), 0, dtype=np.int32)
        return base_conditional_count_nb_agent_per_vertex(self.host.df_population[target_status + '_' +
                                                                                  self.disease_name],
                                                          self.host.df_population[position_attribute],
                                                          self.host.graph.weights.shape[0])
