import numpy as np
import pandas as pd
from sampy.utils.jit_compiled_functions import reprod_part1, reprod_part2
from .jit_compiled_functions import (reproduction_find_random_mate_on_position,
                                     reproduction_find_random_mate_on_position_condition)
from .jit_compiled_functions import repro_marker_find_mate_part_2
from ..pandas_xs.pandas_xs import DataFrameXS


class BasicReproduction:
    """
    This class provides methods to modelize reproduction of agents. Variety of methods are provided.
    """
    def __init__(self, **kwargs):
        if not hasattr(self, 'df_population'):
            self.df_population = DataFrameXS()
        self.df_population['mom_id'] = None
        self.df_population['dad_id'] = None
        self.df_population['gender'] = None
        self.df_population['is_pregnant'] = None
        self.df_population['current_mate'] = None

    def find_random_mate_on_position(self,
                                     prob_get_pregnant,
                                     shuffle=True,
                                     permutation=None,
                                     condition=None,
                                     id_attribute='col_id',
                                     position_attribute='position',
                                     gender_attribute='gender',
                                     mate_attribute='current_mate',
                                     pregnancy_attribute='is_pregnant'):
        """
        Find a mate on the current position of the agent. This mate is randomly picked. By default, the attribute used
        as the position if 'position', but the user may want to use 'territory' instead. For that purpose, the key-word
        argument 'position_attribute' can be used.

        :param prob_get_pregnant: float between 0 and 1. Probability that after mating the female will get pregnant.
        :param shuffle: optional, boolean, default True. By default, in this method the random choice of a mate is done
                        by shuffling the DataFrameXS 'df_population'. If set to False, the df is not shuffled, so that
                        the first male in a cell is paired with the first female in the cell (as they appear in df),
                        the second male with the second female, and so on until there is no male anymore (or no female).
        :param permutation: optional, default None, 1D array of integer. If not None and shuffle is True, this
                            permutation is used to shuffle df_population.
        :param condition: optional, array of bool, default None. Tells which agents should be included.
        :param id_attribute: optional, string, default 'col_id'. Id attribute of the agent. It is not recommended to
                             change this column, as this column is considered internal, and in the future this fact
                             could be used in other methods.
        :param position_attribute: optional, string, default 'position'. Position attribute of the agents. Should be
                                   integers corresponding to indexes of the vertices of the graph on which the agents
                                   live.
        :param gender_attribute: optional, string, default 'gender'.
        :param mate_attribute: optional, string, default 'current_mate'.
        :param pregnancy_attribute: optional, string, default 'is_pregnant'.
        """
        if shuffle:
            self.df_population.scramble(permutation=permutation)

        if condition is None:
            rand = np.random.uniform(0, 1, ((self.df_population[gender_attribute] == 1).sum(),))
            col_mate, col_pregnancy = reproduction_find_random_mate_on_position(self.df_population[id_attribute],
                                                                                self.df_population[position_attribute],
                                                                                self.df_population[gender_attribute],
                                                                                self.graph.connections.shape[0],
                                                                                rand,
                                                                                prob_get_pregnant
                                                                                )
        else:
            rand = np.random.uniform(0, 1, (((self.df_population[gender_attribute] == 1) & condition).sum(),))
            col_mate, col_pregnancy = reproduction_find_random_mate_on_position_condition(
                                                                                self.df_population[id_attribute],
                                                                                self.df_population[position_attribute],
                                                                                self.df_population[gender_attribute],
                                                                                self.graph.connections.shape[0],
                                                                                rand,
                                                                                prob_get_pregnant,
                                                                                condition
                                                                                )

        self.df_population[mate_attribute] = col_mate
        self.df_population[pregnancy_attribute] = col_pregnancy

    def create_offsprings_custom_prob(self,
                                      arr_nb_children,
                                      arr_prob_nb_children,
                                      condition=None,
                                      dico_default_values={},
                                      prob_failure=None,
                                      age_attribute='age',
                                      mother_attribute='mom_id',
                                      father_attribute='dad_id',
                                      gender_attribute='gender',
                                      id_attribute='col_id',
                                      position_attribute='position',
                                      territory_attribute='territory',
                                      mate_attribute='current_mate',
                                      pregnancy_attribute='is_pregnant'):
        """
        Creates offsprings using two 1D arrays of same size, 'arr_nb_children' and 'arr_prob_nb_children', being
        respectively an array of integers and an array of non-negative floats, where for any index i,
        arr_prob_nb_children[i] is the proportion of pregnant females that will give birth to arr_nb_children[i]
        offsprings.

        Note that arr_prob_nb_children is normalized so that it sums to 1.

        :param arr_nb_children: 1D array of int
        :param arr_prob_nb_children: 1d array of floats
        :param condition: optional, 1d array of bool, default None.
        :param dico_default_values: optional, dictionnary, default {}.
        :param prob_failure: optional, float, default None.
        :param age_attribute:
        :param mother_attribute:
        :param father_attribute:
        :param id_attribute:
        :param position_attribute:
        :param territory_attribute:
        :param mate_attribute:
        :param pregnancy_attribute:
        """
        selected_females = self.df_population[pregnancy_attribute]
        if condition is not None:
            selected_females = selected_females & condition
        if prob_failure is not None:
            selected_females = selected_females & \
                               (np.random.uniform(0, 1, (self.df_population.nb_rows,)) >= prob_failure)

        df_selected_female = self.df_population[selected_females]

        # get number of babies per females
        prob = arr_prob_nb_children.astype('float64')
        prob = prob/prob.sum()
        arr_nb_baby = np.random.choice(arr_nb_children, df_selected_female.nb_rows, p=prob)

        # start building the children DataFrame
        df_children = DataFrameXS()
        df_children[mother_attribute] = np.repeat(df_selected_female[id_attribute], arr_nb_baby, axis=0)
        df_children[father_attribute] = np.repeat(df_selected_female[mate_attribute], arr_nb_baby, axis=0)
        pos = np.repeat(df_selected_female[position_attribute], arr_nb_baby, axis=0)
        df_children[position_attribute] = pos
        df_children[territory_attribute] = pos

        # defines the gender of the offsprings
        gender = 1 * (np.random.uniform(0, 1, (df_children.shape[0],)) >= 0.5)
        df_children[gender_attribute] = gender

        # fill non trivial attributes
        df_children[pregnancy_attribute] = False
        df_children[age_attribute] = 0
        df_children[id_attribute] = np.arange(self.counter_id, self.counter_id + df_children.shape[0])
        self.counter_id = self.counter_id + df_children.shape[0]

        # take care of the provided default values
        for attr, def_value in dico_default_values.items():
            df_children[attr] = def_value

        # take care of the rest
        set_treated_col = set([mother_attribute, father_attribute, position_attribute, territory_attribute,
                               gender_attribute, pregnancy_attribute, age_attribute, id_attribute])
        for col_name in self.df_population.list_col_name:
            if col_name in set_treated_col or col_name in dico_default_values:
                continue
            if col_name in self.dict_default_val:
                df_children[col_name] = self.dict_default_val[col_name]
            else:
                df_children[col_name] = None

        # set pregnancy of female that gave birth to False
        self.df_population[pregnancy_attribute] = self.df_population[pregnancy_attribute] & ~selected_females

        # concatenate the two dataframe
        self.df_population.concat(df_children)

    def create_babies_poisson_law(self,
                                  lambda_poisson,
                                  max_offspring_per_litter,
                                  dico_default_values={},
                                  prob_failure=None,
                                  age_attribute='age',
                                  mother_attribute='mom_id',
                                  father_attribute='dad_id',
                                  gender_attribute='gender',
                                  id_attribute='col_id',
                                  position_attribute='position',
                                  territory_attribute='territory',
                                  mate_attribute='current_mate',
                                  pregnancy_attribute='is_pregnant'):
        pass


class MammalReproductionGeneticMarker:
    """
    type of markers are encoded with integers.
    """
    def __init__(self):
        self.list_markers = []
        super().__init__()

    def add_marker(self, name_marker, default_value=0):
        """
        add 4 columns to the dataframe 2 for the markers of the individual, 2 for the markers of its mate if female.
        :param name_marker: string, name of the marker
        :param default_value: default 0.
        """
        if name_marker in self.list_markers:
            raise ValueError("Marker " + name_marker + " already exists.")
        self.list_markers.append(name_marker)
        self.df_population['marker_' + name_marker + '_1'] = default_value
        self.df_population['marker_' + name_marker + '_2'] = default_value
        self.df_population['marker_' + name_marker + '_from_mate_1'] = default_value
        self.df_population['marker_' + name_marker + '_from_mate_2'] = default_value

    def find_random_mate_on_position(self,
                                     id_attribute='col_id',
                                     position_attribute='position',
                                     gender_attribute='gender',
                                     mate_attribute='current_mate',
                                     pregnancy_attribute='is_pregnant'):
        """
        todo
        :param id_attribute:
        :param position_attribute:
        :param gender_attribute:
        :param mate_attribute:
        :param pregnancy_attribute:
        """
        arr_id = np.array(self.df_population[id_attribute], dtype=np.int32)
        position = np.array(self.df_population[position_attribute], dtype=np.int32)
        gender = np.array(self.df_population[gender_attribute] == 'male', dtype=np.bool_)
        nb_vertex = self.graph.connections.shape[0]
        max_male, max_female = reprod_part1(nb_vertex, position, gender)
        rand = np.random.permutation(position.shape[0])
        list_arr_markers = []
        for name in self.list_markers:
            list_arr_markers.append(np.array(self.df_population['marker_' + name + '_1']))
            list_arr_markers.append(np.array(self.df_population['marker_' + name + '_2']))
        stacked_marker_array = np.column_stack(list_arr_markers)
        stacked_marker_array = np.array(stacked_marker_array, dtype=np.int32)
        # print(stacked_marker_array)
        col_pregnancy, col_mate, array_markers = repro_marker_find_mate_part_2(nb_vertex, max_male, max_female, arr_id,
                                                                               position, gender, stacked_marker_array,
                                                                               rand)
        self.df_population[mate_attribute] = col_mate
        self.df_population[pregnancy_attribute] = col_pregnancy
        for i, name in enumerate(self.list_markers):
            self.df_population['marker_' + name + '_from_mate_1'] = array_markers[:, 2 * i]
            self.df_population['marker_' + name + '_from_mate_2'] = array_markers[:, 2 * i + 1]

    def create_babies(self,
                      target_nb_child,
                      prob_nb_children,
                      dico_default_values={},
                      prob_failure=None,
                      age_attribute='age',
                      mother_attribute='mom_id',
                      father_attribute='dad_id',
                      gender_attribute='gender',
                      id_attribute='col_id',
                      position_attribute='position',
                      territory_attribute='territory',
                      mate_attribute='current_mate',
                      pregnancy_attribute='is_pregnant'):
        """
        todo
        :param target_nb_child:
        :param prob_nb_children:
        :param dico_default_values:
        :param prob_failure:
        :param age_attribute:
        :param mother_attribute:
        :param father_attribute:
        :param id_attribute:
        :param position_attribute:
        :param territory_attribute:
        :param mate_attribute:
        :param pregnancy_attribute:
        """
        df_tp = self.df_population[self.df_population[pregnancy_attribute]].copy()
        if prob_failure is not None:
            nb_pregn = df_tp.shape[0]
            df_tp = self.df_population[np.random.uniform(0, 1, (nb_pregn,)) >= prob_failure ]

        # extract features
        mom_id = np.array(df_tp[id_attribute])
        dad_id = np.array(df_tp[mate_attribute])
        pos = np.array(df_tp[territory_attribute])
        list_col = [mom_id, dad_id, pos]
        for name in self.list_markers:
            list_col.append(np.array(df_tp['marker_' + name + '_1']))
            list_col.append(np.array(df_tp['marker_' + name + '_2']))
            list_col.append(np.array(df_tp['marker_' + name + '_from_mate_1']))
            list_col.append(np.array(df_tp['marker_' + name + '_from_mate_2']))

        # concatenate the columns
        concat = np.column_stack(list_col)
        arr_nb_baby = np.random.choice(target_nb_child, concat.shape[0], p=prob_nb_children)

        # array with the non-default values for the children
        result_before_marker_selection = np.repeat(concat, arr_nb_baby, axis=0)

        # we now create the array where the markers have been selected
        list_col = [result_before_marker_selection[:, 0], result_before_marker_selection[:, 1],
                    result_before_marker_selection[:, 2]]
        for i, _ in enumerate(self.list_markers):
            mom_marker = result_before_marker_selection[:, (3 + 4*i):(5 + 4*i)]
            dad_marker = result_before_marker_selection[:, (5 + 4*i):(7 + 4*i)]
            rand_1 = 1 * (np.random.uniform(0, 1, (mom_marker.shape[0],)) > 0.5)
            list_col.append(mom_marker[range(mom_marker.shape[0]), rand_1].copy())
            rand_2 = 1 * (np.random.uniform(0, 1, (dad_marker.shape[0],)) > 0.5)
            list_col.append(dad_marker[range(dad_marker.shape[0]), rand_2].copy())

        result = np.column_stack(list_col)

        # create the df
        df = pd.DataFrame(columns=list(self.df_population))
        df[mother_attribute] = result[:, 0]
        df[father_attribute] = result[:, 1]
        df[position_attribute] = result[:, 2]
        df[territory_attribute] = result[:, 2]
        for i, name in enumerate(self.list_markers):
            df['marker_' + name + '_1'] = result[:, 3+2*i]
            df['marker_' + name + '_2'] = result[:, 4+2*i]
            df['marker_' + name + '_from_mate_1'] = 0
            df['marker_' + name + '_from_mate_2'] = 0

        # defines the gender of the offsprings
        rand = np.random.uniform(0, 1, (df.shape[0],))
        df[gender_attribute] = np.array(['male'*(float(u) >= 0.5)+'female'*(float(u) < 0.5) for u in rand])
        df[pregnancy_attribute] = np.full((df.shape[0],), False)
        df[age_attribute] = np.full((df.shape[0],), 0)
        df[id_attribute] = np.arange(self.counter_id, self.counter_id + df.shape[0])
        self.counter_id = self.counter_id + df.shape[0]
        for attr, def_value in dico_default_values.items():
            df[attr] = np.full((df.shape[0],), def_value)

        # concatenate the two dataframe
        self.df_population = pd.concat([self.df_population, df], copy=True)
        self.df_population[pregnancy_attribute] = False
