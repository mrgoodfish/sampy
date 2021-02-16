from sampy.utils.decorators import use_debug_mode
from sampy.pandas_xs.pandas_xs import DataFrameXS

import numpy as np
from sampy.graph.graph_from_ORM_xml import GraphFromORMxml
from sampy.agent.builtin_agent import BasicMammal
# from sampy.disease.single_species.builtin_disease import ContactCustomProbTransitionPermanentImmunity
# other imports

from constant_examples import (ARR_PROB_DEATH_FEMALE,
                               ARR_PROB_DEATH_MALE)


# lines about debug mode
# use_debug_mode(DataFrameXS)
# use_debug_mode(GraphFromORMxml)
# use_debug_mode(BasicMammal)
# use_debug_mode(ContactCustomProbTransitionPermanentImmunity)

# create the landscape
my_graph = GraphFromORMxml(path_to_xml='C:/post_doc/data/orm_related_data/xml_landscapes/emily/ORMlandscape.xml')
# print(my_graph.dict_cell_id_to_ind)

# create the agents
racoons = BasicMammal(graph=my_graph)
# print(help(racoons))

# create the disease
# rabies = ContactCustomProbTransitionPermanentImmunity(disease_name='rabies', host=racoons)
# print(racoons.df_population.list_col_name)

# create some agents
dict_new_agents = dict()
dict_new_agents['age'] = [52, 52, 52, 52, 52, 52, 52, 52, 52, 52]
dict_new_agents['gender'] = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
dict_new_agents['territory'] = [my_graph.dict_cell_id_to_ind['c166_154'], my_graph.dict_cell_id_to_ind['c166_154'],
                                125, 125, 1021, 1021, 2034, 2034, 6321, 6321]
dict_new_agents['position'] = [my_graph.dict_cell_id_to_ind['c166_154'], my_graph.dict_cell_id_to_ind['c166_154'],
                               125, 125, 1021, 1021, 2034, 2034, 6321, 6321]

racoons.add_agents(dict_new_agents)

# print(racoons.df_population.get_as_pandas_dataframe())

list_count = []
nb_year_simu = 100
for i in range(nb_year_simu * 52 + 1):

    if i % 52 == 0:
        print('year', i//52, ':', racoons.number_agents)
        condition_count = racoons.df_population['gender'] == 1
        list_count.append(racoons.count_pop_per_vertex(position_attribute='territory', condition=condition_count))

    racoons.tick()
    my_graph.tick()

    racoons.kill_too_old(52 * 6 - 1)
    racoons.natural_death_orm_methodology(ARR_PROB_DEATH_MALE, ARR_PROB_DEATH_FEMALE)
    racoons.kill_children_whose_mother_is_dead(11)

    racoons.mov_around_territory(0.8)

    if i % 52 == 15:
        racoons.find_random_mate_on_position(1., position_attribute='territory')
    if i % 52 == 22:
        racoons.create_offsprings_custom_prob(np.array([4, 5, 6, 7, 8, 9]), np.array([0.1, 0.2, 0.2, 0.2, 0.2, 0.1]))
    if i % 52 == 40:
        can_move = racoons.df_population['age'] > 11
        racoons.dispersion_with_varying_nb_of_steps(np.array([1, 2, 3, 4]),
                                                    np.array([.25, .25, .25, .25]),
                                                    condition=can_move)
