from sampy.agent.builtin_agent import BasicMammal
from ORM_related_addons.graph_from_ORM_xml import GraphFromORMxml

import numpy as np

from constant_examples import (ARR_PROB_DEATH_FEMALE,
                               ARR_PROB_DEATH_MALE)

# use_debug_mode(DataFrameXS)
# use_debug_mode(SquareGridWithDiag)
# use_debug_mode(BasicMammal)

# graph
my_graph = GraphFromORMxml(path_to_xml='C:/post_doc/data/orm_related_data/xml_landscapes/emily/ORMlandscape.xml')
# my_graph.create_vertex_attribute('K', 10.)

# agent
agents = BasicMammal(graph=my_graph)
# print(agents.df_population.list_col_name)

# # disease
# disease = ContactCustomProbTransitionPermanentImmunity(disease_name='rabies', host=agents)
# print(agents.df_population.list_col_name)

# create some agents
dict_new_agents = dict()
dict_new_agents['age'] = [52, 52, 52, 52, 52, 52, 52, 52, 52, 52]
dict_new_agents['gender'] = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
dict_new_agents['territory'] = [0, 0, 125, 125, 1021, 1021, 2034, 2034, 6321, 6321]
dict_new_agents['position'] = [0, 0, 125, 125, 1021, 1021, 2034, 2034, 6321, 6321]

agents.add_agents(dict_new_agents)


list_count = []
nb_year_simu = 100
for i in range(nb_year_simu * 52 + 1):

    if i % 52 == 0:
        print('year', i//52, ':', agents.number_agents)
        condition_count = agents.df_population['gender'] == 1
        list_count.append(agents.count_pop_per_vertex(position_attribute='territory', condition=condition_count))

    agents.tick()
    my_graph.tick()

    agents.kill_too_old(52 * 6 - 1)
    agents.natural_death_orm_methodology(ARR_PROB_DEATH_MALE, ARR_PROB_DEATH_FEMALE)
    agents.kill_children_whose_mother_is_dead(11)

    agents.mov_around_territory(0.8)

    if i % 52 == 15:
        agents.find_random_mate_on_position(1., position_attribute='territory')
    if i % 52 == 22:
        agents.create_offsprings_custom_prob(np.array([4, 5, 6, 7, 8, 9]), np.array([0.1, 0.2, 0.2, 0.2, 0.2, 0.1]))
    if i % 52 == 40:
        can_move = agents.df_population['age'] > 11
        agents.dispersion_with_varying_nb_of_steps(np.array([1, 2, 3, 4]),
                                                   np.array([.25, .25, .25, .25]),
                                                   condition=can_move)

# agents.save_population_to_csv("C:/post_doc/courses/example_output_sampy/Agathe/final_pop.csv", sep=';', index=False)
# counts_to_csv(list_count, my_graph, "C:/post_doc/courses/example_output_sampy/Agathe/annual_count_per_cell.csv")
