import numpy as np
from ORM_related_addons.graph_from_ORM_xml import GraphFromORMxml
from sampy.agent.builtin_agent import BasicMammal
from sampy.data_processing.write_file import counts_to_csv
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
# my_graph.save_table_id_of_vertices_to_indices('C:/post_doc/data/orm_related_data/xml_landscapes/emily/table_graph.csv',
#                                               ';')

# create the agents
racoons = BasicMammal(graph=my_graph)
# print(help(racoons))

# create some agents
dict_new_agents = dict()
dict_new_agents['age'] = [52, 52, 52, 52, 52, 52, 52, 52, 52, 52]
dict_new_agents['gender'] = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
dict_new_agents['territory'] = [my_graph.dict_cell_id_to_ind['c166_154'], my_graph.dict_cell_id_to_ind['c166_154'],
                                125, 125, 1021, 1021, 2034, 2034, 6321, 6321]
dict_new_agents['position'] = [my_graph.dict_cell_id_to_ind['c166_154'], my_graph.dict_cell_id_to_ind['c166_154'],
                               125, 125, 1021, 1021, 2034, 2034, 6321, 6321]
racoons.add_agents(dict_new_agents)

# racoons.load_population_from_csv('C:/post_doc/data/output_sim/emily/pop.csv', sep=';')


list_count_female = []
list_count_male = []
nb_year_simu = 10
for i in range(nb_year_simu * 52 + 1):

    if i % 52 == 0:
        print('year', i//52, ':', racoons.number_agents)
        condition_count = racoons.df_population['gender'] == 1
        list_count_female.append(racoons.count_pop_per_vertex(position_attribute='territory', condition=condition_count))
        list_count_male.append(racoons.count_pop_per_vertex(position_attribute='territory', condition=~condition_count))

    racoons.tick()
    my_graph.tick()

    racoons.kill_too_old(52*7 - 1)
    my_condition = racoons.df_population['age'] >= 12
    racoons.natural_death_orm_methodology(ARR_PROB_DEATH_MALE, ARR_PROB_DEATH_FEMALE,
                                          position_attribute='territory',
                                          condition=None, condition_count=my_condition)
    racoons.kill_children_whose_mother_is_dead(20)

    racoons.mov_around_territory(0.2)

    if i % 52 == 15:
        racoons.find_random_mate_on_position(1., position_attribute='territory')
    if i % 52 == 22:
        racoons.create_offsprings_custom_prob(np.array([0, 5, 6, 7, 8, 9]), np.array([0.1, 0.2, 0.2, 0.2, 0.2, 0.1]))
    if i % 52 == 40:
        can_move = racoons.df_population['age'] > 11
        racoons.dispersion_with_varying_nb_of_steps(np.array([1, 2, 3, 4]),
                                                    np.array([.25, .25, .25, .25]),
                                                    condition=can_move)

racoons.save_population_to_csv('C:/post_doc/data/output_sim/emily/pop_3.csv', sep=';')
counts_to_csv(list_count_male, my_graph, 'C:/post_doc/data/output_sim/emily/male_per_cell.csv', sep=';')
counts_to_csv(list_count_female, my_graph, 'C:/post_doc/data/output_sim/emily/female_per_cell.csv', sep=';')
