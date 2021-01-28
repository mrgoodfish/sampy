from sampy.utils.decorators import use_debug_mode
from sampy.pandas_xs.pandas_xs import DataFrameXS

from sampy.data_processing.write_file import counts_to_csv
from sampy.graph.builtin_graph import SquareGridWithDiag
from sampy.agent.builtin_agent import BasicMammal
# from sampy.disease.single_species.builtin_disease import ContactCustomProbTransitionPermanentImmunity
from constant_examples import ARR_PROB_DEATH_FEMALE, ARR_PROB_DEATH_MALE

import numpy as np

# use_debug_mode(DataFrameXS)
# use_debug_mode(SquareGridWithDiag)
# use_debug_mode(BasicMammal)
# use_debug_mode(ContactCustomProbTransitionPermanentImmunity)


my_graph = SquareGridWithDiag(shape=(100, 100))
my_graph.create_vertex_attribute('K', 10.)

agents = BasicMammal(graph=my_graph)
# print(agents.df_population.list_col_name)
# agents.add_attribute('test', def_value=True)
# print(agents.df_population.list_col_name)

dico_agents = {}
dico_agents['age'] = [52, 52, 52, 52, 52, 52, 52, 52, 52, 52]
dico_agents['gender'] = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
dico_agents['territory'] = [0, 0, 125, 125, 1021, 1021, 2034, 2034, 6321, 6321]
dico_agents['position'] = [0, 0, 125, 125, 1021, 1021, 2034, 2034, 6321, 6321]

agents.add_agents(dico_agents)

# rabies = ContactCustomProbTransitionPermanentImmunity(disease_name='rabies', host=agents)


list_count_vects_old = []
nb_year = 50
for i in range(nb_year * 52 + 1):
    if i % 52 == 0:
        print('year', i // 52, ':', agents.number_agents)
        old = agents.df_population['age'] > 52
        list_count_vects_old.append(agents.count_pop_per_vertex(position_attribute='territory', condition=old))
        agents.save_population_to_csv("C:/post_doc/courses/example_output_sampy/Caro/pop_" + str(i//52) + "y_test.csv",
                                      sep=';')


    agents.tick()
    my_graph.tick()

    agents.kill_too_old(52 * 6 - 1)
    agents.natural_death_orm_methodology(ARR_PROB_DEATH_MALE, ARR_PROB_DEATH_FEMALE)
    agents.kill_children_whose_mother_is_dead(12)

    agents.mov_around_territory(0.8)

    if i % 52 == 15:
        agents.find_random_mate_on_position(1., position_attribute='territory')
    if i % 52 == 22:
        agents.create_offsprings_custom_prob(np.array([4, 5, 6, 7, 8, 9]), np.array([0.1, 0.2, 0.2, 0.2, 0.2, 0.1]))
    if i % 52 == 40:
        can_move = agents.df_population['age'] > 11
        agents.change_territory(condition=can_move)

