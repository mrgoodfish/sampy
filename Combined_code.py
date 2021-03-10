# In the Anaconda prompt, start by putting in cd C:\Users\Emily\Documents\Leighton_research\Rabies_work\Francois_python_help\Feb25_2021\sampy_master_EAEdits\
# Then type python3 Combined_code.py

import numpy as np
from ORM_related_addons.graph_from_ORM_xml import GraphFromORMxml
# from sampy.agent.builtin_agent import BasicMammal
from ORM_related_addons.ORM_like_agents import ORMLikeAgent
from sampy.data_processing.write_file import counts_to_csv
# from sampy.disease.single_species.builtin_disease import ContactCustomProbTransitionPermanentImmunity

import time


starting_time = time.time()

# create the landscape
# This is the larger Ontario landscape
my_graph = GraphFromORMxml(path_to_xml='C:/post_doc/data/orm_related_data/xml_landscapes/emily/ORMlandscape.xml')
my_graph.save_table_id_of_vertices_to_indices('C:/post_doc/data/orm_related_data/xml_landscapes/emily/table_graph.csv',
                                              ';')

# create the agents
racoons = ORMLikeAgent(graph=my_graph)
# print(help(racoons))

# create some agents
# Adult female raccoons tend to start breeding around the age of 1 year (i.e., 52 weeks)
dict_new_agents = dict()
dict_new_agents['age'] = [52, 52, 52, 52, 52, 52, 52, 52, 52, 52]
dict_new_agents['gender'] = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
dict_new_agents['territory'] = [my_graph.dict_cell_id_to_ind['c007_032'],
                                my_graph.dict_cell_id_to_ind['c007_032'],
                                my_graph.dict_cell_id_to_ind['c152_142'],
                                my_graph.dict_cell_id_to_ind['c152_142'],
                                my_graph.dict_cell_id_to_ind['c321_289'],
                                my_graph.dict_cell_id_to_ind['c321_289'],
                                my_graph.dict_cell_id_to_ind['c325_131'],
                                my_graph.dict_cell_id_to_ind['c325_131'],
                                my_graph.dict_cell_id_to_ind['c378_005'],
                                my_graph.dict_cell_id_to_ind['c378_005']]
dict_new_agents['position'] = [my_graph.dict_cell_id_to_ind['c007_032'],
                                my_graph.dict_cell_id_to_ind['c007_032'],
                                my_graph.dict_cell_id_to_ind['c152_142'],
                                my_graph.dict_cell_id_to_ind['c152_142'],
                                my_graph.dict_cell_id_to_ind['c321_289'],
                                my_graph.dict_cell_id_to_ind['c321_289'],
                                my_graph.dict_cell_id_to_ind['c325_131'],
                                my_graph.dict_cell_id_to_ind['c325_131'],
                                my_graph.dict_cell_id_to_ind['c378_005'],
                                my_graph.dict_cell_id_to_ind['c378_005']]
racoons.add_agents(dict_new_agents)

# racoons.load_population_from_csv('C:/post_doc/data/output_sim/emily/pop.csv', sep=';')

list_yearly_count = []
nb_year_simu = 150
# which represents 52 weeks per year plus one additional week
for i in range(nb_year_simu * 52 + 1): 

    if i % 52 == 0:
        print('year', i//52, ':', racoons.number_agents)
        list_yearly_count.append(racoons.count_pop_per_vertex(position_attribute='territory'))

    racoons.tick()
    my_graph.tick()

    # I changed this code from 52*7-1 to 52*8-1, since I added an extra year to what Francois originally had in the constant_examples
    racoons.kill_too_old(52*8 - 1)
    my_condition = racoons.df_population['age'] >= 20
    racoons.mortality_from_v08(np.array([0.6, .4, .3, .3, .3, .6, .6, .6]), my_condition,
                               position_attribute='territory')
    # racoons.natural_death_orm_methodology(ARR_PROB_DEATH_MALE, ARR_PROB_DEATH_FEMALE,
    #                                       position_attribute='territory',
    #                                       condition=None, condition_count=my_condition)
    # 20 was put here because that's the raccoon's age of independence
    racoons.kill_children_whose_mother_is_dead(20)

    racoons.mov_around_territory(0.2)

    if i % 52 == 9: # I changed this to 0, since that's when females start mating (this tells the code that i is a multiple of 52
        racoons.find_random_mate_on_position(1., position_attribute='territory') # Probability of getting pregnant. Do we know this?
    if i % 52 == 18: # I changed this to 18 assuming this is where we specify which week the females give birth.
        # Since these were unknown (at least to me), I calculated them using a probability density function calculator (solvemymath.com)
        yy = (racoons.df_population['age'] >= 20) & (racoons.df_population['age'] < 75)
        racoons.create_offsprings_custom_prob(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8]),
                                              np.array([0.0001, 0.0044, 0.0539, 0.2416, 0.4, 0.2416, 0.0539, 0.0044, 0.0001]),
                                              condition=yy, prob_failure=.35)
        adult = racoons.df_population['age'] >= 75
        racoons.create_offsprings_custom_prob(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8]),
                                              np.array([0.0001, 0.0044, 0.0539, 0.2416, 0.4, 0.2416, 0.0539, 0.0044,
                                                        0.0001]),
                                              condition=adult, prob_failure=.05)
        racoons.df_population['is_pregnant'] = False
    if i > 200: # *** I don't remember what this number (40) is supposed to mean ***
        yy = (racoons.df_population['age'] >= 20) & (racoons.df_population['age'] < 75)
        adult = racoons.df_population['age'] >= 75
        male = racoons.df_population['gender'] == 0

        male_yy = male & yy
        female_yy = ~male & yy
        male_adult = male & adult
        female_adult = ~male & adult

        racoons.orm_dispersion(i, [37, 38, 39, 40, 41, 42], male_yy,
                               np.array([0, 1, 2, 3, 4, 5, 6, 8]),
                               np.array([75.1, 12.9, 6.5, 1.8, 0.9, 0.9, 0.92, 1.])/100.)
        racoons.orm_dispersion(i, [37, 38, 39, 40, 41, 42], female_yy,
                               np.array([0, 1, 2, 3, 4, 5, 6, 8]),
                               np.array([90.8, 4.07, 1.4, 0.3, 0.7, 0.3, 1.02, 1.]) / 100.)

        racoons.orm_dispersion(i, list(range(7, 43)), male_adult,
                               np.array([0, 1, 2, 3, 4, 5, 7, 8]),
                               np.array([88.9, 4.25, 2.6, 0.9, 0.7, 0.9, 1., 1.]) / 100.)
        racoons.orm_dispersion(i, [7, 8, 9, 10, 11, 12, 13, 14, 37, 38, 39, 40, 41, 42], female_adult,
                               np.array([0, 1, 2, 3, 4, 5, 6]),
                               np.array([92.3, 3.06, 1., 0.9, 0.4, 0.7, 0.73]) / 100.)

racoons.save_population_to_csv('C:/post_doc/data/output_sim/emily/ontario_project/pop.csv', sep=';')
# counts_to_csv(list_yearly_count, my_graph, 'C:/Users/Emily/Documents/Leighton_Research/Rabies_work/Francois_python_help/Feb25_2021/sampy_master_EAEdits/Feb25_2021_pop_per_cell.csv', sep=';')
# counts_to_csv(list_count_female, my_graph, 'C:/Users/Emily/Documents/Leighton_Research/Rabies_work/Francois_python_help/Feb25_2021/sampy_master_EAEdits/Feb25_2021_female_per_cell.csv', sep=';')

print(' --------- ')
total_time = time.time() - starting_time
minutes = int(total_time) // 60
seconds = int(total_time) % 60
print('Total time to complete the simulation:', str(minutes), 'min', str(seconds), 'seconds.')





























