import numpy as np
from ORM_related_addons.graph_from_ORM_xml import GraphFromORMxml
# from sampy.agent.builtin_agent import BasicMammal
from ORM_related_addons.ORM_like_agents import ORMLikeAgent
from sampy.data_processing.write_file import counts_to_csv
from sampy.disease.single_species.builtin_disease import ContactCustomProbTransitionPermanentImmunity

import time

# ------------------------
# PARAMETERS
path_to_pop_csv = ''
cells_to_infect = ['', '', '']
level_initial_infection = 0.5
contact_propagation_proba = 0.035
prob_death_rabies = 0.8

path_to_output_csv_new_infection = ''
path_to_output_csv_death_from_rabies = ''

# ------------------------

starting_time = time.time()

# create the landscape
# This is the larger Ontario landscape
my_graph = GraphFromORMxml(path_to_xml='C:/post_doc/data/orm_related_data/xml_landscapes/emily/ORMlandscape.xml')
my_graph.save_table_id_of_vertices_to_indices('C:/post_doc/data/orm_related_data/xml_landscapes/emily/table_graph.csv',
                                              ';')

# create the agents
racoons = ORMLikeAgent(graph=my_graph)
racoons.load_population_from_csv(path_to_pop_csv, sep=';')

# create the disease
rabies = ContactCustomProbTransitionPermanentImmunity(host=racoons, disease_name='rabies')

# infect the initial cell
for i, cell in enumerate(cells_to_infect):
    if i == 0:
        infected_agent = racoons.df_population['territory'] == my_graph.dict_cell_id_to_ind[cell]
    else:
        infected_agent = infected_agent | (racoons.df_population['territory'] == my_graph.dict_cell_id_to_ind[cell])
infected_agent = infected_agent & (np.random.uniform(0, 1, racoons.df_population.nb_rows) < level_initial_infection)
racoons.df_population['inf_rabies'] = infected_agent
rabies.initialize_counters_of_newly_infected(infected_agent,
                                             np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]),
                                             np.array([0.01, 0.05, 0.05, 0.1, 0.15, 0.2, 0.15, 0.1, 0.05,
                                                       0.05, 0.05, 0.02, 0.01, 0.01]))

list_new_infection = []
list_dead_from_infection = []
nb_year_simu = 15
# which represents 52 weeks per year plus one additional week
for i in range(nb_year_simu * 52 + 1):

    if i % 52 == 0:
        print('year', i//52, ':', racoons.number_agents)

    racoons.tick()
    my_graph.tick()
    rabies.decrement_counter()

    # natural death
    racoons.kill_too_old(52*8 - 1)
    my_condition = racoons.df_population['age'] >= 20
    racoons.mortality_from_v08(np.array([0.6, .4, .3, .3, .3, .6, .6, .6]), my_condition,
                               position_attribute='territory')

    # rabies part
    arr_new_infected = rabies.contact_contagion(contact_propagation_proba, return_arr_new_infected=True,
                                                condition=(racoons.df_population['age'] >= 20))
    rabies.initialize_counters_of_newly_infected(arr_new_infected,
                                                 np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]),
                                                 np.array([0.01, 0.05, 0.05, 0.1, 0.15, 0.2, 0.15, 0.1, 0.05,
                                                           0.05, 0.05, 0.02, 0.01, 0.01]))
    list_new_infection.append(racoons.count_pop_per_vertex(condition=arr_new_infected))

    list_dead_from_infection.append(rabies.transition_between_states('con', 'death', proba_death=prob_death_rabies,
                                                                     return_transition_count=True))
    rabies.transition_between_states('con', 'imm')
    rabies.transition_between_states('inf', 'con', arr_nb_timestep=np.array([1]),
                                     arr_prob_nb_timestep=np.array([1.]))

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

counts_to_csv(list_new_infection, my_graph, path_to_output_csv_new_infection, sep=';')
counts_to_csv(list_dead_from_infection, my_graph, path_to_output_csv_death_from_rabies, sep=';')

print(' --------- ')
total_time = time.time() - starting_time
minutes = int(total_time) // 60
seconds = int(total_time) % 60
print('Total time to complete the simulation:', str(minutes), 'min', str(seconds), 'seconds.')