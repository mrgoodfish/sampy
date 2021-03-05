from sampy.graph.builtin_graph import SquareGridWithDiag
from sampy.agent.builtin_agent import BasicMammal
from sampy.disease.single_species.builtin_disease import ContactCustomProbTransitionPermanentImmunity
from sampy.intervention.built_in_interventions import BasicVaccination

from constant_examples import ARR_PROB_DEATH_FEMALE, ARR_PROB_DEATH_MALE

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def convert_to_2d_img(array, dict_translate):
    img = np.full((100, 100), 0, dtype=np.int32)
    for i, val in enumerate(array):
        coord = dict_translate[i]
        img[coord[0], coord[1]] = val
    return img


my_graph = SquareGridWithDiag(shape=(100, 100))
my_graph.create_vertex_attribute('K', 10.)
dict_translate = {ind: id_vert for id_vert, ind in my_graph.dict_cell_id_to_ind.items()}
# print(dict_translate)

agents = BasicMammal(graph=my_graph)

# dico_agents = {}
# dico_agents['age'] = [52, 52, 52, 52, 52, 52, 52, 52, 52, 52]
# dico_agents['gender'] = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
# dico_agents['territory'] = [0, 0, 125, 125, 1021, 1021, 2034, 2034, 6321, 6321]
# dico_agents['position'] = [0, 0, 125, 125, 1021, 1021, 2034, 2034, 6321, 6321]
#
# agents.add_agents(dico_agents)
#
# # rabies = ContactCustomProbTransitionPermanentImmunity(disease_name='rabies', host=agents)
#
#
# list_count_vects_old = []
# nb_year = 100
# for i in range(nb_year * 52 + 1):
#     if i % 52 == 0:
#         print('year', i // 52, ':', agents.number_agents)
#         old = agents.df_population['age'] > 52
#         list_count_vects_old.append(agents.count_pop_per_vertex(position_attribute='territory', condition=old))
#
#     agents.tick()
#     my_graph.tick()
#
#     agents.kill_too_old(52 * 6 - 1)
#     agents.natural_death_orm_methodology(ARR_PROB_DEATH_MALE, ARR_PROB_DEATH_FEMALE)
#     agents.kill_children_whose_mother_is_dead(12)
#
#     agents.mov_around_territory(0.8)
#
#     if i % 52 == 15:
#         agents.find_random_mate_on_position(1., position_attribute='territory')
#     if i % 52 == 22:
#         agents.create_offsprings_custom_prob(np.array([4, 5, 6, 7, 8, 9]), np.array([0.1, 0.2, 0.2, 0.2, 0.2, 0.1]))
#     if i % 52 == 40:
#         can_move = agents.df_population['age'] > 11
#         agents.dispersion_with_varying_nb_of_steps(np.array([1, 2, 3, 4]),
#                                                    np.array([.25, .25, .25, .25]),
#                                                    condition=can_move)
#
# agents.save_population_to_csv('C:/post_doc/data/output_sim/pop_test_vaccine/pop.csv', ';')

agents.load_population_from_csv('C:/post_doc/data/output_sim/pop_test_vaccine/pop.csv', sep=';')

disease = ContactCustomProbTransitionPermanentImmunity(host=agents, disease_name='rabies')
agents.df_population['inf_rabies'] = agents.df_population['territory'] == my_graph.dict_cell_id_to_ind[(10, 10)]

intervention = BasicVaccination(disease=disease, duration_vaccine=200)

dict_vaccine = {(40 + i, j): 0.7 for i in range(20) for j in range(100)}

list_infected = []
nb_year = 8
for i in range(nb_year * 52 + 1):
    if i % 52 == 0:
        print('year', i // 52, ':', agents.number_agents)

    list_infected.append(convert_to_2d_img(disease.count_nb_status_per_vertex('inf'), dict_translate))

    intervention.update_vaccine_status()
    if i % 52 == 0:
        condition = ~agents.df_population['imm_rabies'] & \
                    ~agents.df_population['inf_rabies'] & \
                    ~agents.df_population['con_rabies']
        intervention.apply_vaccine_from_dict(my_graph, dict_vaccine, condition=condition)

    agents.tick()
    my_graph.tick()
    disease.decrement_counter()

    agents.kill_too_old(52 * 6 - 1)
    agents.natural_death_orm_methodology(ARR_PROB_DEATH_MALE, ARR_PROB_DEATH_FEMALE)
    agents.kill_children_whose_mother_is_dead(12)

    arr_new_infected = disease.contact_contagion(0.05, return_arr_new_infected=True)
    disease.initialize_counters_of_newly_infected(arr_new_infected, np.array([2, 3, 4, 5]),
                                                  np.array([0.25, 0.25, 0.25, 0.25]))
    disease.transition_between_states('con', 'death', proba_death=0.9)
    disease.transition_between_states('con', 'imm')
    disease.transition_between_states('inf', 'con', arr_nb_timestep=np.array([2, 3, 4, 5]),
                                      arr_prob_nb_timestep=np.array([0.25, 0.25, 0.25, 0.25]))

    agents.mov_around_territory(0.2)

    if i % 52 == 15:
        agents.find_random_mate_on_position(1., position_attribute='territory')
    if i % 52 == 22:
        agents.create_offsprings_custom_prob(np.array([4, 5, 6, 7, 8, 9]), np.array([0.1, 0.2, 0.2, 0.2, 0.2, 0.1]))
    if i % 52 == 40:
        can_move = agents.df_population['age'] > 11
        agents.dispersion_with_varying_nb_of_steps(np.array([1, 2, 3, 4]),
                                                   np.array([.25, .25, .25, .25]),
                                                   condition=can_move)

fig = plt.figure()
img = plt.imshow(list_infected[0], animated=True, vmin=0, vmax=8)


def iter_frame(k):
    img.set_data(list_infected[k])
    return img,


anim = FuncAnimation(fig, iter_frame, frames=len(list_infected), interval=50)
plt.show()
