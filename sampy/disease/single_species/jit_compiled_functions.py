import numba as nb
import numpy as np


# ----------------------------------------------------------------------------------------------------------------------
# generic functions
@nb.njit
def conditional_count_return_full_array(nb_vertex, arr_pos, arr_cond):
    pos_count = np.zeros((nb_vertex,), dtype=np.int32)
    for i in range(arr_pos.shape[0]):
        if arr_cond[i]:
            pos_count[arr_pos[i]] += 1
    rv = np.zeros((arr_pos.shape[0],), dtype=np.int32)
    for i in range(arr_pos.shape[0]):
        rv[i] = pos_count[arr_pos[i]]
    return rv

# ----------------------------------------------------------------------------------------------------------------------
# specialised functions


@nb.njit
def base_conditional_count_nb_agent_per_vertex(arr_condition, arr_pos, nb_vertex):
    rv = np.zeros((nb_vertex,), dtype=np.int32)
    for i in range(arr_pos.shape[0]):
        if arr_condition[i]:
            rv[arr_pos[i]] += 1
    return rv


@nb.njit
def transition_initialize_counters_of_newly_infected(arr_new_infected, arr_cnt, arr_new_cnt):
    counter = 0
    for i in range(arr_new_infected.shape[0]):
        if arr_new_infected[i]:
            arr_cnt[i] = arr_new_cnt[counter]
            counter += 1


@nb.njit
def transition_conditional_count_nb_agent_per_vertex(arr_condition, arr_pos, nb_vertex):
    rv = np.zeros((nb_vertex,), dtype=np.int32)
    for i in range(arr_pos.shape[0]):
        if arr_condition[i]:
            rv[arr_pos[i]] += 1
    return rv


@nb.njit
def transition_falsify_when_condition(arr_bool, condition):
    for i in range(arr_bool.shape[0]):
        if condition[i]:
            arr_bool[i] = False

