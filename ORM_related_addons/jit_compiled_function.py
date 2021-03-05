import numpy as np
import numba as nb


@nb.njit
def orm_like_agent_orm_dispersion(arr_can_move, arr_will_move, arr_rand_nb_steps, arr_rand_direction,
                                  col_position, col_territory, col_has_moved, connections):
    counter_arr_will_move = 0
    counter_arr_rand_nb_steps = 0
    counter_arr_rand_direction = 0
    for i in range(arr_can_move.shape[0]):
        if arr_can_move[i]:

            # first we determine if the agent is allowed to move. If not, we skip to the next agent
            will_move = arr_will_move[counter_arr_will_move]
            counter_arr_will_move += 1
            if not will_move:
                continue

            # now we update the 'has moved' flag (in ORM, this is done even if the agent moves 0 cells)
            col_has_moved[i] = True

            nb_steps = arr_rand_nb_steps[counter_arr_rand_nb_steps]
            counter_arr_rand_nb_steps += 1
            if nb_steps == 0:
                continue

            for j in range(nb_steps):
                rand_direction = arr_rand_direction[counter_arr_rand_direction]
                counter_arr_rand_direction += 1

                if j == 0:
                    direction = int(np.floor(rand_direction * 6)) % 6
                else:
                    if rand_direction <= 0.2:
                        direction = (direction - 1) % 6
                    elif rand_direction >= 0.8:
                        direction = (direction + 1) % 6

                if connections[col_territory[i]][direction] == -1:
                    break

                col_position[i] = connections[col_territory[i]][direction]
                col_territory[i] = connections[col_territory[i]][direction]


@nb.njit
def orm_like_agents_mortality_from_v08_no_condition_no_alpha_beta(arr_count, arr_condition_count, arr_position, arr_k,
                                                                  arr_annual_mortality, arr_age, arr_rand):
    rv = np.full(arr_age.shape, True, dtype=np.bool_)
    for i in range(rv.shape[0]):
        if arr_k[arr_position[i]] == 0.:
            rv[i] = False
            if arr_condition_count[i]:
                arr_count[arr_position[i]] -= 1
            continue

        mort_adjust = float(arr_count[arr_position[i]]) / float(arr_k[arr_position[i]])
        age_agent = arr_age[i] // 52
        p0 = 1 - np.exp(-arr_annual_mortality[age_agent])
        alpha = 4. * 1.5 / p0
        if arr_rand[i] <= (p0 / (1 + np.exp(-(mort_adjust - 1.5)*alpha))):
            rv[i] = False
            if arr_condition_count[i]:
                arr_count[arr_position[i]] -= 1
    return rv
