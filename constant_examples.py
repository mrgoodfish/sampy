import numpy as np


# just writing the death prob array here, in order to keep the example scripts light
ARR_DEATH = []
for i in range(52):
    ARR_DEATH.append(0.6 / 52)
for i in range(52):
    ARR_DEATH.append(0.3 / 52)
for i in range(52):
    ARR_DEATH.append(0.6 / 52)
for i in range(52):
    ARR_DEATH.append(0.3 / 52)
for i in range(52):
    ARR_DEATH.append(0.3 / 52)
for i in range(52):
    ARR_DEATH.append(0.99 / 52)
for i in range(52):
    ARR_DEATH.append(0.99 / 52)
ARR_PROB_DEATH_MALE = np.array(ARR_DEATH)
ARR_PROB_DEATH_FEMALE = np.array(ARR_DEATH)
