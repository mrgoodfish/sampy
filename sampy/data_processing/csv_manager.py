import numpy as np


class ParamManager:
    def __init__(self, names, values):
        for name, value in zip(names, values):
            setattr(self, name, value)


class CsvManager:
    """
    Class used to manage parameters stored in a CSV. This class comes with a buffer, which enable the
    """
    def __init__(self, path_to_csv, sep, dict_types={}, buffer_size=1000, nb_cores=1, id_process=0):
        self.path_to_csv = path_to_csv
        self.sep = sep
        self.dict_types = dict_types

        self.buffer_size = buffer_size
        self.nb_line_consumed = 0
        self.buffer = []
        self.counter_buffer = 0

        self.nb_usable_lines_in_csv = 0

        self.dict_arr = {}
        self.dict_const = {}

        self.nb_cores = nb_cores
        self.id_process = id_process

        with open(self.path_to_csv, 'r') as f_in:
            for i, line in enumerate(f_in):
                if i == 0:
                    self.header = line.replace('\n', '')
                    self.extract_info_header()
                    continue
                if i % self.nb_cores == self.id_process:
                    self.nb_usable_lines_in_csv += 1

    def extract_info_header(self):
        list_header = self.header.split(self.sep)
        dict_col_to_index = {col_name: ind for ind, col_name in enumerate(list_header)}
        r_dict_const = {}
        temp_dict_arr = {}
        for col_name in list_header:
            if col_name.split('_')[0] == 'arr':
                name_param = '_'.join(col_name.split('_')[1:-1])
                try:
                    temp_dict_arr[name_param].append(col_name)
                except KeyError:
                    temp_dict_arr[name_param] = [col_name]
            else:
                r_dict_const[col_name] = dict_col_to_index[col_name]
        r_dict_arr = {}
        for name_arr, arr in temp_dict_arr.items():
            sorted_arr = sorted(arr, key=lambda y: int(y.split('_')[-1]))
            r_dict_arr[name_arr] = [dict_col_to_index[name_col] for name_col in sorted_arr]
        self.dict_arr = r_dict_arr
        self.dict_const = r_dict_const

    def get_parameters(self):
        try:
            line = self.buffer[self.counter_buffer]
            self.counter_buffer += 1
            self.nb_line_consumed += 1
        except IndexError:
            if self.nb_line_consumed == self.nb_usable_lines_in_csv:
                return
            self.fill_buffer()
            line = self.buffer[0]
            self.counter_buffer = 1
            self.nb_line_consumed += 1
        return self.create_param_manager_from_line(line)

    def fill_buffer(self):
        self.buffer = []
        size_current_buffer = 0
        with open(self.path_to_csv) as f:
            seen_lines = 0
            for i, line in enumerate(f):
                if i == 0:
                    continue
                if i % self.nb_cores == self.id_process:
                    seen_lines += 1
                    if seen_lines <= self.nb_line_consumed:
                        continue
                    self.buffer.append(line.replace('\n', ''))
                    size_current_buffer += 1
                    if size_current_buffer == self.buffer_size:
                        break
        return

    def create_param_manager_from_line(self, line):
        data = line.split(self.sep)
        names = []
        values = []
        for name in self.dict_const:
            names.append(name)
            if name in self.dict_types:
                if self.dict_types[name] == bool:
                    values.append(data[self.dict_const[name]].lower() == 'true')
                else:
                    values.append(self.dict_types[name](data[self.dict_const[name]]))
            else:
                values.append(data[self.dict_const[name]])
        for name in self.dict_arr:
            names.append(name)
            if name in self.dict_types:
                if self.dict_types[name] == bool:
                    values.append(np.array([data[u].lower() == 'true' for u in self.dict_arr[name]]))
                else:
                    values.append(np.array([self.dict_types[name](data[u]) for u in self.dict_arr[name]]))
            else:
                values.append(np.array([data[u] for u in self.dict_arr[name]]))
        return ParamManager(names, values)



