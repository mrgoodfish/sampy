class ParamManager:
    def __init__(self, header, line_values, dict_type, sep):
        names = header.split(sep)
        values = line_values.split(sep)

        for name, value in zip(names, values):
            if name in dict_type:
                setattr(self, name, dict_type[name](value))
            else:
                setattr(self, name, value)


class CsvManager:
    """
    Class used to manage parameters stored in a CSV. This class comes with a buffer, which enable the
    """
    def __init__(self, path_to_csv, sep, dict_types={}, buffer_size=1000):
        self.path_to_csv = path_to_csv
        self.sep = sep
        self.dict_types = dict_types

        self.buffer_size = buffer_size
        self.nb_buffer_used = 0
        self.buffer = []
        self.counter_buffer = 0

        self.nb_line_in_csv = 0

        with open(self.path_to_csv, 'r') as f_in:
            for i, line in enumerate(f_in):
                if i == 0:
                    self.header = line.replace('\n', '')
                    continue
                self.nb_line_in_csv += 1

    def extract_info_header(self, header, sep):
        list_header = header.split(sep)
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
                r_dict_const[name_param] = dict_col_to_index[col_name]
        r_dict_arr = {}
        for name_arr, arr in temp_dict_arr.items():
            sorted_arr = sorted(arr, key=lambda y: int(y.split('_')[-1]))
            r_dict_arr[name_arr] = [dict_col_to_index[name_col] for name_col in sorted_arr]
        return r_dict_arr, r_dict_const



    def fill_buffer(self, nb_buffer_filled):
        self.buffer = []
        with open(self.path_to_csv, 'r') as f_in:
            for i, line in enumerate(f_in):
                if i < nb_buffer_filled*self.buffer_size + 1:
                    continue
                self.buffer.append(ParamManager(self.header, line.replace('\n', ''), self.dict_types, self.sep))
                if i == (nb_buffer_filled+1)*self.buffer_size:
                    break

    def get_param(self):
        if self.nb_param_requested == self.nb_line_in_csv:
            return None
        if self.nb_param_requested % self.buffer_size == 0:
            self.fill_buffer(self.nb_param_requested // self.buffer_size)
        self.nb_param_requested += 1
        return self.buffer[(self.nb_param_requested - 1) % self.buffer_size]


class CsvManagerMultiThread:
    def __init__(self, path_to_csv, list_in_queue, list_out_queue, dict_types={}, max_nb_line=None):
        """
        todo
        :param path_to_csv:
        :param dict_types:
        :param max_nb_line:
        """
        self.path = path_to_csv
        self.dict_types = dict_types
        self.max_nb_lines = max_nb_line
