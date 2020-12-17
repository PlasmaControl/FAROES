from faroes.yaml_data import SimpleYamlData
from openmdao.utils import units as u
from importlib import resources
from copy import deepcopy
import numbers


class Accessor():
    def __init__(self, config=None):
        self.config = config

    def accessor(self, pre_path):
        """Helper accessor"""
        if self.config is not None:

            def f(post_path, units=None):
                return self.config.get_value(pre_path + post_path, units=units)

            return f
        else:
            return None

    def _wrap_if_str(self, name):
        if isinstance(name, str):
            name = [name]
        return name

    def set_output(self,
                   component,
                   accessor,
                   name,
                   component_name=None,
                   units=None):
        config_name = self._wrap_if_str(name)

        if component_name is None:
            component_name = config_name[-1]

        if accessor is not None:
            val = accessor(config_name, units=units)
            component.add_output(component_name, val, units=units)
        else:
            component.add_output(component_name, units=units)


class UserConfigurator():
    default_str = "default"
    ignore_units = 6666666

    def __init__(self, user_data_file=None):
        """Load all default configuration files
        """
        f_extension = '.yaml'
        files = [
            "materials", "magnet_geometry", "fits", "radial_build", "machine",
            "plasma",
        ]

        default_data_dir = "faroes.data"

        self.default_data = {}

        for f in files:
            resource_name = f + f_extension

            if resources.is_resource(default_data_dir, resource_name):
                with resources.path(default_data_dir, resource_name) as mats:
                    resource_data = SimpleYamlData(mats)
                    # stitch dicts together
                    self.default_data[f] = resource_data.data
            else:
                raise FileNotFoundError(resource_name +
                                        " not found in data directory")

        # validate default tree
        excl = ["references"]

        def validator(x):
            self._validate_entry(x,
                                 units=self.ignore_units,
                                 string_acceptable=True)

        self._walk_yaml_tree(self.default_data,
                             leaf_fun=validator,
                             exclude_keys=excl)

        self.data = self.default_data

        if user_data_file is not None:
            self.update_configuration(user_data_file)

    def update_configuration(self, filename, exclude_keys=[]):
        ud = SimpleYamlData(filename)
        updated_data = self._fold_in_new_data(self.data, ud.data, exclude_keys)
        self.data = updated_data

    def accessor(self, pre_path):
        """Helper accessor"""
        def f(post_path, units=None):
            return self.get_value(pre_path + post_path, units=units)

        return f

    def get_value(self, path, units=None):
        """Retrieve a value, ensuring the correct units
        """
        if isinstance(path, str):
            return self.get_value((path, ), units=units)

        if units is not None:
            self._validate_unit(units)

        # multi-key tuple
        entry = self.data
        for key in path:
            entry = entry[key]

        self._validate_entry(entry, units=units, string_acceptable=True)
        value = self._convert_or_pass_through(entry, desired_units=units)
        return value

    def _fold_in_new_data(self, old_data, new_data, exclude_keys):
        """Folds in a subset of the user data.
        """
        dd = deepcopy(old_data)
        self._walk_two_yaml_trees(dd, new_data, exclude_keys=exclude_keys)
        return dd

    def _cross_validate_with_list(self, n1, n2):
        """Check compatibility of n2 with the list n1
        """
        if isinstance(n2, str) and n2.lower() == self.default_str:
            return
        if isinstance(n2, list):
            if len(n1) != len(n2):
                raise AttributeError(f"Lists {n1}, {n2} of unequal length")
            else:
                return
        err_str = f"""Error: first options tree provides a list,
        while second options tree does not.
        First tree's list: {n1}
        Second tree's bad entry: {n2}
        """
        raise AttributeError(err_str)

    def _cross_validate_with_dict(self, n1, n2, exclude_keys=[]):
        if isinstance(n2, str) and n2.lower() == self.default_str:
            return
        if isinstance(n2, dict):
            n1keyset = set(n1.keys())
            n2keyset = set(n2.keys())
            bads = set.difference(n2keyset, n1keyset, set(exclude_keys))
            if len(bads) > 0:
                err_str = f"""Error: key(s) {bads} found in tree
                {n2} which are not present in
                {n1}"""
                raise KeyError(err_str)
            return
        err_str = f"""Error: first options tree provides a dict,
        while second options tree does not.
        First tree's dict: {n1}
        Second tree's bad entry: {n2}
        """
        raise AttributeError(err_str)

    def _cross_validate_leaf_types(self, n1, n2):
        """
        """
        self._validate_entry(n2,
                             units=self.ignore_units,
                             string_acceptable=True)
        if isinstance(n2, str) and n2.lower() == self.default_str:
            # this is always okay
            return
        if isinstance(n1, dict):
            if isinstance(n2, dict):
                self._cross_validate_leaf_dicts(n1, n2)
            else:
                err_string = f"""Error: Entry {n2} is not a dict entry."""
                raise TypeError(err_string)
        elif isinstance(n1, numbers.Number):
            if isinstance(n2, numbers.Number):
                return
            else:
                err_string = f"""Error: Entry {n2} is not a number."""
                raise TypeError(err_string)
        elif isinstance(n1, str):
            if isinstance(n2, str):
                return
            else:
                err_string = f"""Error: Entry {n2} is not a string."""
                raise TypeError(err_string)
        else:
            raise TypeError("Unhandled leaf type " + str(type(n1)))

    def _cross_validate_leaf_dicts(self, n1, n2):
        """Both are dicts
        """
        units = None
        if 'units' in n1.keys():
            units = n1['units']
        self._validate_dict_leaf(n2, units=units)

    def _handle_leaf_matching(self, n1, n2, key):
        self._cross_validate_leaf_types(n1[key], n2[key])
        if isinstance(n2[key], str) and n2[key].lower() == self.default_str:
            return
        elif isinstance(n1[key], str) or isinstance(n1[key], numbers.Number):
            # optional logging goes here
            n1[key] = n2[key]
            return
        elif isinstance(n1[key], dict):
            # handle case like
            # {value: 1.0, units:m} X
            # {value:default, units:mm} ->
            # {value:1000.0, units:mm}
            if 'units' in n2[key] and \
                  isinstance(n2[key]['value'], str) and \
                  n2[key]['value'].lower() == self.default_str and \
                  n2[key]['units'] != self.default_str:
                if n1[key]['units'] != n2[key]['units']:
                    new_val = self._unit_conversion(n1[key]['value'],
                                                    n1[key]['units'],
                                                    n2[key]['units'])
                    n1[key]['value'] = new_val
                    n1[key]['units'] = n2[key]['units']
            for k, v in n1[key].items():
                if k in n2[key]:
                    if not (isinstance(n2[key][k], str)
                            and n2[key][k].lower() == self.default_str):
                        n1[key][k] = n2[key][k]

        else:
            raise TypeError("Unhandled leaf type " + str(type(n1[key])))

    def _walk_two_yaml_trees(self, n1, n2, exclude_keys=[]):
        if isinstance(n1, list):
            self._cross_validate_with_list(n1, n2)
            if isinstance(n2, str) and n2.lower() == self.default_str:
                pass
            if isinstance(n2, list):
                for i in range(len(n1)):
                    if self._is_leaf(n1[i]):
                        self._handle_leaf_matching(n1, n2, i)
                    else:
                        self._walk_two_yaml_trees(n1[i], n2[i], exclude_keys)
        elif isinstance(n1, dict):
            self._cross_validate_with_dict(n1, n2, exclude_keys)
            if isinstance(n2, str) and n2.lower() == self.default_str:
                pass
            if isinstance(n2, dict):
                for key, val in n1.items():
                    if key in n2 and key not in exclude_keys:
                        if self._is_leaf(n1[key]):
                            self._handle_leaf_matching(n1, n2, key)
                        else:
                            self._walk_two_yaml_trees(n1[key], n2[key],
                                                      exclude_keys)
        else:
            raise TypeError("Error: entries should dict, list, \
                    or number [this error should never happen.]")

    def _walk_yaml_tree(self, node, leaf_fun=None, exclude_keys=None):
        """Recursively walk the tree and execute leaf_function for each leaf

        parameters
        ----------
        node : [dict, list, number]
            node of the tree to start from
        leaf_fun : function
            executed for each leaf. return value is not captured.
        exclude_keys : list
            does not explore any keys in the list.

        returns
        -------
        nothing

        raises
        ------
        NotImplementedError
        """
        if self._is_leaf(node):
            if leaf_fun is not None:
                leaf_fun(node)
            return
        elif isinstance(node, list):
            for subnode in node:
                self._walk_yaml_tree(subnode,
                                     leaf_fun=leaf_fun,
                                     exclude_keys=exclude_keys)
            return
        elif isinstance(node, dict):
            for key, subnode in node.items():
                if exclude_keys is not None and key in exclude_keys:
                    pass
                else:
                    self._walk_yaml_tree(subnode,
                                         leaf_fun=leaf_fun,
                                         exclude_keys=exclude_keys)
            return
        else:
            raise NotImplementedError("Unrecognized object in yaml tree.")

    def _is_leaf(self, node):
        """Check whether there are more levels or not"""
        is_num_leaf = isinstance(node, numbers.Number)
        is_dict_leaf = isinstance(node, dict) and 'value' in node
        is_string_leaf = isinstance(node, str)
        return is_num_leaf or is_dict_leaf or is_string_leaf

    def _validate_entry(self, entry, units=None, string_acceptable=False):
        """Entry is either a number or a dict with
        {"value": ... "units": ...}
        or a string which is "default"
        """
        if isinstance(entry, numbers.Number):
            self._validate_number(entry, units)
        elif isinstance(entry, dict):
            self._validate_dict_leaf(entry, units)
        elif isinstance(entry, str):
            self._validate_string_leaf(entry, string_acceptable)
        else:
            raise TypeError(f"Error: unacceptable entry: {entry}.")

    def _validate_string_leaf(self, entry, string_acceptable=False):
        if isinstance(entry, str) and entry.lower() == self.default_str:
            return
        elif not string_acceptable:
            err_str = f"""Error: {entry} is not an acceptable value.
            The default value is '{self.default_str}'."""
            raise AttributeError(err_str)
        else:
            return

    def _validate_number(self, entry, units=None):
        if units is not None and units is not self.ignore_units:
            raise AttributeError(
                "A unit " + str(units) +
                " is specified; but only a number was provided.")

    def _validate_dict_leaf(self, leaf, units=None):
        if 'units' in leaf and 'value' not in leaf:
            raise KeyError("Unit specified without value.")
        if 'value' not in leaf:
            raise KeyError("No value specified.")
        if 'units' in leaf and units is None:
            err_string = "Extra key 'units' found in " + str(leaf) + "."
            raise KeyError(err_string)
        if 'units' not in leaf and (units is not None
                                    and units is not self.ignore_units):
            err_string = "No key+val 'units'='" + str(
                units) + "' (or compatible) found in " + str(leaf) + "."
            raise KeyError(err_string)
        if 'units' in leaf and units is not None:
            self._validate_unit(leaf['units'])
            # ensure compatibility
            if units != self.ignore_units and not u.is_compatible(
                    units, leaf['units']):
                err_string = "Units " + str(
                    leaf['units']) + " and " + units + " are incompatible."
                raise AttributeError(err_string)
        lv = leaf['value']
        if isinstance(lv, numbers.Number):
            pass
        if isinstance(lv, str):
            if lv.lower() != self.default_str:
                raise AttributeError(
                    str(leaf) + " is not an acceptable \
                leaf value. The default value is '" + self.default_str + "'.")

    def _validate_unit(self, units):
        if not u.valid_units(units):
            raise KeyError("Invalid physical unit specification string: " +
                           str(units))

    def _unit_conversion(self, val, u1, u2):
        """
        Raises
        ------
        TypeError if units are not compatible
        """
        conv = u.unit_conversion(u1, u2)
        return conv[0] * (val + conv[1])

    def _convert_or_pass_through(self, entry, desired_units=None):
        if isinstance(entry, dict):
            # can assume that it's valid, that units are compatible
            if desired_units is not None:
                if entry["units"] == desired_units:
                    return entry["value"]
                else:
                    return self._unit_conversion(entry['value'],
                                                 entry['units'], desired_units)
            else:
                return entry["value"]
        else:
            return entry


if __name__ == "__main__":
    uc = UserConfigurator()
