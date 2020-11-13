import numbers
from ruamel.yaml import YAML


def validate_dict_leaf(leaf):
    if 'units' in leaf and 'value' not in leaf:
        raise KeyError("Unit specified without value")
    if 'value' not in leaf:
        raise KeyError("No value specified")


def validateleaf(leaf):
    if isinstance(leaf, numbers.Number):
        return
    if isinstance(leaf, dict):
        validate_dict_leaf(leaf)


def assign(prob, name, leaf):
    validateleaf(leaf)
    if isinstance(leaf, numbers.Number):
        assign_number_leaf(prob, name, leaf)
    if isinstance(leaf, dict):
        assign_dict_leaf(prob, name, leaf)


def assign_number_leaf(prob, name, leaf):
    val = leaf
    prob.set_val(name, val)


def assign_dict_leaf(prob, name, leaf):
    val = leaf['value']
    if 'units' in leaf:
        units = leaf['units']
        prob.set_val(name, val, units)
    else:
        prob.set_val(name, val)


def is_leaf(node):
    """Check whether there are more levels or not"""
    is_num_leaf = isinstance(node, numbers.Number)
    is_dict_leaf = isinstance(node, dict) and 'value' in node
    return is_num_leaf or is_dict_leaf


def walk_dict_tree(prob, name, dict_tree):
    """Recursively walk the dict tree and assign values"""
    for key, val in dict_tree.items():
        if is_leaf(val):
            leaf = val
            assign(prob, name + key, leaf)
        else:
            node_name = name + key + '.'
            walk_dict_tree(prob, node_name, val)


def load_yaml_to_problem(prob, yaml_str):
    yaml = YAML(typ='safe')
    data = yaml.load(yaml_str)
    walk_dict_tree(prob, "", data)


if __name__ == "__main__":
    import openmdao.api as om
    from simple_tf_magnet import MagnetRadialBuild

    yaml = YAML(typ='safe')

    yaml_here_string = """
    R0 : 3
    geometry:
        r_ot: {'value': 405, 'units': 'mm'}
        r_iu: 8.025
    windingpack:
        f_HTS: 0.76
        j_eff_max : {'value' : 160, 'units': 'MA/m**2'}
    """

    prob = om.Problem()

    prob.model = MagnetRadialBuild()

    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'

    prob.model.add_design_var('r_is', lower=0.03, upper=0.4)
    prob.model.add_design_var('r_im', lower=0.05, upper=0.5)
    prob.model.add_design_var('j_HTS', lower=0, upper=300)

    prob.model.add_objective('obj')

    # set constraints
    prob.model.add_constraint('max_stress_con', lower=0)
    prob.model.add_constraint('con2', lower=0)
    prob.model.add_constraint('con3', lower=0)

    prob.setup()

    load_yaml_to_problem(prob, yaml_here_string)

    prob.run_driver()
