import openmdao.utils.units as units


def add_local_units():
    units.add_unit('n19', '10**19/m**3')
    units.add_unit('n20', '10**20/m**3')
