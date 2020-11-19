from ruamel.yaml import YAML


class SimpleYamlData():
    def __init__(self, fname=None):
        self.data = {}
        if fname is not None:
            self.read_yaml(fname)

    def read_yaml(self, fname):
        yaml = YAML(typ='safe')
        yfile = open(fname)
        data = yaml.load(yfile)
        self.data = data
        yfile.close()

    def __getitem__(self, key):
        return self.data[key]

    def __repr__(self):
        return str(self.data)


if __name__ == "__main__":
    from importlib import resources

    pd = SimpleYamlData()
    with resources.path("faroes.data", "materials.yaml") as mats:
        pd.read_yaml(mats)
    print(type(pd.data))
