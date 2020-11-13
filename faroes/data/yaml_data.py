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


if __name__ == "__main__":

    pd = SimpleYamlData()
    pd.read_yaml("materials.yaml")
    #   print(pd["lead"]["bulk cost"])
    pd = SimpleYamlData("magnet_geometry.yaml")
    print(pd["inter-block clearance"])
