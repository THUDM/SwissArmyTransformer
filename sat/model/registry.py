class Registry:
    def __init__(self, name):
        self.name = name
        self.member = {}

    def register(self, cls):
        if type(cls) is str:
            def func(f):
                self.member[cls] = f
                return f
            return func
        self.member[cls.__name__] = cls
        return cls
    
    def get(self, name):
        if name not in self.member:
            raise ValueError(f'model_class {name} not found.')
        return self.member[name]
    
    def __repr__(self):
        return 'Registry: ' + self.name + " " + str(self.member)

model_registry = Registry('sat_models')

class MetaModel(type):
    def __new__(cls, clsname, bases, attrs):
        newclass = super().__new__(cls, clsname, bases, attrs)
        model_registry.register(newclass)
        return newclass