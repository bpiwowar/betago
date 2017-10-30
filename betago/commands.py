import logging
import argparse

commands = {}

class Command:
    def __init__(self, method, description=None):
        self.method = method
        self.name = method.__name__ 
        self.parser = argparse.ArgumentParser(self.name, description=description)
        self.arguments = []

    def __repr__(self):
        return "Command(%s)" % (self.name)

    def __call__(self, cfg, args):
        logging.debug("Parsing remaining arguments: %s", args.arguments)
        for posargs, kwargs in self.arguments:
            self.parser.add_argument(*posargs, **kwargs)
        self.parser.add_argument("arguments", nargs=argparse.REMAINDER, help="Subcommand arguments")
        pargs = self.parser.parse_args(args.arguments)
        self.method(cfg, pargs)


class command:
    def __init__(self, description=None):
        self.description = description

    def __call__(self, m):
        c = Command(m, description=self.description)
        commands[c.name.replace("_", "-")] = c
        return c


class subcommand:
    def __init__(self, description=None):
        self.description = description

    def __call__(self, m):
        c = Command(m, description=self.description)
        return c

class argument:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
    
    def __call__(self, obj):
        if isinstance(obj, Command):
            arguments = obj.arguments
        else:
            arguments = obj.__dict__.get("__arguments__", None)
            if arguments is None:
                arguments = []
                setattr(obj, "__arguments__", arguments)

        arguments.insert(0, (self.args, self.kwargs))

        return obj

def configure(object, args):
    # Get all arguments
    import argparse
    parser = argparse.ArgumentParser("Configuration of %s" % object.__class__)
    for c in object.__class__.__mro__:
        arguments = getattr(c, "__arguments__", {})
        for posargs, kwargs in arguments:
            parser.add_argument(*posargs, **kwargs)
    r = parser.parse_args(args)

    object.__parameters__ = []
    for key, value in r.__dict__.items():
        setattr(object, key, value)
        object.__parameters__.append(key)
