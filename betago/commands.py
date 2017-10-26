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
        self.parser.add_argument("arguments", nargs=argparse.REMAINDER, help="Arguments")
        pargs = self.parser.parse_args(args.arguments)
        self.method(cfg, pargs)

class argument:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
    
    def __call__(self, command):
        command.arguments.insert(0, (self.args, self.kwargs))
        return command

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

