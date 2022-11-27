import logging


class Pipeline(object):
    def __init__(self, name, source=None):
        self.source = source
        self.timer = 0
        self.name = name

    def __iter__(self):
        return self.generator()

    def generator(self):
        while self.has_next():
            try:
                data = next(self.source) if self.source else {}
                if self.filter(data):
                    yield self.map(data)
            except StopIteration:
                return

    def __or__(self, other):
        """
        this pipe operator allows to connect the pipeline tasks
        """
        if other is not None:
            other.source = self.generator()
            return other
        else:
            return self

    def filter(self, data):
        """override this to filter out the pipeline data."""
        return True

    def map(self, data):
        """override to map the pipeline data."""
        return data

    def has_next(self):
        """override to stop the generator in certain conditions."""
        return True

    def cleanup(self):
        """override to perform cleanup tasks."""
        logging.debug("time spent on the stage %s: %.3f [s]" % (self.name, self.timer))
        return
