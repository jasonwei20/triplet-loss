
class Format:

    def skip(self, line):
        False

    def before(self, line):
        return s, None

    def after(self, line, data):
        return s

class fastText(Format):

    LABEL_PREFIX = '__label__'

    def before(self, line):
        labels = []
        _ = []
        for w in line.split(' '):
            if w.startswith(fastText.LABEL_PREFIX):
                labels.append(w)
            else:
                _.append(w)
        return ' '.join(_), labels

    def after(self, line, data):
        return ' '.join(data) + ' ' + line
