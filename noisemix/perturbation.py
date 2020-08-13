class Perturbation:
    name = "default"
    function=None

    def __init__(self, name, function,frequency,repeat_num):
        self.name = name
        self.function = function
        self.frequency = frequency
        self.repeat_num = repeat_num

    def change_frequency(self, frequency):
        self.frequency = frequency

    def change_repeatable(self, repeatable):
        self.isRepeatable = repeatable