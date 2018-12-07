

class AgentPair:

    def __init__(self, model):
        self.model = model

    def train(self, inputs, labels):
        return self.model.train(inputs, labels)

    def test(self, inputs, labels):
        return self.model.test(inputs, labels)

    def infer(self, inputs):
        raise NotImplementedError

    def train_and_test(self):
        self.data.train, self.data.test = self.model.generate_train_and_test()
        self.train(*self.data.train)
        self.test(*self.data.test)

    def test_all(self):
        raise NotImplementedError

    def get_utterances(self, inputs):
        raise NotImplementedError

    def parse_utternace(self, utt):
        raise NotImplementedError

'''
a model has:
    - encoder_input
    - utterance (e_output/d_input)
    - decoder_output
    - generate_train_and_test

I want AP to mangage the graph and whatnot. THe model will actually build all
of the nodes, but it will only provide the specific nodes that we need for
input, output, utterances, logging.

I guess the model will also have to provide the input space (for now).
'''
