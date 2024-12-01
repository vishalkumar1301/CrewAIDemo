class Output1model:
    def __init__(self):
        self.model = None

    def load_model(self):
        self.model = load_model("output1model.h5")
