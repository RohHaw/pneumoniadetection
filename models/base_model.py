class BaseModel:
    def __init__(self):
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def preprocess(self, image):
        pass

    def to_device(self, tensor):
        return tensor.to(self.device) 