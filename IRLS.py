class IRLS:

    def __init__(self, name, variables, X, y) -> None:
        self.name = name
        self.variables = variables
        self.X = X
        self.y = y

    def fit(self, iters: int):
        