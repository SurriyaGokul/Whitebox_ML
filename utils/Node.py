class Node:
    def __init__(
        self,
        feature = None,
        split_value = None,
        node_type = None,
        impurity = None,
        prediction = False,
        left=None,
        right=None
    ):
        self.feature = feature
        self.split_value = split_value
        self.node_type = node_type
        self.impurity = impurity
        self.left = left
        self.prediction = prediction
        self.right = right
