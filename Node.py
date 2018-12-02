class Node:
    def __init__(self, parent=None):
        self.parent = parent
        self.children = []
        self.label = None
        self.split_feature_value = None
        self.split_feature = None
