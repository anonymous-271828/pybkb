class InternalBKBError(Exception):
    """
    Base exception for errors internal to the BKB processing.
    
    :param message: Error message to be displayed.
    :type message: str
    """
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return self.message


class INodeInBKBError(Exception):
    """
    Exception raised when an I-node to be added already exists in the BKB.
    
    :param component_name: Name of the I-node component already in the BKB.
    :param state_name: Name of the I-node state already in the BKB.
    :param message: Optional custom error message.
    """
    def __init__(self, component_name, state_name, message="I-node exists in BKB."):
        self.component_name = component_name
        self.state_name = state_name
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{(self.component_name, self.state_name)}: {self.message}'


class NoINodeError(Exception):
    """
    Exception raised when a specified I-node does not exist in the BKB.
    
    :param component_name: Name of the non-existent I-node component.
    :param state_name: Name of the non-existent I-node state.
    :param message: Optional custom error message.
    """
    def __init__(self, component_name, state_name, message="I-node does not exist in BKB."):
        self.component_name = component_name
        self.state_name = state_name
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{(self.component_name, self.state_name)}: {self.message}'


class SNodeProbError(Exception):
    """
    Exception raised for invalid S-node probability values.
    
    :param prob: The S-node probability value causing the error.
    """
    def __init__(self, prob, message="S-node probability not between 0 and 1."):
        self.prob = prob
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'Invalid S-node probability {self.prob}: {self.message}'


class BKBNotMutexError(Exception):
    """
    Exception raised when a BKB fails to meet mutual exclusivity constraints.
    
    :param snode_idx1: Index of the first non-mutex S-node.
    :param snode_idx2: Index of the second non-mutex S-node.
    :param message: Optional custom error message.
    """
    def __init__(self, snode_idx1, snode_idx2, message="BKB is not mutex."):
        self.snode_idx1 = snode_idx1
        self.snode_idx2 = snode_idx2
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'S-node with index {self.snode_idx1} is not mutex with S-node with index {self.snode_idx2}: {self.message}'


class InvalidProbabilityError(Exception):
    """
    Exception raised for inconsistencies in joint probability calculations.
    
    :param p_xp: Probability value for p(x, π(x)).
    :param p_p: Optional probability value for p(π(x)).
    :param p_x: Optional probability value for p(x).
    :param message: Optional custom error message.
    """
    def __init__(self, p_xp, p_p=None, p_x=None, message="Invalid joint probabilities."):
        self.p_xp = p_xp
        self.p_x = p_x
        self.p_p = p_p
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        s = f'Impossible for p(x, π(x)) = {self.p_xp}'
        if self.p_x is not None:
            s += f' when p(x) = {self.p_x}'
        if self.p_p is not None:
            s += f' and p(π(x)) = {self.p_p}'
        s += '.'
        return s
