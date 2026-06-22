"""Epsilon Policy Strategies"""

class EpsilonGreedy:
    """Epsilon-greedy policy"""

    def __init__(self,
                 epsilon_start : float | int = 1.,
                 epsilon_coeff : float | int = 0.999,
                 epsilon_end : float | int = 0.,
                 **kwargs):
        """
        Applies epsilon-greedy scheduling to balance exploration and exploitation during training.

        Parameters
        ----------
        epsilon_start : float | int, optional
            Initial value of epsilon (the exploration rate), by default 1.0
        epsilon_coeff : float | int, optional
            Decay coefficient for epsilon, by default 0.999
        epsilon_end : float | int, optional
            Minimum value of epsilon, by default 0.0
            When reached epsilon will stop decaying and will remain constant at this value.
        """
        self.eps = epsilon_start
        self.coeff = epsilon_coeff
        self.limit = epsilon_end


    def step(self):
        """Updates the value of epsilon according to the decay coefficient and the minimum limit."""
        if self.eps > self.limit:
            self.eps *= self.coeff
        else:
            self.eps = self.limit
        return self.eps



class EpsilonConstant(EpsilonGreedy):
    """Constant epsilon-greedy policy"""

    def __init__(self, **kwargs):
        """
        Initializes a constant epsilon-greedy policy.

        Parameters
        ----------
        **kwargs:
            Additional keyword arguments (not used in this class).
        """
        super().__init__(epsilon_start=kwargs.get("epsilon_start", 1.0),
                         epsilon_coeff=1.0,
                         epsilon_end=kwargs.get("epsilon_end", 0.0))

class Regularizer:
    pass


class ExponentialRegularizer(Regularizer):
    pass


class LinearRegularizer(Regularizer):
    pass


class PropToEncoderLossRegularizer(Regularizer):
    pass