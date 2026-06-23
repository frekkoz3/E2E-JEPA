"""Epsilon Policy & Parameters Regularization Strategies"""
import torch


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
    """Base class for Policy Loss regularization strategies."""
    def __init__(self, **kwargs):
        pass

    def step(self, **kwargs):
        """Updates the regularization weight according to the specific regularization strategy."""
        pass

class ExponentialRegularizer(Regularizer):
    """An exponentially increasing regularizer up to a given limit."""

    def __init__(self,
                 reg_weight_start : float | int = 0.001,
                 reg_weight_end : float | int = 1,
                 reg_weight_step : float | int = 1.01):

        assert reg_weight_step > 1, \
            f"Exponential growth step = {reg_weight_step} must be > 1."
        assert reg_weight_start < reg_weight_end, \
            f"Start weight = {reg_weight_start} must be < end weight = {reg_weight_end}."

        self.reg_weight = reg_weight_start
        self.reg_weight_end = reg_weight_end
        self.reg_weight_step = reg_weight_step


    def step(self, **kwargs):
        """Updates the regularization weight according to the exponential growth coefficient and the maximum limit."""
        if self.reg_weight < self.reg_weight_end:
            self.reg_weight = min(self.reg_weight * self.reg_weight_step, self.reg_weight_end)
        return self.reg_weight



class LinearRegularizer(Regularizer):

    def __init__(self,
                 reg_weight_start : float | int = 0.001,
                 reg_weight_end : float | int = 1,
                 reg_weight_step : float | int = 0.001):

        assert reg_weight_step > 0, \
            f"Linear growth step = {reg_weight_step} must be > 0."
        assert reg_weight_start < reg_weight_end, \
            f"Start weight = {reg_weight_start} must be < end weight = {reg_weight_end}."

        self.reg_weight = reg_weight_start
        self.reg_weight_end = reg_weight_end
        self.reg_weight_step = reg_weight_step


    def step(self, **kwargs):
        """Updates the regularization weight according to the linear growth step and the maximum limit."""
        if self.reg_weight < self.reg_weight_end:
            self.reg_weight = min(self.reg_weight + self.reg_weight_step, self.reg_weight_end)
        return self.reg_weight



class PropToOtherLossRegularizer(Regularizer):
    """
    A regularizer that takes the magnitude of the Loss to regularize `LossReg` and another target loss `LossTarget`.
    It compares the magnitudes (log10) and returns a regularization value `ValueReg` so that:

            ValueReg * log10(LossReg) = log10(TargetReg) / Threshold

    Default value for Treshold = 10.
    This allows the LossReg not to overcome the TargetLoss.
    Useful for setting the alpha parameter for the SigReg loss.

    Notes
    -----
    When given a Batch of point-wise losses, ValueReg is computed from the mean loss among the batch
    """

    def __init__(self, reg_threshold : float | int = 10., eps : float | int = 1e-8):
        self.threshold = reg_threshold
        self.eps = eps

    def step(self, loss_reg : torch.Tensor, loss_target : torch.Tensor, **kwargs) -> float | int:
        """Computes the regularization value based on the magnitudes of the regularized loss and the target loss."""
        log_loss_reg = torch.log10(loss_reg + self.eps).mean(dim=0)
        log_loss_target = torch.log10(loss_target + self.eps).mean(dim=0)
        if torch.abs(log_loss_reg) < self.eps:
            return 0.0

        value_reg = log_loss_target / (log_loss_reg * self.threshold)
        return float(value_reg.item())


class PropToOtherLossChangeRegularizer(Regularizer):
    """
    A regularizer that takes  old and new values of a target loss `LossTarget`.
    The regularization value `ValueReg` will be the inverse of the percentage absolute magnitude change between the two values:

            ValueReg = log10( LossTarget(old) / | log10(LossTarget(new)) - log10(LossTarget(old)) |

    This allows the regularized loss to scale inversely w.r.t. LossTarget.
    Useful for scaling the Policy Loss so that it is small when Prediction Loss is high, and viceversa.

    Notes
    -----
    When given a Batch of point-wise losses, ValueReg is computed from the mean loss among the batch.
    Magnitude change is clamped to [1e-6, 100] to avoid extreme values.
    """

    def __init__(self, eps : float | int = 1e-8, max_weight : float | int = 100.0):
        self.eps = eps
        self.max_weight = max_weight
        self.old_loss_target = None

    def step(self, loss_target : torch.Tensor, **kwargs) -> float | int:
        """Computes the regularization value based on the change in magnitude of the target loss."""
        current_log_loss = torch.log10(loss_target.detach() + self.eps).mean()

        if self.old_loss_target is None:
            self.old_loss_target = current_log_loss.item() # Store as float, not tensor
            return 1.0

        change = abs(current_log_loss.item() - self.old_loss_target)
        change = max(change, self.eps)

        numerator = abs(current_log_loss.item())

        value_reg = numerator / change

        self.old_loss_target = current_log_loss.item()

        # 4. Cap the maximum possible weight to prevent destabilizing training
        return min(float(value_reg), self.max_weight)



