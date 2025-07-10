import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassBalancedFocalLoss(nn.Module):
    """
    Class-Balanced Focal Loss.

    This loss combines Focal Loss with a class balancing term.
    Focal Loss helps to focus training on hard misclassified examples,
    while class balancing addresses data imbalance.

    The formula for Focal Loss is: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    where p_t is the model's estimated probability for the ground truth class.

    The class balancing term uses an effective number of samples: (1 - beta^N_y) / (1 - beta)
    where N_y is the number of samples for class y.

    Reference: "Class-Balanced Loss Based on Effective Number of Samples"
               (Cui et al., CVPR 2019) - for the class balancing part.
               "Focal Loss for Dense Object Detection" (Lin et al., ICCV 2017) - for Focal Loss.

    Parameters:
    -----------
    beta : float, default=0.9999
        Hyperparameter for calculating the effective number of samples.
        Close to 1.0 for large datasets. (0.0 means no re-weighting).
    gamma : float, default=2.0
        Focusing parameter for Focal Loss. Higher gamma gives more weight to hard examples.
    alpha : list or None, default=None
        Alpha weighting factor for each class. If None, it's derived from beta
        and samples_per_class. If a list, it should have one entry per class.
    samples_per_class : list or None, default=None
        Number of samples for each class. Required if alpha is None for class balancing.
        E.g., [1000, 100, 10] for 3 classes.
    reduction : str, default='mean'
        Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
    """
    # Defaults beta=0.9999, gamma=2.0 as per AUROC spec
    def __init__(self, beta=0.9999, gamma=2.0, alpha=None, samples_per_class=None, reduction='mean'):
        super(ClassBalancedFocalLoss, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.alpha_weights = alpha # Can be pre-calculated weights or calculated dynamically
        self.samples_per_class = samples_per_class
        self.reduction = reduction

        if self.alpha_weights is None and self.samples_per_class is None and self.beta > 0:
            # This is a warning, as alpha will default to 1 if samples_per_class is not provided later
            print("Warning: ClassBalancedFocalLoss initialized without alpha or samples_per_class. "
                  "Class balancing will not be effective unless samples_per_class is provided during forward pass "
                  "or alpha is set.")

        if self.alpha_weights is not None and not isinstance(self.alpha_weights, torch.Tensor):
            self.alpha_weights = torch.tensor(self.alpha_weights, dtype=torch.float)


    def forward(self, logits, targets):
        """
        Compute the Class-Balanced Focal Loss.

        Args:
            logits (torch.Tensor): Raw, unnormalized scores from the model (before softmax).
                                   Shape: (N, C), where N is batch size, C is number of classes.
            targets (torch.Tensor): Ground truth labels. Shape: (N,). Integer values from 0 to C-1.

        Returns:
            torch.Tensor: The computed loss value.
        """
        num_classes = logits.shape[1]

        if self.alpha_weights is None and self.samples_per_class is not None:
            # Calculate class balancing weights based on effective number of samples
            if not isinstance(self.samples_per_class, torch.Tensor):
                self.samples_per_class = torch.tensor(self.samples_per_class, dtype=torch.float, device=logits.device)

            effective_num = 1.0 - torch.pow(self.beta, self.samples_per_class)
            weights = (1.0 - self.beta) / effective_num
            weights = weights / torch.sum(weights) * num_classes # Normalize
            cb_alpha = weights
        elif self.alpha_weights is not None:
            if self.alpha_weights.device != logits.device:
                self.alpha_weights = self.alpha_weights.to(logits.device)
            cb_alpha = self.alpha_weights
        else:
            # No class balancing, alpha is effectively 1 for all classes in the focal loss part
            cb_alpha = torch.ones(num_classes, dtype=torch.float, device=logits.device)

        # Calculate Focal Loss components
        # p_t = P(class_true | input)
        # For numerical stability, use log_softmax and then gather.
        log_probs = F.log_softmax(logits, dim=1)
        # Gather the log probabilities of the true classes.
        # targets.unsqueeze(1) makes targets [N,1] to be used with gather
        log_p_t = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1) # Shape: (N,)

        # (1 - p_t)^gamma
        # p_t = exp(log_p_t)
        p_t = torch.exp(log_p_t)
        focal_term = torch.pow(1 - p_t, self.gamma)

        # Get the alpha weights for each sample in the batch based on its target class
        alpha_t = cb_alpha.gather(0, targets) # Shape: (N,)

        # Compute the final loss for each sample
        loss = -alpha_t * focal_term * log_p_t

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise ValueError(f"Unsupported reduction type: {self.reduction}")

# Example Usage
if __name__ == '__main__':
    # Configuration
    N_BATCH = 4  # Batch size
    N_CLASSES = 3 # Number of classes
    SAMPLES_PER_CLASS = [100, 50, 10] # Example: class 2 is minority
    BETA = 0.9999 # As specified
    GAMMA = 2.0   # As specified

    # Dummy data
    # Raw scores (logits) from a model
    dummy_logits = torch.randn(N_BATCH, N_CLASSES, requires_grad=True)
    # Ground truth labels
    dummy_targets = torch.tensor([0, 1, 2, 1]) # Example targets

    print("Dummy Logits:\n", dummy_logits)
    print("Dummy Targets:\n", dummy_targets)

    # --- Test Case 1: Using samples_per_class to derive alpha ---
    print("\n--- Test Case 1: Using samples_per_class ---")
    cb_focal_loss_fn_1 = ClassBalancedFocalLoss(
        beta=BETA,
        gamma=GAMMA,
        samples_per_class=SAMPLES_PER_CLASS,
        reduction='mean'
    )
    loss1 = cb_focal_loss_fn_1(dummy_logits, dummy_targets)
    print(f"Loss (with samples_per_class): {loss1.item()}")
    if loss1.requires_grad:
        loss1.backward()
        print("Gradient computed for loss1.")
        # print("Logits grad:\n", dummy_logits.grad)

    # Reset grads for next test
    if dummy_logits.grad is not None:
        dummy_logits.grad.zero_()

    # --- Test Case 2: Providing explicit alpha weights ---
    # Let's pre-calculate some alpha weights for demonstration
    # (these might not exactly match the ones derived from samples_per_class due to normalization details)
    print("\n--- Test Case 2: Using pre-defined alpha ---")
    explicit_alpha = [0.25, 0.5, 0.75] # Example alpha weights
    cb_focal_loss_fn_2 = ClassBalancedFocalLoss(
        beta=BETA, # beta is not used if alpha is provided directly for class balancing part
        gamma=GAMMA,
        alpha=explicit_alpha,
        reduction='sum'
    )
    loss2 = cb_focal_loss_fn_2(dummy_logits, dummy_targets)
    print(f"Loss (with explicit alpha, sum reduction): {loss2.item()}")
    if loss2.requires_grad:
        loss2.backward()
        print("Gradient computed for loss2.")

    if dummy_logits.grad is not None:
        dummy_logits.grad.zero_()

    # --- Test Case 3: No class balancing (alpha=None, samples_per_class=None, beta=0 or not effective) ---
    # This should behave like standard Focal Loss if alpha is effectively [1,1,1,...]
    print("\n--- Test Case 3: Standard Focal Loss (no explicit class balancing) ---")
    # To achieve standard Focal Loss, we can pass alpha=None and samples_per_class=None,
    # or beta=0. The cb_alpha will default to ones.
    focal_loss_fn = ClassBalancedFocalLoss(
        beta=0.0, # Setting beta=0 means no class re-weighting from effective samples
        gamma=GAMMA,
        samples_per_class=None, # Ensure it doesn't try to calculate weights
        reduction='mean'
    )
    loss3 = focal_loss_fn(dummy_logits, dummy_targets)
    print(f"Loss (standard Focal Loss behavior): {loss3.item()}")
    if loss3.requires_grad:
        loss3.backward()
        print("Gradient computed for loss3.")

    # --- Test Case 4: Check with CrossEntropyLoss for a simple case (gamma=0, no balancing)
    # When gamma=0 and no class balancing (alpha_t = 1), Focal Loss should be similar to CrossEntropyLoss.
    # Note: Our alpha_t is derived from class balancing, so to make it 1, we need beta=0
    # and also ensure the internal alpha calculation doesn't scale things.
    # The standard CE loss is -log(p_t). Our formula is -alpha_t * (1-p_t)^gamma * log(p_t)
    # If alpha_t=1 and gamma=0, it becomes -log(p_t).
    print("\n--- Test Case 4: Comparison with CrossEntropy (gamma=0, no balancing) ---")

    # To make alpha_t effectively 1 for all classes for this test:
    # Set beta=0 (so effective_num calculation doesn't apply)
    # Set alpha=None and samples_per_class=None (so cb_alpha defaults to ones)
    cb_focal_gamma0_nobalance = ClassBalancedFocalLoss(
        beta=0.0, # No class re-weighting
        gamma=0.0, # Makes (1-p_t)^gamma = 1
        alpha=None, samples_per_class=None, # cb_alpha becomes ones
        reduction='mean'
    )
    loss4 = cb_focal_gamma0_nobalance(dummy_logits, dummy_targets)

    # Standard PyTorch CrossEntropyLoss
    ce_loss_fn = nn.CrossEntropyLoss(reduction='mean')
    loss_ce = ce_loss_fn(dummy_logits, dummy_targets)

    print(f"ClassBalancedFocalLoss (gamma=0, no balance): {loss4.item()}")
    print(f"Standard CrossEntropyLoss: {loss_ce.item()}")
    # These should be very close.

    assert torch.isclose(loss4, loss_ce), "Focal Loss with gamma=0 and no balancing should match CrossEntropyLoss"
    print("Assertion passed: Focal Loss (gamma=0, no balance) matches CrossEntropyLoss.")
