class ClassificationTrainer(BaseTrainer):
    """
    A trainer class extending BaseTrainer for training image classification models.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: Optional[Dict[str, Any]] = None, _callbacks=None):
        """
        Initialize a ClassificationTrainer object.
        """
        if overrides is None:
            overrides = {}
        overrides["task"] = "classify"
        if overrides.get("imgsz") is None:
            overrides["imgsz"] = 224
        super().__init__(cfg, overrides, _callbacks)

    def get_model(self, cfg=None, weights=None, verbose: bool = True):
        """
        Return a modified PyTorch model configured for training YOLO classification.
        """
        model = ClassificationModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)

        for m in model.modules():
            if not self.args.pretrained and hasattr(m, "reset_parameters"):
                m.reset_parameters()
            if isinstance(m, torch.nn.Dropout) and self.args.dropout:
                m.p = self.args.dropout  # set dropout
        for p in model.parameters():
            p.requires_grad = True  # for training
        return model