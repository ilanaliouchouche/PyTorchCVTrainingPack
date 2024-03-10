import datetime
from typing import Callable, Dict, List, Optional, Union
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from utils.logger_config import get_logger
from src.models.inception import BinaryInceptionLoss
from tqdm.auto import tqdm

logger = get_logger()

class FakeDetectorTrainer:
    """
    Class for training the FakeDetector model.
    """

    model: nn.Module
    optimizer: torch.optim.Optimizer
    train_loss_fn: torch.nn.modules.loss.BCEWithLogitsLoss
    val_loss_fn: Union[torch.nn.modules.loss.BCEWithLogitsLoss, BinaryInceptionLoss]
    n_labels: int
    train_step: float
    val_step: float
    train_loader: torch.utils.data.DataLoader
    val_loader: Optional[torch.utils.data.DataLoader]
    writer: Optional[SummaryWriter]
    losses: List
    val_losses: List
    total_epochs: int
    gradients: Dict

    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 train_loss_fn: torch.nn.modules.loss.BCEWithLogitsLoss,
                 val_loss_fn: Optional[Union[torch.nn.modules.loss.BCEWithLogitsLoss, BinaryInceptionLoss]]) -> None:
        """
        Constructor for the FakeDetectorTrainer class.

        Args:
            model: FakeDetector model
            optimizer: Optimizer
            train_loss_fn: Loss function for training
            val_loss_fn: Loss function for validation
        """

        self.model = model
        self.optimizer = optimizer
        self.train_loss_fn = train_loss_fn
        self.val_loss_fn = val_loss_fn
        self.n_labels = 1
        self.train_step = self._make_train_step()
        self.val_step = self._make_val_step()

        # placeholders
        self.train_loader = None
        self.val_loader = None
        self.writer = None
        self.handles = {}
        self.losses = []
        self.val_losses = []
        self.total_epochs = 0
        self.gradients = {}

    def to(self,
           device: str) -> None:
        """
        Set the device for training.

        Args:
            device: Device for training
        """
        
        try:
            self.device = device
            self.model.to(device)
        except RuntimeError:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.warning(f"Device {device} not found. Using {self.device} instead.")
            self.model.to(self.device)
    
    @staticmethod
    def set_seed(seed: int = 42) -> None:
        """
        Set the seed for reproducibility.

        Args:
            seed: Seed
        """

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    def set_loaders(self,
                    train_loader: torch.utils.data.DataLoader,
                    val_loader: Optional[torch.utils.data.DataLoader] = None) -> None:
        """
        Set the train and validation loaders.

        Args:
            train_loader: Train loader
            val_loader: Validation loader
        """

        self.train_loader = train_loader
        self.val_loader = val_loader

        if self.val_loader is not None and self.val_loss_fn is None:
            raise ValueError("Validation loss function not found.")
        
        if self.val_loader is None and self.val_loss_fn is not None:
            logger.warning("Validation loader not found. Validation loss function will not be used.")
        
        if len(train_loader.dataset.dataset.classes) != self.n_labels + 1:
            raise ValueError(f"Number of labels in the dataset ({len(train_loader.dataset.dataset.classes)})"
                             f"does not match the number of labels in the model ({self.n_labels + 1}).")
          
    def set_tensorboard(self,
                        name: str,
                        log_dir: str = "runs") -> None:
        """
        Set the tensorboard.

        Args:
            name: Name of the tensorboard
            log_dir: Directory for the tensorboard
        """

        suffix = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.writer = SummaryWriter(log_dir=f"{log_dir}/{name}_{suffix}")

    
    def _make_train_step(self) -> float:
        """
        Make a train step.

        Returns:
            function: Train step function
        """

        def train_step(x: torch.Tensor, 
                       y: torch.Tensor) -> float:
            """
            Train step.

            Args:
                x: Input data
                y: Target data

            Returns:
                float: Loss
            """

            self.model.train()
            self.optimizer.zero_grad()
            yhat = self.model(x)
            loss = self.train_loss_fn(yhat, y)
            loss.backward()
            self.optimizer.step()
            return loss.item()

        return train_step
    
    def _make_val_step(self) -> float:
        """
        Make a validation step.

        Returns:
            function: Validation step function
        """

        def val_step(x: torch.Tensor, 
                     y: torch.Tensor) -> float:
            """
            Validation step.

            Args:
                x: Input data
                y: Target data

            Returns:
                float: Loss
            """

            self.model.eval()
            with torch.no_grad():
                yhat = self.model(x)
                loss = self.val_loss_fn(yhat, y)
                return loss.item()

        return val_step

    def _mini_batch(self,
                    validation: bool = False) -> float:
        """
        Loop over the mini batches and compute the loss.

        Args:
            validation: If the mini batch is for validation

        Returns:
            float: Loss of the mini batches
        """

        if validation:
            data_loader = self.val_loader
            step = self.val_step

        else:
            data_loader = self.train_loader
            step = self.train_step

        if data_loader is None:
            return None
        
        mini_batch_loss = 0.0
        for x, y in data_loader:
            x, y = x.to(self.device), y.to(self.device)

            mini_batch_loss += step(x, y)
        
        return mini_batch_loss / len(data_loader)
    
    def train(self,
              n_epochs: int,
              seed: int = 42) -> None:
        """
        Train the model.

        Args:
            n_epochs: Number of epochs
            seed: Seed for reproducibility
        """

        self.set_seed(seed)
        
        logger.info(f"Training for {n_epochs} epochs. {self.total_epochs} epochs have already been trained.")

        for epoch in tqdm(range(n_epochs)):
            self.total_epochs += 1

            train_loss = self._mini_batch()
            self.losses.append(train_loss)

            with torch.inference_mode():
                val_loss = self._mini_batch(validation=True)
                self.val_losses.append(val_loss)

            if self.writer:
                scalars = {"loss/train": train_loss}
                if val_loss:
                    scalars["loss/val"] = val_loss
                self.writer.add_scalars("loss", scalars, epoch)
        
        logger.info(f"Training completed. {n_epochs} epochs trained. Total epochs trained: {self.total_epochs}")

        if self.writer:
            self.writer.flush()

    def save_model(self,
                   path: str) -> None:
        """
        Save the model.
    
        Args:
            path: Path to save the model
        """
    
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_epochs": self.total_epochs,
            "losses": self.losses,
            "val_losses": self.val_losses,
            "date": datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        }

        torch.save(checkpoint, path)
    

    def load_model(self,
                   path: str,
                   eval_mode: bool = False) -> None:
        """
        Load the model.
    
        Args:
            path: Path to load the model
        """

        checkpoint = torch.load(path)

        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.total_epochs = checkpoint["total_epochs"]
        self.losses = checkpoint["losses"]
        self.val_losses = checkpoint["val_losses"]

        logger.info(f"Model loaded from {path}. Model last training date: {checkpoint['date']}")

        if eval_mode:
            self.model.eval()
        else:
            self.model.train()
        
    
    def predict(self,
                x: torch.Tensor) -> torch.Tensor:
        """
        Predict the output of the model.
    
        Args:
            x: Input data
    
        Returns:
            torch.Tensor: Output of the model
        """

        logger.info("Predicting the output of the model.")
        self.model.eval()
        with torch.inference_mode():
            x = x.to(self.device)
            logger.info("Prediction completed.")
            return self.model(x)
        
    def add_graph(self) -> None:
        """
        Add the graph to tensorboard.
        """
        
        logger.info("Adding graph to tensorboard.")
        if self.train_loader is None:
            logger.warning("Train loader not found. Cannot add graph to tensorboard.")
            return
        
        if self.writer:
            logger.info("Graph added to tensorboard.")
            self.writer.add_graph(self.model, next(iter(self.train_loader))[0].to(self.device))

    def hook_gradients(self,
                       layers: List[str]) -> None:
        """
        Hook the gradients of the specified layers.

        Args:
            layers: List of layers to hook
        """

        modules = list(self.model.named_modules())

        def make_log_fn(name: str, 
                        id: str) -> Callable[[torch.Tensor], None]:
            """
            Make a log function.

            Args:
                name: Name of the layer
                id: ID of the parameter

            Returns:
                Callable: Log function
            """

            def log_fn(grad: torch.Tensor) -> None:
                """
                Log the gradient.
                
                Args:
                    grad: Gradient
                """

                self.gradients[name][id].append(grad)

            return log_fn
        
        for name, module in modules:
            if name in layers:
                self.gradients[name] = {}
                for id, param in module.named_parameters():
                    self.gradients[name][id] = []
                    log_fn = make_log_fn(name, id)
                    self.handles[f"{name}_{id}_grad"] = param.register_hook(log_fn)
    
    def unhook(self) -> None:
        """
        Unhook hook functions.
        """

        for handle in self.handles.values():
            handle.remove()
        self.handles = {}
        self.gradients = {}

