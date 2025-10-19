"""
Abstract logging tool supporting SwanLab and TensorBoard
提供统一的日志记录接口，支持 SwanLab 和 TensorBoard
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pathlib import Path
from loguru import logger


class LoggerBackend(ABC):
    """Abstract base class for logging backends"""
    
    @abstractmethod
    def init(self, config: Dict[str, Any]) -> None:
        """Initialize the logging backend"""
        pass
    
    @abstractmethod
    def log(self, metrics: Dict[str, Any]) -> None:
        """Log metrics"""
        pass
    
    @abstractmethod
    def finish(self) -> None:
        """Finish logging and cleanup resources"""
        pass


class SwanLabBackend(LoggerBackend):
    """SwanLab logging backend"""
    
    def __init__(self):
        self.available = False
        self.enabled = False
        try:
            import swanlab
            self.swanlab = swanlab
            self.available = True
        except ImportError:
            logger.warning("SwanLab is not installed. To use it, run: pip install swanlab")
    
    def init(self, config: Dict[str, Any]) -> None:
        """Initialize SwanLab with project and config"""
        if not self.available:
            logger.warning("SwanLab is not available, skipping initialization")
            return
        
        try:
            project = config.pop("project", "opt4")
            workspace = config.pop("workspace", "Sail2Dream")
            
            self.swanlab.init(
                project=project,
                workspace=workspace,
                config=config
            )
            self.enabled = True
            logger.info(f"SwanLab initialized (project={project}, workspace={workspace})")
        except Exception as e:
            logger.warning(f"Failed to initialize SwanLab: {e}")
    
    def log(self, metrics: Dict[str, Any]) -> None:
        """Log metrics to SwanLab"""
        if not self.enabled:
            return
        
        try:
            self.swanlab.log(metrics)
        except Exception as e:
            logger.warning(f"Failed to log metrics to SwanLab: {e}")
    
    def finish(self) -> None:
        """Finish SwanLab run"""
        if not self.enabled:
            return
        
        try:
            self.swanlab.finish()
            logger.info("SwanLab run completed")
        except Exception as e:
            logger.warning(f"Failed to finish SwanLab run: {e}")


class TensorBoardBackend(LoggerBackend):
    """TensorBoard logging backend"""
    
    def __init__(self):
        self.available = False
        self.enabled = False
        self.writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.SummaryWriter = SummaryWriter
            self.available = True
        except ImportError:
            logger.warning("TensorBoard is not installed. To use it, run: pip install tensorboard")
    
    def init(self, config: Dict[str, Any]) -> None:
        """Initialize TensorBoard with log directory"""
        if not self.available:
            logger.warning("TensorBoard is not available, skipping initialization")
            return
        
        try:
            log_dir = config.pop("log_dir", "runs/experiment")
            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)
            
            self.writer = self.SummaryWriter(str(log_path))
            self.enabled = True
            self.step_count = 0
            
            # Store config metadata
            self.config_text = "\n".join([f"{k}: {v}" for k, v in config.items()])
            if self.config_text:
                self.writer.add_text("config", self.config_text)
            
            logger.info(f"TensorBoard initialized (log_dir={log_dir})")
        except Exception as e:
            logger.warning(f"Failed to initialize TensorBoard: {e}")
    
    def log(self, metrics: Dict[str, Any]) -> None:
        """Log metrics to TensorBoard"""
        if not self.enabled or self.writer is None:
            return
        
        try:
            self.step_count += 1
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(key, value, self.step_count)
            self.writer.flush()
        except Exception as e:
            logger.warning(f"Failed to log metrics to TensorBoard: {e}")
    
    def finish(self) -> None:
        """Close TensorBoard writer"""
        if not self.enabled or self.writer is None:
            return
        
        try:
            self.writer.close()
            logger.info("TensorBoard run completed")
        except Exception as e:
            logger.warning(f"Failed to finish TensorBoard run: {e}")


class ExperimentLogger:
    """
    Unified experiment logger supporting multiple backends.
    统一的实验日志记录器，支持多个后端。
    
    Example:
        >>> # Initialize with SwanLab
        >>> logger = ExperimentLogger(backends=["swanlab"])
        >>> logger.init({
        ...     "project": "opt4",
        ...     "workspace": "Sail2Dream",
        ...     "batch_size": 16,
        ...     "learning_rate": 0.001
        ... })
        >>> 
        >>> # Log metrics during training
        >>> logger.log({"loss": 0.5, "epoch": 1})
        >>> 
        >>> # Finish logging
        >>> logger.finish()
    """
    
    def __init__(self, backends: Optional[list] = None):
        """
        Initialize logger with specified backends.
        
        Args:
            backends: List of backend names. Supported: ["swanlab", "tensorboard"]
                     If None, no backends are used.
        """
        self.backends = {}
        
        if backends is None:
            backends = []
        
        # Normalize backend names to lowercase
        backends = [b.lower() for b in backends]
        
        if "swanlab" in backends:
            self.backends["swanlab"] = SwanLabBackend()
        
        if "tensorboard" in backends:
            self.backends["tensorboard"] = TensorBoardBackend()
        
        if not self.backends:
            logger.warning("No logging backends initialized")
    
    def init(self, config: Dict[str, Any]) -> None:
        """
        Initialize all backends with config.
        
        Args:
            config: Configuration dictionary. Can contain backend-specific settings:
                   - For SwanLab: project, workspace
                   - For TensorBoard: log_dir
                   - Other keys are treated as hyperparameters
        """
        # Create copies for each backend to avoid config mutation
        for name, backend in self.backends.items():
            config_copy = config.copy()
            try:
                backend.init(config_copy)
            except Exception as e:
                logger.warning(f"Failed to initialize {name} backend: {e}")
    
    def log(self, metrics: Dict[str, Any]) -> None:
        """
        Log metrics to all active backends.
        
        Args:
            metrics: Dictionary of metric names to values
        """
        for backend in self.backends.values():
            try:
                backend.log(metrics)
            except Exception as e:
                logger.warning(f"Failed to log metrics: {e}")
    
    def finish(self) -> None:
        """Finish logging on all backends"""
        for backend in self.backends.values():
            try:
                backend.finish()
            except Exception as e:
                logger.warning(f"Failed to finish backend: {e}")
    
    def is_enabled(self) -> bool:
        """Check if any backend is enabled"""
        return any(backend.enabled for backend in self.backends.values())


# Convenience functions for quick setup
def create_logger(backends: Optional[list] = None, config: Optional[Dict[str, Any]] = None) -> ExperimentLogger:
    """
    Create and initialize an ExperimentLogger in one step.
    
    Args:
        backends: List of backend names
        config: Configuration dictionary
    
    Returns:
        Initialized ExperimentLogger instance
    """
    experiment_logger = ExperimentLogger(backends)
    if config:
        experiment_logger.init(config)
    return experiment_logger
