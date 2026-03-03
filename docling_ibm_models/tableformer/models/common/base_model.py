import glob
import logging
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path

import torch

import docling_ibm_models.tableformer.settings as s

LOG_LEVEL = logging.INFO
# LOG_LEVEL = logging.DEBUG


class BaseModel(ABC):
    r"""
    BaseModel provides some common functionality for all models:
    - Saves checkpoint files for each epoch
    - Loads the model from the best available checkpoint
    - Save repository branch and commit
    """

    def __init__(self, config, init_data, device):
        r"""
        Inputs:
            config: The configuration file
            init_data: Dictionary with initialization data. This dictionary can be used to pass any
                       kind of initialization data for the models
            device: The device used to move the tensors of the model
        """
        super(BaseModel, self).__init__()

        # Set config and device
        self._config = config
        self._init_data = init_data

        self._device = device

        self._save_dir = config["model"]["save_dir"]
        self._load_checkpoint = None
        if "load_checkpoint" in config["model"]:
            self._load_checkpoint = config["model"]["load_checkpoint"]

        self._branch_name = "dev/next"
        self._commit_sha = "1"

        # Keep a dictionary with the starting times per epoch.
        # NOTICE: Epochs start from 0
        self._epoch_start_ts = {0: time.time()}

    def _log(self):
        # Setup a custom logger
        return s.get_custom_logger(self.__class__.__name__, LOG_LEVEL)

    @abstractmethod
    def predict(self, img, max_steps, beam_size, return_attention):
        pass

    def count_parameters(self):
        r"""Counts the number of trainable parameters of this model

        Output:
            num_parameters: number of trainable parameters
        """
        num_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return num_parameters

    def get_code_version(self):
        r"""Gets the source control version of this model code

        Returns
        -------
        branch_name : str
            The name of the Git branch of this model code
        commit_sha : str
            The unique identifier of the Git commit of this model code
        """

        return self._branch_name, self._commit_sha

    def get_save_directory(self):
        r"""
        Return the save directory
        """
        return self._save_dir

    def is_saved(self):
        r"""
        This method returns True if both conditions are met:
        1. There is a checkpoint file for the model.
        2. The checkpoint file corresponds to the last training epoch set in the configuration file.
        """
        # Get the saved_model
        saved_model, _ = self._load_best_checkpoint()

        if saved_model is None:
            return False

        epochs = self._config["train"]["epochs"]
        self._log().debug(
            "Best epoch in saved model: {}; Number of epochs in config: {}".format(
                saved_model["epoch"], epochs
            )
        )
        if epochs == saved_model["epoch"] + 1:
            return True

        return False

    def save(self, epoch=None, optimizers=None, losses=None, model_parameters=None):
        r"""
        Save the model data to the disk as a pickle file.

        Parameters
        ----------
        epoch: Training epoch
        optimizers: Dictionary with the optimizers. The key specifies what the optimizer is
                     used for. The 'state_dict' of each optimizer will be saved in the
                     checkpoint file.
        losses: Dictionary with the losses. The key specifies what the loss is used for. Each
                value is a list
        model_parameters: Dictionary with model specific parameters that we need to save in the
                              checkpoint file.
        Returns
        -------
        True if success, False otherwise
        """
        # Get the checkpoint_filename
        c_filename = self._build_checkpoint_filename(epoch)
        self._log().debug("Trying to save checkpoint file: {}".format(c_filename))

        # Prepare a dictionary with all data we want to save
        optimizers_state_dict = None
        if optimizers is not None:
            optimizers_state_dict = {k: v.state_dict() for k, v in optimizers.items()}

        model_data = {
            "model_state_dict": self.state_dict(),
            "epoch": epoch,
            "optimizers": optimizers_state_dict,
            "losses": losses,
            "model_parameters": model_parameters,
        }

        # Add the processing time per epoch
        now = time.time()
        self._epoch_start_ts[epoch + 1] = now
        if epoch in self._epoch_start_ts:
            dt = now - self._epoch_start_ts[epoch]
            model_data["epoch_start_ts"] = self._epoch_start_ts[epoch]
            model_data["epoch_dt"] = dt

        # Create the save directory
        Path(self._save_dir).mkdir(parents=True, exist_ok=True)

        # Save the model
        torch.save(model_data, c_filename)

        # Return true if file is present, otherwise false
        if not os.path.isfile(c_filename):
            self._log().error("Cannot find the file to save: " + c_filename)
            return False

        # store code branch name and commit
        version_file = os.path.join(self._save_dir, "_version")
        with open(version_file, "w") as text_file:
            print("Model is using code [commit:branch]", file=text_file)
            print("{}:{}".format(self._commit_sha, self._branch_name), file=text_file)

        return True

    def load(self, optimizers=None):
        r"""
        Load the model data from the disk.
        The method will iterate over all *.check files and try to load the one from the highest
        epoch.

        Input:
            -optimizers: Dictionary with optimizers. If it is not null the keys will be used to
                         associate the corresponding state_dicts from the checkpoint file and update
                         the internal states of the provided optimizers.

        Output:
            - Success: True/ False
            - epoch: Loaded epoch or -1 if there are no checkpoint files
            - optimizers: Dictionary with loaded optimizers or empty dictionary of there is no
                          checkpoint file
            - losses: Dictionary with loaded losses or empty dictionary of there is no checkpoint
                      file
            - model_parameters: Dictionary with the model parameters or empty dictionary if there
                                are no checkpoint files
        """
        # Get the saved_model
        saved_model, _ = self._load_best_checkpoint()

        # Restore the model
        if saved_model is None:
            self._log().debug("No saved model checkpoint found")
            return False, -1, optimizers, {}, {}

        self._log().debug("Loading model from checkpoint file")
        self.load_state_dict(saved_model["model_state_dict"])

        epoch = 0
        if "epoch" in saved_model:
            epoch = saved_model["epoch"]
        losses = {}
        if "losses" in saved_model:
            losses = saved_model["losses"]
        model_parameters = saved_model["model_parameters"]

        if optimizers is not None:
            for key, optimizer_state_dict in saved_model["optimizers"].items():
                optimizers[key].load_state_dict(optimizer_state_dict)

        # Reset the start_ts of the next epoch
        self._epoch_start_ts[epoch + 1] = time.time()

        return True, epoch, optimizers, losses, model_parameters

    def _load_best_checkpoint(self):
        r"""
        If a "load_checkpoint" file has been provided, load this one.
        Otherwise use the "save_dir" and load the one with the most advanced epoch

        Returns
        -------
        saved_model : dictionary
            Checkpoint file contents generated by torch.load, or None
        checkpoint_file : string
            Filename of the loaded checkpoint, or None
        """
        checkpoint_files = []
        # If a "load_checkpoint" file is provided, try to load it
        if self._load_checkpoint is not None:
            if not os.path.exists(self._load_checkpoint):
                self._log().error(
                    "Cannot load the checkpoint: {}".format(self._load_checkpoint)
                )
                return None, None
            checkpoint_files.append(self._load_checkpoint)
        else:
            # Iterate over all check files from the directory by reverse alphabetical order
            # This will get the biggest epoch first
            checkpoint_files = glob.glob(os.path.join(self._save_dir, "*.check"))
            checkpoint_files.sort(reverse=True)

        for checkpoint_file in checkpoint_files:
            try:
                # Try to load the file
                self._log().info(
                    "Loading model checkpoint file: {}".format(checkpoint_file)
                )
                saved_model = torch.load(
                    checkpoint_file, map_location=self._device, weights_only=False
                )
                return saved_model, checkpoint_file
            except RuntimeError:
                self._log().error("Cannot load file: {}".format(checkpoint_file))

        return None, None

    def _build_checkpoint_filename(self, epoch):
        r"""
        Construct the full path for the filename of this checkpoint
        """
        dataset_name = self._config["dataset"]["name"]
        model_type = self._config["model"]["type"]
        model_name = self._config["model"]["name"]
        filename = "{}_{}_{}_{:03}.check".format(
            model_type, model_name, dataset_name, epoch
        )
        c_filename = os.path.join(self._save_dir, filename)

        return c_filename
