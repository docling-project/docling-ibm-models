import logging
import sys


def get_custom_logger(logger_name, level, stream=sys.stdout):
    r"""
    Create a custom logger with a standard formatting

    Inputs:
    - logger_name: Name of the logger. You can get the class name as self.__class__.__name__
    - level: logging level (e.g. logging.INFO, logging.DEBUG, etc.)
    - stream: One of sys.stdout or sys.stderr

    Outputs:
        logger
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    # Set the handler
    if not logger.hasHandlers():
        handler = logging.StreamHandler(stream)
        formatter = logging.Formatter(
            "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


###################################################################################
# System constants
#

r"""
This is a "generic" logger available to all scripts.
It is encouraged that each class has it's own custom logger with the name of the class.
You can use the "get_custom_logger" function to build a custom logger with a standard format.
"""
LOGGER = get_custom_logger("docling-pm", logging.INFO)

# Supported dataset types
supported_datasets = ["TF_prepared"]  # TF prepared dataset

# Split names
TRAIN_SPLIT = "train"
VAL_SPLIT = "val"
TEST_SPLIT = "test"

# Prepared data parts and filename templates
PREPARED_DATA_PARTS = {
    # Array with the bboxes (x1y1x2y2) for all cells of the images across all splits.
    # The bboxes are indexed with the filename.
    # Notices:
    #   - The bboxes are NOT transformed.
    #   - If the image filenames are the same across splits, there will be one one entry in the file
    "BBOXES": "BBOXES.json",
    # Image filenames used for train and val
    "IMAGES": "IMAGES.json",
    # Mean, std, variance as arrays of 3 (for each color)
    "STATISTICS": "STATISTICS_<POSTFIX>.json",  # PRECOMPUTED
    # Bboxes of the cells in the form [1, x1, x2, y1, y2] or [0, 0, 0, 0, 0] in case of no box.
    "TRAIN_CELLBBOXES": "TRAIN_CELLBBOXES_<POSTFIX>.json",  # NOT USED.
    # Array with arrays of the length + 2 of the original cells per image.
    "TRAIN_CELLLENS": "TRAIN_CELLLENS_<POSTFIX>.json",
    # Indices of the cells between <start> <end> and <pad> at the end.
    "TRAIN_CELLS": "TRAIN_CELLS_<POSTFIX>.json",
    # Array with the length + 2 of the original tags per image.
    "TRAIN_TAGLENS": "TRAIN_TAGLENS_<POSTFIX>.json",
    # Indices of the tags between <start> <end> and <pad> at the end.
    "TRAIN_TAGS": "TRAIN_TAGS_<POSTFIX>.json",
    # Ground truth for the evaluation dataset per eval image.
    "VAL": "VAL.json",
    # Vocabulary: Indices of the word_map_cells and word_map_tags
    "WORDMAP": "WORDMAP_<POSTFIX>.json",  # PRECOMPUTED
}

# Purposes
TRAIN_PURPOSE = "train"
VAL_PURPOSE = "val"
TEST_PURPOSE = "test"
PREDICT_PURPOSE = "predict"

# The DDP world size when we train in CPU with DDP enabled
DDP_CPU_WORLD_SIZE = 2
