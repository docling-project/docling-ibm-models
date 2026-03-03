import json
import tempfile

import docling_ibm_models.tableformer.common as c


test_config_a = {
    "base_dir": "./tests/test_data/",
    "curr_dir": "./tests/test_data/test_common/",
    "data_top_dir": "./tests/test_data/",
    "dataset": {
        "name": ["PhysRevB"],
        "limit": 10,
        "split": {"test": 0.2, "train": 0.5, "evaluate": 0.3},
    },
    "features": {
        "name": "Data2Features03b",
        "parameters": {
            "normalize_features": True,
            "normalize_features_method": "Z-Score",
        },
    },
}


test_config_b = {"preparation": {"max_tag_len": 300}, "model": {"seq_len": 30}}

test_config_c = {"preparation": {"max_tag_len": 300}, "model": {"seq_len": 302}}

test_config_d = {"preparation": {"max_tag_len": 300}, "model": {"seq_len": 303}}


def test_safe_get_parameters():
    val = c.safe_get_parameter(None, None, 10)
    assert val == 10, "Failed with null objects"

    index_path = ["features", "parameters", "normalize_features_method"]
    val = c.safe_get_parameter(test_config_a, index_path, None)
    assert val == "Z-Score", "Cannot find existing parameter"

    index_path = ["features", "parameters", "wrong"]
    val = c.safe_get_parameter(test_config_a, index_path, "hello")
    assert val == "hello", "Default value should be here"

    index_path = ["features", "wrong", "normalize_features_method"]
    val = c.safe_get_parameter(test_config_a, index_path, 10)
    assert val == 10, "Default value should be here"

    index_path = ["model", "parameters", "normalize_features_method"]
    val = c.safe_get_parameter(test_config_a, index_path, "hello")
    assert val == "hello", "Default value should be here"

    # Test exception throwing
    exRaised = False
    try:
        index_path = ["missing"]
        val = c.safe_get_parameter(test_config_a, index_path, required=True)
    except ValueError:
        exRaised = True
    assert exRaised, "Exception should had been raised here"


def test_config_validation():
    configs = [test_config_b, test_config_c, test_config_d]

    for i, config in enumerate(configs):
        try:
            val = c.validate_config(config)
            if i == 0 or i == 1:
                assert val, "Valid configuration didn't pass the validation test"
        except AssertionError:
            assert i == 2, "Configuration validation error"

def test_read_config():
    r"""
    Testing the read_config() function
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fp:
        # Write a tmp file
        json.dump(test_config_a, fp)
        fp.close()

        # Read the tmp file and extract the config
        config = c.read_config(fp.name)
        assert isinstance(config, dict)
