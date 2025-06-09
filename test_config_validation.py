import unittest
import tempfile
import os
import json

from JSON_Loader import load_config


class TestConfigValidation(unittest.TestCase):

    def setUp(self):
        # Create a dummy image file that will act as a valid input
        self.dummy_input_path = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        self.dummy_input_path.write(b"dummy image data")
        self.dummy_input_path.close()
        self.input_path = self.dummy_input_path.name

    def tearDown(self):
        # Clean up dummy file
        os.remove(self.input_path)

    def create_config_file(self, data):
        """Helper to create a temporary config JSON file"""
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump(data, f)
        f.close()
        return f.name

    def test_valid_config_display_true(self):
        cfg = {
            "input": self.input_path,
            "output": "",
            "display": True,
            "operations": [
                {"type": "brightness", "value": 1.2}
            ]
        }
        cfg_path = self.create_config_file(cfg)
        result = load_config(cfg_path)
        self.assertEqual(result["display"], True)
        os.remove(cfg_path)

    def test_valid_config_with_output(self):
        cfg = {
            "input": self.input_path,
            "output": "output.png",
            "display": False,
            "operations": [
                {"type": "box", "width": 3, "height": 3}
            ]
        }
        cfg_path = self.create_config_file(cfg)
        result = load_config(cfg_path)
        self.assertEqual(result["output"], "output.png")
        os.remove(cfg_path)

    def test_missing_required_field(self):
        cfg = {
            "output": "output.png",
            "display": True,
            "operations": []
        }
        cfg_path = self.create_config_file(cfg)
        with self.assertRaises(ValueError) as cm:
            load_config(cfg_path)
        self.assertIn("Schema validation error", str(cm.exception))
        os.remove(cfg_path)

    def test_invalid_operation_parameter(self):
        cfg = {
            "input": self.input_path,
            "output": "",
            "display": True,
            "operations": [
                {"type": "brightness"}  # missing required "value"
            ]
        }
        cfg_path = self.create_config_file(cfg)
        with self.assertRaises(ValueError) as cm:
            load_config(cfg_path)
        self.assertIn("Schema validation error", str(cm.exception))
        os.remove(cfg_path)

    def test_input_file_does_not_exist(self):
        cfg = {
            "input": "not_a_real_file.jpg",
            "output": "",
            "display": True,
            "operations": []
        }
        cfg_path = self.create_config_file(cfg)
        with self.assertRaises(FileNotFoundError):
            load_config(cfg_path)
        os.remove(cfg_path)

    def test_both_output_and_display_inactive(self):
        cfg = {
            "input": self.input_path,
            "output": "",
            "display": False,
            "operations": []
        }
        cfg_path = self.create_config_file(cfg)
        with self.assertRaises(ValueError) as cm:
            load_config(cfg_path)
        self.assertIn("Config must include either", str(cm.exception))
        os.remove(cfg_path)


if __name__ == "__main__":
    unittest.main()
