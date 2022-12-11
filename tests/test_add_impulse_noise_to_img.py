import unittest
import sys
import src

from mock import patch


class TestAddImpulseNoiseToImg(unittest.TestCase):
    def test_bad_input_for_check(self):
        self.assertRaises(Exception, src.__convert_inputs, ('asd.jpg', 0.125))
