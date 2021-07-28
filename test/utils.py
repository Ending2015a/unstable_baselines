# --- built in ---
import os
import sys
import time
import logging
import unittest

# --- 3rd party ---
import numpy as np

# --- my module ---

class TestCase(unittest.TestCase):
    def assertArrayEqual(self, a, b, msg=None):
        a = np.asarray(a)
        b = np.asarray(b)
        self.assertEqual(a.shape, b.shape,
            f'Shape mismatch: expected {a.shape}, got {b.shape}')
        np.testing.assert_array_equal(a, b, msg)