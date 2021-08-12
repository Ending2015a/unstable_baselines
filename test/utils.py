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
    def assertArrayEqual(self, a, b, msg=''):
        a = np.asarray(a)
        b = np.asarray(b)
        self.assertEqual(a.shape, b.shape,
            f'Shape mismatch: {a.shape} vs {b.shape}')
        np.testing.assert_array_equal(a, b, msg)

    def assertArrayClose(self, a, b, decimal=6, msg=''):
        a = np.asarray(a)
        b = np.asarray(b)
        self.assertEqual(a.shape, b.shape,
            f'Shape mismatch: {a.shape} vs {b.shape}')
        np.testing.assert_array_almost_equal(a, b, decimal, msg)


