# --- built in ---
import os
import sys
import time
import json
import logging
import unittest

# --- 3rd party ---
import numpy as np
import tensorflow as tf

from scipy import stats as sp_stats
from parameterized import parameterized

# --- my module ---
from unstable_baselines.lib import prob
from test.utils import TestCase
from unstable_baselines.lib import utils

TEST_SEED = 1

def set_test_seed():
    utils.set_seed(TEST_SEED)

def get_test_seed():
    return TEST_SEED

def make_categorical(batch_shape, num_classes, dtype=tf.int32, seed=get_test_seed()):
    logits = -50. + tf.random.uniform(
        list(batch_shape) + [num_classes], minval=-10, maxval=10,
        dtype=tf.float32, seed=seed)
    return prob.Categorical(logits, dtype=dtype)

def make_normal(mean_shape, scale_shape, dtype=tf.float32, seed=get_test_seed()):
    mean = tf.random.uniform(list(mean_shape), minval=-10, maxval=10,
            dtype=dtype, seed=seed)
    scale = tf.random.uniform(list(scale_shape), minval=-10, maxval=10,
            dtype=dtype, seed=seed)
    return prob.Normal(mean, scale, dtype)

def make_multinormal(mean_shape, scale_shape, dtype=tf.float32, seed=get_test_seed()):
    mean = tf.random.uniform(list(mean_shape), minval=-10, maxval=10,
            dtype=dtype, seed=seed)
    scale = tf.random.uniform(list(scale_shape), minval=-10, maxval=10,
            dtype=dtype, seed=seed)
    return prob.MultiNormal(mean, scale, dtype)

class TestProbModuleCategorical(TestCase):
    @parameterized.expand([
        ([],), 
        ([1],), 
        ([2, 3, 4],)
    ])
    def test_shapes(self, batch_shape):
        shape = list(batch_shape) + [10]
        dist = make_categorical(batch_shape, 10, dtype=tf.int32)
        self.assertArrayEqual(shape, dist.logits.shape)
        self.assertArrayEqual(shape, dist._p().shape)
        self.assertArrayEqual(batch_shape, dist.log_prob(np.zeros(batch_shape)).shape)
        self.assertArrayEqual(batch_shape, dist.mode().shape)
        self.assertArrayEqual(batch_shape, dist.sample().shape)
        self.assertArrayEqual(batch_shape, dist.entropy().shape)
        dist2 = make_categorical(batch_shape, 10, dtype=tf.int32)
        self.assertArrayEqual(batch_shape, dist.kl(dist2).shape)
    
    @parameterized.expand([
        (tf.int32,), 
        (tf.int64,),
    ])
    def test_dtypes(self, dtype):
        dist = make_categorical([], 10, dtype=dtype)
        self.assertEqual(dtype, dist.dtype)
        self.assertEqual(tf.float32, dist._p().dtype)
        self.assertEqual(tf.float32, dist.log_prob(0).dtype)
        self.assertEqual(dtype, dist.mode().dtype)
        self.assertEqual(dtype, dist.sample().dtype)
        self.assertEqual(tf.float32, dist.entropy().dtype)
        dist2 = make_categorical([], 10, dtype=dtype)
        self.assertEqual(tf.float32, dist.kl(dist2).dtype)

    def test_prob_without_batch(self):
        p = np.array([0.2, 0.8], dtype=np.float32)
        logits = np.log(p) - 50.
        dist = prob.Categorical(logits)
        # test logits
        self.assertArrayEqual(dist.logits, logits)
        self.assertArrayEqual(dist.logits.shape, p.shape)
        # test _p, _log_p
        self.assertAllClose(dist._p(), p)
        # test prob, log_prob
        self.assertAllClose(dist.prob(0), p[0])
        self.assertAllClose(dist.prob(1), p[1])

    def test_mode_without_batch(self):
        p = np.array([0.2, 0.8], dtype=np.float32)
        logits = np.log(p) - 50.
        dist = prob.Categorical(logits)
        # test mode
        modes = dist.mode()
        self.assertArrayEqual(1, modes)

    def test_sample_without_batch(self):
        p = np.array([0.2, 0.8], dtype=np.float32)
        logits = np.log(p) - 50.
        dist = prob.Categorical(logits)
        # test sample
        set_test_seed()
        draws = np.asarray(dist.sample(10000))
        self.assertFalse(np.any(draws < 0))
        self.assertFalse(np.any(draws > 1))
        self.assertAllClose(np.mean(draws==0, axis=0), 0.2, atol=1e-2)
        self.assertAllClose(np.mean(draws==1, axis=0), 0.8, atol=1e-2)

    def test_entropy_without_batch(self):
        p = np.array([0.2, 0.8], dtype=np.float32)
        logits = np.log(p) - 50.
        dist = prob.Categorical(logits)
        # test entropy
        self.assertAllClose(-(0.2 * np.log(0.2) + 0.8 * np.log(0.8)),
                        dist.entropy(), atol=0, rtol=1e-5)
    
    def test_prob_with_batch(self):
        p = np.array([[0.2, 0.8], [0.4, 0.6]], dtype=np.float32)
        logits = np.log(p) - 50.
        dist = prob.Categorical(logits)
        # test logits
        self.assertArrayEqual(dist.logits, logits)
        self.assertArrayEqual(dist.logits.shape, p.shape)
        # test _p, _log_p
        self.assertAllClose(dist._p(), p)
        # test prob, log_prob
        self.assertAllClose(dist.prob([0, 0]), p[..., 0])
        self.assertAllClose(dist.prob([1, 1]), p[..., 1])

    def test_mode_with_batch(self):
        p = np.array([[0.2, 0.8], [0.6, 0.4]], dtype=np.float32)
        logits = np.log(p) - 50.
        dist = prob.Categorical(logits)
        # test mode
        modes = dist.mode()
        self.assertArrayEqual([1, 0], modes)

    def test_sample_with_batch(self):
        p = np.array([[0.2, 0.8], [0.4, 0.6]], dtype=np.float32)
        logits = np.log(p) - 50.
        dist = prob.Categorical(logits)
        # test sample
        set_test_seed()
        draws = np.asarray(dist.sample(10000))
        self.assertFalse(np.any(draws < 0))
        self.assertFalse(np.any(draws > 1))
        self.assertAllClose(np.mean(draws==0, axis=0), [0.2, 0.4], atol=1e-2)
        self.assertAllClose(np.mean(draws==1, axis=0), [0.8, 0.6], atol=1e-2)
    
    def test_entropy_with_batch(self):
        p = np.array([[0.2, 0.8], [0.4, 0.6]], dtype=np.float32)
        logits = np.log(p) - 50.
        dist = prob.Categorical(logits)
        self.assertAllClose([-(0.2 * np.log(0.2) + 0.8 * np.log(0.8)),
                             -(0.4 * np.log(0.4) + 0.6 * np.log(0.6))],
                             dist.entropy(), atol=0, rtol=1e-5)
    
    def test_entropy_with_neg_inf_logits(self):
        probs = np.array([[0, 0.5, 0.5], [0, 1, 0]])
        dist = prob.Categorical(np.log(probs))

        ans = [-(0.5*np.log(0.5) + 0.5*np.log(0.5)), -(np.log(1))]
        self.assertAllClose(ans, dist.entropy())

    @parameterized.expand([
        ([1], 2),
        ([1], 4),
        ([10], 2),
        ([10], 4)
    ])
    def test_categorical_kl(self, batch_shape, num_classes):
        dist_a = make_categorical(batch_shape, num_classes, seed=1)
        dist_b = make_categorical(batch_shape, num_classes, seed=2)

        exp_a = np.exp(dist_a.logits)
        exp_b = np.exp(dist_b.logits)
        prob_a = exp_a / exp_a.sum(axis=-1, keepdims=True)
        prob_b = exp_b / exp_b.sum(axis=-1, keepdims=True)

        kl_val = dist_a.kl(dist_b)
        kl_same = dist_a.kl(dist_a)
        kl_exp = np.sum(prob_a * (np.log(prob_a) - np.log(prob_b)), axis=-1)
        self.assertArrayEqual(kl_val.shape, batch_shape)
        self.assertAllClose(kl_val, kl_exp, rtol=1e-6, atol=1e-6)
        self.assertAllClose(kl_same, np.zeros_like(kl_exp), rtol=1e-6, atol=1e-6)

class TestProbModuleNormal(TestCase):
    @parameterized.expand([
        ([], []), 
        ([1], [1]), 
        ([2, 3, 4], [1, 1, 4]),
        ([2, 3, 4], [1]),
        ([2, 3, 4], []),
        ([1, 1, 4], [2, 3, 4]),
        ([1], [2, 3, 4]),
        ([], [2, 3, 4])
    ])
    def test_shapes(self, mean_shape, scale_shape):
        batch_shape = utils.broadcast_shape(tf.TensorShape(mean_shape), 
                                            tf.TensorShape(scale_shape))
        dist = make_normal(mean_shape, scale_shape, dtype=tf.float32)
        self.assertArrayEqual(mean_shape, dist.mean.shape)
        self.assertArrayEqual(scale_shape, dist.scale.shape)
        self.assertArrayEqual(batch_shape, dist.log_prob(np.zeros(batch_shape)).shape)
        self.assertArrayEqual(batch_shape, dist.mode().shape)
        self.assertArrayEqual(batch_shape, dist.sample().shape)
        self.assertArrayEqual(batch_shape, dist.entropy().shape)
        dist2 = make_normal(mean_shape, scale_shape, dtype=tf.float32)
        self.assertArrayEqual(batch_shape, dist.kl(dist2).shape)

    @parameterized.expand([
        (tf.float16,),
        (tf.float32,),
        (tf.float64,),
    ])
    def test_dtypes(self, dtype):
        dist = make_normal([], [], dtype=dtype)
        self.assertEqual(dtype, dist.dtype)
        self.assertEqual(dtype, dist.mean.dtype)
        self.assertEqual(dtype, dist.scale.dtype)
        self.assertEqual(dtype, dist.log_prob(0).dtype)
        self.assertEqual(dtype, dist.mode().dtype)
        self.assertEqual(dtype, dist.sample().dtype)
        self.assertEqual(dtype, dist.entropy().dtype)
        dist2 = make_normal([], [], dtype=dtype)
        self.assertEqual(dtype, dist.kl(dist2).dtype)

    def test_prob(self):
        batch_size = 6
        mu = np.asarray([3.0] * batch_size, dtype=np.float32)
        sigma = np.asarray([np.sqrt(10.0)] * batch_size, dtype=np.float32)
        x = np.array([-2.5, 2.5, 4.0, 0.0, -1.0, 2.0], dtype=np.float32)
        dist = prob.Normal(mean=mu, scale=sigma)
        # test mean scale
        self.assertArrayEqual(mu, dist.mean)
        self.assertArrayEqual(sigma, dist.scale)
        # test prob, log_prob
        expected_log_prob = sp_stats.norm(mu, sigma).logpdf(x)
        self.assertArrayClose(expected_log_prob, dist.log_prob(x))
        self.assertArrayClose(np.exp(expected_log_prob), dist.prob(x))

    def test_prob_multidims(self):
        batch_size = 6
        mu = np.asarray([[3.0, -3.0]] * batch_size, dtype=np.float32)
        sigma = np.asarray(
            [[np.sqrt(10.0), np.sqrt(15.0)]] * batch_size, dtype=np.float32)
        x = np.array([[-2.5, 2.5, 4.0, 0.0, -1.0, 2.0]], dtype=np.float32).T
        dist = prob.Normal(mean=mu, scale=sigma)
        # test mean scale
        self.assertArrayEqual(mu, dist.mean)
        self.assertArrayEqual(sigma, dist.scale)
        # test prob, log_prob
        expected_log_prob = sp_stats.norm(mu, sigma).logpdf(x)
        self.assertArrayClose(expected_log_prob, dist.log_prob(x))
        self.assertArrayClose(np.exp(expected_log_prob), dist.prob(x))

    def test_mode(self):
        batch_size = 6
        mu = np.asarray([3.0] * batch_size, dtype=np.float32)
        sigma = np.asarray([np.sqrt(10.0)] * batch_size, dtype=np.float32)
        x = np.array([-2.5, 2.5, 4.0, 0.0, -1.0, 2.0], dtype=np.float32)
        dist = prob.Normal(mean=mu, scale=sigma)
        # test prob, log_prob
        self.assertArrayEqual(mu.shape, dist.mode().shape)
        self.assertArrayClose(mu, dist.mode())

    def test_mode_multidims(self):
        batch_size = 6
        mu = np.asarray([[3.0, -3.0]] * batch_size, dtype=np.float32)
        sigma = np.asarray(
            [[np.sqrt(10.0), np.sqrt(15.0)]] * batch_size, dtype=np.float32)
        x = np.array([[-2.5, 2.5, 4.0, 0.0, -1.0, 2.0]], dtype=np.float32).T
        dist = prob.Normal(mean=mu, scale=sigma)
        # test prob, log_prob
        self.assertArrayEqual(mu.shape, dist.mode().shape)
        self.assertArrayClose(mu, dist.mode())

    def test_sample(self):
        mu = np.asarray(3.0)
        sigma = np.sqrt(3.0)
        dist = prob.Normal(mean=mu, scale=sigma)
        set_test_seed()
        draws = np.asarray(dist.sample(100000))
        self.assertArrayEqual(draws.shape, (100000,))
        self.assertAllClose(draws.mean(), mu, atol=1e-1)
        self.assertAllClose(draws.std(), sigma, atol=1e-1)

    def test_sample_with_batch(self):
        batch_size = 2
        mu = np.asarray([[3.0, -3.0]] * batch_size)
        sigma = np.asarray([[np.sqrt(2.0), np.sqrt(3.0)]] * batch_size)
        dist = prob.Normal(mean=mu, scale=sigma)
        set_test_seed()
        draws = np.asarray(dist.sample(100000))
        self.assertArrayEqual(draws.shape, (100000, batch_size, 2))
        self.assertAllClose(draws[:, 0, 0].mean(), mu[0, 0], atol=1e-1)
        self.assertAllClose(draws[:, 0, 0].std(), sigma[0, 0], atol=1e-1)
        self.assertAllClose(draws[:, 0, 1].mean(), mu[0, 1], atol=1e-1)
        self.assertAllClose(draws[:, 0, 1].std(), sigma[0, 1], atol=1e-1)
    
    def test_sample_multidims(self):
        mu = np.asarray(3.0)
        sigma = np.sqrt(3.0)
        dist = prob.Normal(mean=mu, scale=sigma)
        set_test_seed()
        draws = np.asarray(dist.sample([100, 1000]))
        self.assertArrayEqual(draws.shape, (100, 1000))
        self.assertAllClose(draws.mean(), mu, atol=1e-1)
        self.assertAllClose(draws.std(), sigma, atol=1e-1)

    def test_entropy(self):
        mu = np.asarray(2.34)
        sigma = np.asarray(4.56)
        dist = prob.Normal(mean=mu, scale=sigma)
        self.assertArrayEqual((), dist.entropy().shape)
        self.assertAllClose(sp_stats.norm(mu, sigma).entropy(), dist.entropy())

    def test_entropy_multidims(self):
        mu = np.asarray([1.0, 1.0, 1.0])
        sigma = np.asarray([[1.0, 2.0, 3.0]]).T
        dist = prob.Normal(mean=mu, scale=sigma)
        expected_ent = 0.5 * np.log(2 * np.pi * np.exp(1) * (mu*sigma)**2)
        self.assertArrayEqual(expected_ent.shape, dist.entropy().shape)
        self.assertAllClose(expected_ent, dist.entropy())

    def test_kl(self):
        batch_size = 6
        mu_a = np.array([3.0] * batch_size)
        sigma_a = np.array([1.0, 2.0, 3.0, 1.5, 2.5, 3.5])
        mu_b = np.array([-3.0] * batch_size)
        sigma_b = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        dist_a = prob.Normal(mean=mu_a, scale=sigma_a)
        dist_b = prob.Normal(mean=mu_b, scale=sigma_b)
        kl = dist_a.kl(dist_b)
        expected_kl = ((mu_a - mu_b)**2 / (2 * sigma_b**2) + 0.5 * (
            (sigma_a**2 / sigma_b**2) - 1 - 2 * np.log(sigma_a / sigma_b)))
        self.assertArrayEqual(kl.shape, (batch_size,))
        self.assertAllClose(expected_kl, kl)
        # test estimate kl
        set_test_seed()
        draws = dist_a.sample(100000)
        sample_kl = dist_a.log_prob(draws) - dist_b.log_prob(draws)
        sample_kl = tf.reduce_mean(sample_kl, axis=0)
        self.assertAllClose(expected_kl, sample_kl, atol=0.0, rtol=1e-2)

class TestProbModuleMultiNormal(TestCase):
    def test_shape_exception(self):
        mu = 1.
        sigma = -5.
        with self.assertRaises(RuntimeError):
            prob.MultiNormal(mean=mu, scale=sigma)

    def test_shape_no_exception(self):
        mu = [1.]
        sigma = [-5.]
        prob.MultiNormal(mean=mu, scale=sigma)

    @parameterized.expand([
        ([1], [1]), 
        ([2, 3, 4], [1, 1, 4]),
        ([2, 3, 4], [1]),
        ([2, 3, 4], []),
        ([1, 1, 4], [2, 3, 4]),
        ([1], [2, 3, 4]),
        ([], [2, 3, 4])
    ])
    def test_shapes(self, mean_shape, scale_shape):
        full_shape = (np.ones(mean_shape) * np.ones(scale_shape)).shape
        batch_shape = full_shape[:-1]
        dist = make_multinormal(mean_shape, scale_shape, dtype=tf.float32)
        self.assertArrayEqual(mean_shape, dist.mean.shape)
        self.assertArrayEqual(scale_shape, dist.scale.shape)
        self.assertArrayEqual(batch_shape, dist.log_prob(np.zeros(full_shape)).shape)
        self.assertArrayEqual(full_shape, dist.mode().shape)
        self.assertArrayEqual(full_shape, dist.sample().shape)
        self.assertArrayEqual(batch_shape, dist.entropy().shape)
        dist2 = make_multinormal(mean_shape, scale_shape, dtype=tf.float32)
        self.assertArrayEqual(batch_shape, dist.kl(dist2).shape)

    @parameterized.expand([
        (tf.float16,),
        (tf.float32,),
        (tf.float64,),
    ])
    def test_dtypes(self, dtype):
        dist = make_multinormal([1], [1], dtype=dtype)
        self.assertEqual(dtype, dist.dtype)
        self.assertEqual(dtype, dist.mean.dtype)
        self.assertEqual(dtype, dist.scale.dtype)
        self.assertEqual(dtype, dist.log_prob(0).dtype)
        self.assertEqual(dtype, dist.mode().dtype)
        self.assertEqual(dtype, dist.sample().dtype)
        self.assertEqual(dtype, dist.entropy().dtype)
        dist2 = make_multinormal([1], [1], dtype=dtype)
        self.assertEqual(dtype, dist.kl(dist2).dtype)

    def test_prob(self):
        mu = np.asarray([1.0, -1.0], dtype=np.float32)
        sigma = np.asarray([3.0, 2.0], dtype=np.float32)
        x = np.array([2.5, 0.5], dtype=np.float32)
        dist = prob.MultiNormal(mean=mu, scale=sigma)
        # test mean scale
        self.assertArrayEqual(mu, dist.mean)
        self.assertArrayEqual(sigma, dist.scale)
        # test prob, log_prob
        exp_mvn = sp_stats.multivariate_normal(mu, np.diag(sigma)**2)
        self.assertArrayClose(exp_mvn.logpdf(x), dist.log_prob(x))
        self.assertArrayClose(np.exp(exp_mvn.logpdf(x)), dist.prob(x))

    def test_sample(self):
        mu = np.asarray([1.0, -1.0])
        sigma = np.asarray([1.0, 5.0])
        dist = prob.MultiNormal(mean=mu, scale=sigma)
        set_test_seed()
        draws = np.asarray(dist.sample(100000))
        self.assertArrayEqual(draws.shape, (100000,2))
        self.assertAllClose(draws.mean(axis=0), mu, atol=1e-1)
        self.assertAllClose(draws.var(axis=0), sigma**2, atol=1e-1)

    def test_entropy(self):
        mu = np.asarray([1.0, 0.0, -1.0])
        sigma = np.asarray([1.0, 2.0, 3.0])
        dist = prob.MultiNormal(mean=mu, scale=sigma)
        exp_mn = sp_stats.multivariate_normal(mean=mu, cov=np.diag(sigma)**2)
        self.assertArrayEqual(exp_mn.entropy().shape, dist.entropy().shape)
        self.assertAllClose(exp_mn.entropy(), dist.entropy())

    def test_kl(self):
        mu_a = np.array([3.0, -1.0])
        sigma_a = np.array([1.0, 2.5])
        mu_b = np.array([-3.0, 1.5])
        sigma_b = np.array([0.5, 1.0])
        dist_a = prob.MultiNormal(mean=mu_a, scale=sigma_a)
        dist_b = prob.MultiNormal(mean=mu_b, scale=sigma_b)
        kl = dist_a.kl(dist_b)
        expected_kl = ((mu_a - mu_b)**2 / (2 * sigma_b**2) + 0.5 * (
            (sigma_a**2 / sigma_b**2) - 1 - 2 * np.log(sigma_a / sigma_b))).sum()
        self.assertArrayEqual(kl.shape, [])
        self.assertAllClose(expected_kl, kl)
        # test estimate kl
        set_test_seed()
        draws = dist_a.sample(100000)
        sample_kl = dist_a.log_prob(draws) - dist_b.log_prob(draws)
        sample_kl = tf.reduce_mean(sample_kl, axis=0)
        self.assertAllClose(expected_kl, sample_kl, atol=0.0, rtol=1e-2)
