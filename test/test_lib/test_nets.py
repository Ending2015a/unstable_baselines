# --- built in ---
import os
import sys
import time
import json
import logging
import unittest

# --- 3rd party ---
import gym
import numpy as np
import tensorflow as tf

# --- my module ---
from unstable_baselines import nets as ub_nets
from unstable_baselines import prob as ub_prob
from unstable_baselines import utils as ub_utils
from unstable_baselines import patch as ub_patch
from test.utils import TestCase


class CustomMultiHeadValueNets(ub_nets.MultiHeadValueNets):
    def __init__(self, output_dim, n_heads=2, **kwargs):
        super().__init__(n_heads, **kwargs)
        self.output_dim = output_dim

    def create_value_model(self):
        return tf.keras.Sequential([
            tf.keras.layers.Dense(64),
            ub_patch.ReLU(),
            tf.keras.layers.Dense(64),
            ub_patch.ReLU(),
            tf.keras.layers.Dense(self.output_dim)
        ])

class StateIndepDiagGaussianPolicyNet(ub_nets.DiagGaussianPolicyNet):
    def create_logstd_model(self):
        return ub_nets.Constant(
            tf.Variable(np.zeros((self.action_dims,), 
                        dtype=np.float32))
        )

class TestNetsModule(TestCase):
    '''Test unstable_baselines.nets module
    '''
    def test_identity(self):
        batch_size = 5
        input_dim = 64
        net = ub_nets.Identity()
        inputs = tf.random.normal((batch_size, input_dim), dtype=tf.float32)
        outputs = net(inputs)
        self.assertArrayEqual(inputs.shape, outputs.shape)
        self.assertArrayEqual(inputs, outputs)

    def test_constant(self):
        batch_size = 5
        input_dim = 32
        const_dim = 64
        const = np.zeros((const_dim,), dtype=np.float32)
        batch_const = np.asarray([const] * batch_size)
        net = ub_nets.Constant(tf.Variable(const))
        inputs = tf.random.normal((batch_size, input_dim), dtype=tf.float32)
        outputs = net(inputs)
        self.assertArrayEqual(batch_const.shape, outputs.shape)
        self.assertArrayEqual(batch_const, outputs)
        
    def test_mlp_net(self):
        batch_size = 5
        input_dim = 64
        hid1 = 128
        hid2 = 256
        net = ub_nets.MlpNet([hid1, hid2])
        outputs = net(tf.zeros((batch_size, input_dim)))
        self.assertArrayEqual((batch_size, hid2), outputs.shape)
        self.assertEqual(4, len(net.trainable_variables))

    def test_nature_cnn(self):
        batch_size = 5
        w = 64
        h = 32
        c = 3
        output_dim = 512
        net = ub_nets.NatureCnn()
        outputs = net(tf.zeros((batch_size, h, w, c)))
        self.assertArrayEqual((batch_size, output_dim), outputs.shape)
        self.assertEqual(8, len(net.trainable_variables))

    def test_categorical_policy_net(self):
        batch_size = 5
        act_dim = 16
        obs_dim = 60
        action_space = gym.spaces.Discrete(act_dim)
        pi   = ub_nets.CategoricalPolicyNet(action_space)
        obs  = tf.zeros((batch_size, obs_dim), dtype=tf.float32)
        dist = pi(obs)

        self.assertTrue(isinstance(dist, ub_prob.Categorical))
        act = dist.sample()
        self.assertArrayEqual((batch_size,), act.shape)

    def test_diag_gaussian_policy_net_wo_squash(self):
        batch_size = 5
        act_dim = 16
        obs_dim = 60
        low  = np.ones((act_dim,), dtype=np.float32) * -1.0
        high = np.ones((act_dim,), dtype=np.float32) * 1.0
        action_space = gym.spaces.Box(low=low, high=high)
        pi   = ub_nets.DiagGaussianPolicyNet(action_space, squash=False)
        obs  = tf.zeros((batch_size, obs_dim), dtype=tf.float32)
        dist = pi(obs)

        self.assertTrue(isinstance(dist, ub_prob.MultiNormal))
        act = dist.sample()
        self.assertArrayEqual((batch_size, act_dim), act.shape)

    def test_diag_gaussian_policy_net_w_squash(self):
        batch_size = 5
        act_dim = 16
        obs_dim = 60
        low  = np.ones((act_dim,), dtype=np.float32) * -1.0
        high = np.ones((act_dim,), dtype=np.float32) * 1.0
        action_space = gym.spaces.Box(low=low, high=high)
        pi   = ub_nets.DiagGaussianPolicyNet(action_space, squash=True)
        obs  = tf.zeros((batch_size, obs_dim), dtype=tf.float32)
        dist = pi(obs)

        self.assertTrue(isinstance(dist, ub_prob.Tanh))
        self.assertTrue(isinstance(dist.dist, ub_prob.MultiNormal))
        act = dist.sample()
        self.assertArrayEqual((batch_size, act_dim), act.shape)

    def test_policy_net_categorical_wo_net(self):
        batch_size = 5
        act_dim = 16
        obs_dim = 60
        action_space = gym.spaces.Discrete(act_dim)
        pi = ub_nets.PolicyNet(action_space)
        obs  = tf.zeros((batch_size, obs_dim), dtype=tf.float32)
        dist = pi(obs)
        self.assertTrue(isinstance(dist, ub_prob.Categorical))
        act = dist.sample()
        self.assertArrayEqual((batch_size,), act.shape)
        self.assertEqual(2, len(pi.trainable_variables))

    def test_policy_net_categorical_w_net(self):
        batch_size = 5
        act_dim = 16
        obs_dims = (32, 32, 3)
        action_space = gym.spaces.Discrete(act_dim)
        net = ub_nets.NatureCnn()
        pi = ub_nets.PolicyNet(action_space, net=net)
        obs  = tf.zeros((batch_size, *obs_dims), dtype=tf.float32)
        dist = pi(obs)
        self.assertTrue(isinstance(dist, ub_prob.Categorical))
        act = dist.sample()
        self.assertArrayEqual((batch_size,), act.shape)
        self.assertEqual(10, len(pi.trainable_variables))

    def test_policy_net_diag_gaussian_wo_net(self):
        batch_size = 5
        act_dim = 16
        obs_dim = 60
        low = np.ones((act_dim,), dtype=np.float32) * -1.0
        high = np.ones((act_dim,), dtype=np.float32) * 1.0
        action_space = gym.spaces.Box(low=low, high=high)
        pi   = ub_nets.PolicyNet(action_space, squash=False)
        obs  = tf.zeros((batch_size, obs_dim), dtype=tf.float32)
        dist = pi(obs)
        self.assertTrue(isinstance(dist, ub_prob.MultiNormal))
        act = dist.sample()
        self.assertArrayEqual((batch_size, act_dim), act.shape)
        self.assertEqual(4, len(pi.trainable_variables))

    def test_policy_net_diag_gaussian_w_net(self):
        batch_size = 5
        act_dim = 16
        obs_dim = 60
        hid1 = 32
        hid2 = 64
        low = np.ones((act_dim,), dtype=np.float32) * -1.0
        high = np.ones((act_dim,), dtype=np.float32) * 1.0
        action_space = gym.spaces.Box(low=low, high=high)
        net  = ub_nets.MlpNet([hid1, hid2])
        pi   = ub_nets.PolicyNet(action_space, squash=False, net=net)
        obs  = tf.zeros((batch_size, obs_dim), dtype=tf.float32)
        dist = pi(obs)
        self.assertTrue(isinstance(dist, ub_prob.MultiNormal))
        act = dist.sample()
        self.assertArrayEqual((batch_size, act_dim), act.shape)
        self.assertEqual(8, len(pi.trainable_variables))

    def test_policy_net_diag_gaussian_squashed_wo_net(self):
        batch_size = 5
        act_dim = 16
        obs_dim = 60
        low = np.ones((act_dim,), dtype=np.float32) * -1.0
        high = np.ones((act_dim,), dtype=np.float32) * 1.0
        action_space = gym.spaces.Box(low=low, high=high)
        pi   = ub_nets.PolicyNet(action_space, squash=True)
        obs  = tf.zeros((batch_size, obs_dim), dtype=tf.float32)
        dist = pi(obs)
        self.assertTrue(isinstance(dist, ub_prob.Tanh))
        self.assertTrue(isinstance(dist.dist, ub_prob.MultiNormal))
        act = dist.sample()
        self.assertArrayEqual((batch_size, act_dim), act.shape)
        self.assertEqual(4, len(pi.trainable_variables))

    def test_value_net(self):
        batch_size = 5
        act_dim = 16
        obs_dim = 60
        vf  = ub_nets.ValueNet()
        obs = tf.zeros((batch_size, obs_dim), dtype=tf.float32)
        val = vf(obs)
        self.assertArrayEqual((batch_size, 1), val.shape)
        self.assertEqual(2, len(vf.trainable_variables))

    def test_value_net_w_net(self):
        batch_size = 5
        act_dim = 16
        obs_dim = 60
        hid1 = 32
        hid2 = 64
        net = ub_nets.MlpNet([hid1, hid2])
        vf = ub_nets.ValueNet(net=net)
        obs = tf.zeros((batch_size, obs_dim), dtype=tf.float32)
        val = vf(obs)
        self.assertArrayEqual((batch_size, 1), val.shape)
        self.assertEqual(6, len(vf.trainable_variables))

    def test_multi_head_value_nets_wo_nets(self):
        batch_size = 5
        act_dim = 16
        obs_dim = 60
        n_heads = 2
        vf  = ub_nets.MultiHeadValueNets(n_heads=n_heads)
        obs = tf.zeros((batch_size, obs_dim), dtype=tf.float32)
        val = vf(obs)
        self.assertArrayEqual((n_heads, batch_size, 1), val.shape)
        self.assertEqual(2*n_heads, len(vf.trainable_variables))
        # test first head
        val = vf[0](obs)
        self.assertArrayEqual((batch_size, 1), val.shape)
        # test second head
        val = vf[1](obs)
        self.assertArrayEqual((batch_size, 1), val.shape)
    
    def test_multi_head_value_nets_w_share_net(self):
        batch_size = 5
        act_dim = 16
        obs_dim = 60
        n_heads = 2
        hid1 = 32
        hid2 = 64
        net = ub_nets.MlpNet([hid1, hid2])
        vf  = ub_nets.MultiHeadValueNets(n_heads=n_heads, nets=net)
        obs = tf.zeros((batch_size, obs_dim), dtype=tf.float32)
        val = vf(obs)
        self.assertArrayEqual((n_heads, batch_size, 1), val.shape)
        self.assertEqual(2*n_heads + 4, len(vf.trainable_variables))
        # test first head
        val = vf[0](obs)
        self.assertArrayEqual((batch_size, 1), val.shape)
        # test second head
        val = vf[1](obs)
        self.assertArrayEqual((batch_size, 1), val.shape)

    def test_multi_head_value_nets_w_nets(self):
        batch_size = 5
        act_dim = 16
        obs_dim = 60
        n_heads = 2
        hid1 = 32
        hid2 = 64
        net1 = ub_nets.MlpNet([hid1, hid2])
        net2 = ub_nets.MlpNet([hid1])
        vf  = ub_nets.MultiHeadValueNets(n_heads=n_heads, nets=[net1, net2])
        obs = tf.zeros((batch_size, obs_dim), dtype=tf.float32)
        val = vf(obs)
        self.assertArrayEqual((n_heads, batch_size, 1), val.shape)
        self.assertEqual(2*n_heads + 4 + 2, len(vf.trainable_variables))
        # test first head
        val = vf[0](obs)
        self.assertArrayEqual((batch_size, 1), val.shape)
        # test second head
        val = vf[1](obs)
        self.assertArrayEqual((batch_size, 1), val.shape)

    def test_categorical_policy_net_exception(self):
        space = gym.spaces.Box(low=0., high=1., shape=(3,), dtype=np.float32)
        with self.assertRaises(ValueError):
            ub_nets.CategoricalPolicyNet(space)

    def test_diag_gaussian_policy_net_exception(self):
        space = gym.spaces.Discrete(6)
        with self.assertRaises(ValueError):
            ub_nets.DiagGaussianPolicyNet(space)

    def test_policy_net_exception(self):
        space = gym.spaces.MultiDiscrete([3, 4])
        with self.assertRaises(ValueError):
            ub_nets.PolicyNet(space)

    def test_multi_head_exception(self):
        batch_size = 5
        act_dim = 16
        obs_dim = 60
        n_heads = 2
        vf  = ub_nets.MultiHeadValueNets(n_heads=n_heads)
        with self.assertRaises(ValueError):
            vf[0] # Model not created
        obs = tf.zeros((batch_size, obs_dim), dtype=tf.float32)
        val = vf(obs)
        vf[0] # Model created
        with self.assertRaises(IndexError):
            vf[2] # Index out of range

        with self.assertRaises(KeyError):
            vf['1']
        with self.assertRaises(KeyError):
            vf[:] # slice

    def test_custom_multi_head_value_nets(self):
        batch_size = 5
        act_dim = 16
        obs_dim = 60
        n_heads = 2
        vf  = CustomMultiHeadValueNets(act_dim, n_heads=n_heads)
        obs = tf.zeros((batch_size, obs_dim), dtype=tf.float32)
        val = vf(obs)
        self.assertArrayEqual((n_heads, batch_size, act_dim), val.shape)
        self.assertEqual(6*n_heads, len(vf.trainable_variables))
        
    def test_diag_gaussian_policy_net_with_state_independent_cov(self):
        batch_size = 5
        act_dim = 16
        obs_dim = 60
        low  = np.ones((act_dim,), dtype=np.float32) * -1.0
        high = np.ones((act_dim,), dtype=np.float32) * 1.0
        action_space = gym.spaces.Box(low=low, high=high)
        pi   = StateIndepDiagGaussianPolicyNet(action_space, squash=False)
        obs  = tf.zeros((batch_size, obs_dim), dtype=tf.float32)
        dist = pi(obs)

        self.assertTrue(isinstance(dist, ub_prob.MultiNormal))
        act = dist.sample()
        self.assertArrayEqual((batch_size, act_dim), act.shape)
        self.assertEqual(3, len(pi.trainable_variables))