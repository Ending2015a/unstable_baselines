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

    def get_model(self):
        return tf.keras.Sequential([
            tf.keras.layers.Dense(64),
            ub_patch.ReLU(),
            tf.keras.layers.Dense(64),
            ub_patch.ReLU(),
            tf.keras.layers.Dense(self.output_dim)
        ])

class StateIndepDiagGaussianPolicyNet(ub_nets.DiagGaussianPolicyNet):
    def get_logstd_model(self):
        return ub_nets.Constant(
            tf.Variable(np.zeros((self.action_dims,), 
                        dtype=np.float32))
        )

class TestNetsModule(TestCase):
    '''Test unstable_baselines.nets module

    Class list:
    [-] Identity
    [x] Constant
    [-] MlpNet
    [-] NatureCnn
    [x] CategoricalPolicyNet
    [x] DiagGaussianPolicyNet
    [x] PolicyNet
    [x] ValueNet
    [x] MultiHeadValueNets
    '''
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
        self.assertArrayEqual(act.shape, (batch_size,))

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
        self.assertArrayEqual(act.shape, (batch_size, act_dim))

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
        self.assertArrayEqual(act.shape, (batch_size, act_dim))

    def test_policy_net(self):
        batch_size = 5
        act_dim = 16
        obs_dim = 60
        action_space = gym.spaces.Discrete(act_dim)
        pi   = ub_nets.PolicyNet(action_space)
        obs  = tf.zeros((batch_size, obs_dim), dtype=tf.float32)
        dist = pi(obs)
        self.assertTrue(isinstance(dist, ub_prob.Categorical))

        low = np.ones((act_dim,), dtype=np.float32) * -1.0
        high = np.ones((act_dim,), dtype=np.float32) * 1.0
        action_space = gym.spaces.Box(low=low, high=high)
        pi   = ub_nets.DiagGaussianPolicyNet(action_space, squash=False)
        obs = tf.zeros((batch_size, obs_dim), dtype=tf.float32)
        dist = pi(obs)
        self.assertTrue(isinstance(dist, ub_prob.MultiNormal))

        pi   = ub_nets.DiagGaussianPolicyNet(action_space, squash=True)
        obs = tf.zeros((batch_size, obs_dim), dtype=tf.float32)
        dist = pi(obs)
        self.assertTrue(isinstance(dist, ub_prob.Tanh))
        self.assertTrue(isinstance(dist.dist, ub_prob.MultiNormal))

    def test_value_net(self):
        batch_size = 5
        act_dim = 16
        obs_dim = 60
        vi  = ub_nets.ValueNet()
        obs = tf.zeros((batch_size, obs_dim), dtype=tf.float32)
        val = vi(obs)
        self.assertArrayEqual(val.shape, (batch_size, 1))

    def test_multi_head_value_nets(self):
        batch_size = 5
        act_dim = 16
        obs_dim = 60
        n_heads = 2
        vi  = ub_nets.MultiHeadValueNets(n_heads=n_heads)
        obs = tf.zeros((batch_size, obs_dim), dtype=tf.float32)
        val = vi(obs)
        self.assertArrayEqual(val.shape, (n_heads, batch_size, 1))

    def test_custom_multi_head_value_nets(self):
        batch_size = 5
        act_dim = 16
        obs_dim = 60
        n_heads = 2
        vi  = CustomMultiHeadValueNets(act_dim, n_heads=n_heads)
        obs = tf.zeros((batch_size, obs_dim), dtype=tf.float32)
        val = vi(obs)
        self.assertArrayEqual(val.shape, (n_heads, batch_size, act_dim))
        
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
        self.assertArrayEqual(act.shape, (batch_size, act_dim))