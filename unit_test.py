import unittest
from gcn.unit_test.test_layer import TestFullyConnectedLayer
from gcn.unit_test.test_msg import TestMessagePassing
from gcn.unit_test.test_gcn import TestGraphConvolution
from gcn.unit_test.test_read_graph import TestReadGraph
from gcn.unit_test.test_batch_msg import TestBatchMessagePassing
from gcn.unit_test.test_batch_graph import TestBatchGraph
from gcn.unit_test.test_batch_policy import TestBatchPolicy
from gcn.unit_test.test_param import TestParams
from environment.unit_test.test_env import TestEnv


if __name__ == '__main__':
    unittest.main()