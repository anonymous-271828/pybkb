import os
import json
import tempfile
import unittest

from pybkb.bkb import BKB
from pybkb.exceptions import BKBNotMutexError, NoINodeError

class TestBKBInitializationAndProperties(unittest.TestCase):
    def setUp(self):
        # Basic setup for the tests, creating an instance of BKB.
        self.bkb = BKB(name="TestBKB", description="A test BKB instance", sparse_array_format='dok')

    def test_initialization(self):
        # Test that the BKB is initialized with the correct properties.
        self.assertEqual(self.bkb.name, "TestBKB")
        self.assertEqual(self.bkb.description, "A test BKB instance")
        self.assertEqual(self.bkb.sparse_array_format, 'dok')

    def test_basic_properties(self):
        # Test the basic properties before adding any nodes.
        self.assertEqual(len(self.bkb.source_inodes), 0)
        self.assertEqual(len(self.bkb.non_source_features), 0)
        self.assertEqual(len(self.bkb.non_source_inodes), 0)

        # Adding I-nodes, including a source I-node for testing.
        self.bkb.add_inode('Feature1', 'State1')
        self.bkb.add_inode('__Source__', 'State2')  # Assuming source nodes are prefixed with '__Source__'

        # Adding S-nodes to test property retrieval.
        self.bkb.add_snode('Feature1', 'State1', prob=0.5)

        # Testing properties after adding nodes.
        self.assertIn(('__Source__', 'State2'), self.bkb.source_inodes)
        self.assertIn('Feature1', self.bkb.non_source_features)
        self.assertIn(('Feature1', 'State1'), self.bkb.non_source_inodes)

    def tearDown(self):
        # Any cleanup code needed after tests have run.
        pass

class TestBKBNodeAdditions(unittest.TestCase):
    def setUp(self):
        # Setup for tests, creating a fresh instance of BKB for each test case.
        self.bkb = BKB(name="NodeAdditionTestBKB")

    def test_add_inode(self):
        # Test adding various I-nodes and checking their presence.
        self.bkb.add_inode('Feature1', 'State1')
        self.bkb.add_inode('Feature2', 'State2')
        
        # Check if the I-nodes were added correctly.
        self.assertIn(('Feature1', 'State1'), self.bkb.inodes)
        self.assertIn(('Feature2', 'State2'), self.bkb.inodes)
        
        # Attempt to add a duplicate I-node and ensure it doesn't create duplicates.
        self.bkb.add_inode('Feature1', 'State1')
        self.assertEqual(self.bkb.inodes.count(('Feature1', 'State1')), 1, "Duplicate I-node was added.")

    def test_add_snode(self):
        # Add I-nodes for the S-node head and tail.
        self.bkb.add_inode('Feature1', 'State1')
        self.bkb.add_inode('Feature2', 'State2')

        # Add an S-node and verify its presence.
        self.bkb.add_snode('Feature1', 'State1', 0.5, [('Feature2', 'State2')])
        self.assertEqual(len(self.bkb.snode_probs), 1)
        self.assertEqual(self.bkb.snode_probs[0], 0.5)

        # Test adding an S-node with an invalid probability (should fail if ignore_prob=False).
        with self.assertRaises(Exception):  # Replace Exception with the specific exception class you expect.
            self.bkb.add_snode('Feature1', 'State1', 1.1, [('Feature2', 'State2')], ignore_prob=False)
        
        # Test adding an S-node with ignore_prob=True with an invalid probability.
        self.bkb.add_snode('Feature1', 'State1', -0.1, [('Feature2', 'State2')], ignore_prob=True)
        self.assertIn(-0.1, self.bkb.snode_probs, "S-node with invalid probability was not added with ignore_prob=True.")

    def tearDown(self):
        # Cleanup if needed
        pass

class TestBKBSnodeInodeRetrieval(unittest.TestCase):
    def setUp(self):
        # Setup for tests, creating a BKB instance and populating it with nodes.
        self.bkb = BKB()
        # Adding I-nodes
        self.bkb.add_inode('Feature1', 'High')
        self.bkb.add_inode('Feature1', 'Low')
        self.bkb.add_inode('Feature2', 'True')
        self.bkb.add_inode('Feature2', 'False')
        # Adding S-nodes
        self.bkb.add_snode('Feature1', 'High', 0.8)
        self.bkb.add_snode('Feature1', 'Low', 0.2, [('Feature2', 'True')])
    
    def test_get_snode_head(self):
        # Testing retrieval of the head for the first S-node.
        head = self.bkb.get_snode_head(0)
        self.assertEqual(head, ('Feature1', 'High'))
    
    def test_get_snode_tail(self):
        # Testing retrieval of the tail for the second S-node.
        tail = self.bkb.get_snode_tail(1)
        self.assertIn(('Feature2', 'True'), tail)

    def test_find_snodes(self):
        # Test finding S-nodes by head.
        snode_indices = self.bkb.find_snodes('Feature1', 'High')
        self.assertIn(0, snode_indices)
        # Test finding S-nodes by head and specific tail.
        snode_indices_tail = self.bkb.find_snodes('Feature1', 'Low', tail_subset=[('Feature2', 'True')])
        self.assertIn(1, snode_indices_tail)
        # Test finding S-nodes with non-existing criteria.
        with self.assertRaises(NoINodeError):
            snode_indices_non_existing = self.bkb.find_snodes('Feature3', 'NonExisting')
            self.assertEqual(len(snode_indices_non_existing), 0)

    def tearDown(self):
        # Cleanup if needed
        pass

class TestBKBEquivalenceAndHash(unittest.TestCase):
    def setUp(self):
        # Setup for tests, creating BKB instances.
        self.bkb1 = BKB(name="EqualityTestBKB")
        self.bkb2 = BKB(name="EqualityTestBKB")

        # Adding I-nodes and S-nodes
        inode_pairs = [('Feature1', 'State1'), ('Feature2', 'State2')]
        for inode in inode_pairs:
            self.bkb1.add_inode(*inode)
            self.bkb2.add_inode(*inode)
        
        self.bkb1.add_snode('Feature1', 'State1', 0.8)
        self.bkb2.add_snode('Feature1', 'State1', 0.8)

    def test_eq_positive(self):
        # Test equality of two identical BKBs.
        self.assertTrue(self.bkb1 == self.bkb2, "Identical BKBs are not considered equal.")

    def test_eq_negative(self):
        # Modify bkb2 and test inequality.
        self.bkb2.add_snode('Feature2', 'State2', 0.5)
        self.assertFalse(self.bkb1 == self.bkb2, "Different BKBs are considered equal.")

    def test_eq_different_type(self):
        # Compare BKB instance against a non-BKB type.
        self.assertFalse(self.bkb1 == [], "BKB instance considered equal to a non-BKB type.")

    def test_hash_equality(self):
        # Test that identical BKBs produce the same hash value.
        hash1 = hash(self.bkb1)
        hash2 = hash(self.bkb2)
        self.assertEqual(hash1, hash2, "Identical BKBs produce different hash values.")

    def test_hash_inequality(self):
        # Modify bkb2 and test that hash values are different.
        self.bkb2.add_snode('Feature2', 'State2', 0.5)
        hash1 = hash(self.bkb1)
        hash2 = hash(self.bkb2)
        self.assertNotEqual(hash1, hash2, "Different BKBs produce the same hash value.")

    def tearDown(self):
        # Cleanup if needed.
        pass

class TestBKBMutexChecks(unittest.TestCase):
    def setUp(self):
        # Setup for the tests, creating a BKB instance.
        self.bkb = BKB(name="MutexTestBKB")

        # Adding I-nodes necessary for testing mutex conditions.
        self.bkb.add_inode('Feature1', 'High')
        self.bkb.add_inode('Feature1', 'Low')
        self.bkb.add_inode('Feature2', 'True')
        self.bkb.add_inode('Feature2', 'False')

    def test_are_snodes_mutex_with_different_heads(self):
        # Test S-nodes with different heads are considered mutex.
        self.bkb.add_snode('Feature1', 'High', 0.8)
        self.bkb.add_snode('Feature1', 'Low', 0.2)
        self.assertTrue(self.bkb.are_snodes_mutex(0, 1), "S-nodes with different heads are not considered mutex.")

    def test_are_snodes_mutex_with_identical_tails(self):
        # Test S-nodes with identical tails are considered not mutex.
        snode_idx1 = self.bkb.add_snode('Feature1', 'High', 0.8, [('Feature2', 'True')])
        snode_idx2 = self.bkb.add_snode('Feature1', 'High', 0.2, [('Feature2', 'True')])
        self.assertFalse(self.bkb.are_snodes_mutex(snode_idx1, snode_idx2, check_head=False), "S-nodes with identical tails are considered mutex.")

    def test_is_mutex_positive(self):
        # Test a BKB structure that should be mutex.
        self.bkb.add_snode('Feature1', 'High', 0.8, [('Feature2', 'True')])
        self.bkb.add_snode('Feature1', 'Low', 0.2, [('Feature2', 'False')])
        self.assertTrue(self.bkb.is_mutex(), "Mutex BKB is considered not mutex.")

    def test_is_mutex_negative(self):
        # Test a BKB structure that violates mutex conditions.
        self.bkb.add_snode('Feature1', 'High', 0.8)
        self.bkb.add_snode('Feature1', 'High', 0.2)  # This should violate the mutex condition.
        with self.assertRaises(BKBNotMutexError):
            self.bkb.is_mutex()

    def tearDown(self):
        # Cleanup actions after tests.
        pass

class TestBKBUnion(unittest.TestCase):
    def create_bkb_instance(self, name, inodes, snodes):
        """
        Helper function to create a BKB instance with specified I-nodes and S-nodes.
        """
        bkb = BKB(name=name)
        for comp, state in inodes:
            bkb.add_inode(comp, state)
        for head_comp, head_state, prob, tail in snodes:
            bkb.add_snode(head_comp, head_state, prob, tail)
        return bkb

    def test_union_basic(self):
        """
        Tests basic union functionality of two BKBs with no overlapping nodes.
        """
        bkb1 = self.create_bkb_instance("BKB1", [('Feature1', 'State1')], [('Feature1', 'State1', 0.5, [])])
        bkb2 = self.create_bkb_instance("BKB2", [('Feature2', 'State2')], [('Feature2', 'State2', 0.8, [])])
        
        unioned_bkb = BKB.union(bkb1, bkb2)
        
        # Check that the union contains all nodes from both BKBs.
        self.assertIn(('Feature1', 'State1'), unioned_bkb.inodes)
        self.assertIn(('Feature2', 'State2'), unioned_bkb.inodes)
        self.assertEqual(len(unioned_bkb.snode_probs), 2)

    def test_union_with_overlap(self):
        """
        Tests union functionality where BKBs have overlapping I-nodes and S-nodes.
        """
        bkb1 = self.create_bkb_instance("BKB1", [('Feature1', 'State1')], [('Feature1', 'State1', 0.5, [])])
        bkb2 = self.create_bkb_instance("BKB2", [('Feature1', 'State1'), ('Feature2', 'State2')], [('Feature1', 'State1', 0.5, [])])

        unioned_bkb = BKB.union(bkb1, bkb2)
        
        # Ensure no duplicate I-nodes or S-nodes are present in the union.
        self.assertEqual(len(unioned_bkb.inodes), 2)
        self.assertEqual(len(unioned_bkb.snode_probs), 1, "Duplicate S-nodes should not be added.")

    def test_union_result_integrity(self):
        """
        Tests the integrity of the union operation ensuring all original relationships are preserved.
        """
        bkb1 = self.create_bkb_instance("BKB1", [('Feature1', 'State1'), ('Feature2', 'True')], [('Feature1', 'State1', 0.5, [('Feature2', 'True')])])
        bkb2 = self.create_bkb_instance("BKB2", [('Feature2', 'True')], [])

        unioned_bkb = BKB.union(bkb1, bkb2)
        
        # Check that S-nodes maintain correct relationships in the unioned BKB.
        self.assertTrue(any(unioned_bkb.get_snode_tail(idx) == [('Feature2', 'True')] for idx in range(len(unioned_bkb.snode_probs))))

    def tearDown(self):
        # Cleanup actions post-tests.
        pass


class TestBKBSerialization(unittest.TestCase):
    def setUp(self):
        # Setup for tests, creating a BKB instance with some nodes.
        self.bkb = BKB(name="SerializationTestBKB")
        self.bkb.add_inode('Feature1', 'State1')
        self.bkb.add_inode('Feature2', 'State2')
        self.bkb.add_snode('Feature1', 'State1', 0.8, [('Feature2', 'State2')])

        # Create a temporary directory for saving test files.
        self.temp_dir = tempfile.TemporaryDirectory()

    def test_to_dict(self):
        # Test converting a BKB to a dictionary.
        bkb_dict = self.bkb.to_dict()
        self.assertIn('Feature1', bkb_dict["Instantiation Nodes"])
        self.assertEqual(len(bkb_dict["Support Nodes"]), 1)

    def test_json(self):
        # Test converting a BKB to a JSON string and back.
        bkb_json_str = self.bkb.json()
        bkb_dict_from_json = json.loads(bkb_json_str)
        self.assertIn('Feature1', bkb_dict_from_json["Instantiation Nodes"])

    def test_save_and_load(self):
        # Test saving a BKB to a file and loading it back.
        file_path = os.path.join(self.temp_dir.name, 'test_bkb.bkb')
        self.bkb.save(file_path)
        loaded_bkb = BKB.load(file_path)
        self.assertEqual(self.bkb, loaded_bkb)

    def test_save_load_from_json(self):
        # Test saving a BKB to a file and loading it back.
        file_path = os.path.join(self.temp_dir.name, 'test_json_bkb.bkb')
        self.bkb.save_to_json(file_path)
        loaded_bkb = BKB.load_from_json(file_path)
        self.assertEqual(self.bkb, loaded_bkb)
        
    def tearDown(self):
        # Cleanup the temporary directory after tests.
        self.temp_dir.cleanup()

class TestBKBCausalRulesetRetrieval(unittest.TestCase):
    def setUp(self):
        # Setup for tests, creating a BKB instance with some nodes and rulesets.
        self.bkb = BKB(name="CausalRulesetTestBKB")
        self.bkb.add_inode('Feature1', 'High')
        self.bkb.add_inode('Feature1', 'Low')
        self.bkb.add_inode('Feature2', 'True')
        self.bkb.add_inode('Feature2', 'False')
        
        # Adding S-nodes to represent causal rulesets.
        self.bkb.add_snode('Feature1', 'High', 0.8, [('Feature2', 'True')])
        self.bkb.add_snode('Feature1', 'Low', 0.2, [('Feature2', 'False')])

    def test_get_causal_ruleset(self):
        # Retrieve the causal ruleset and perform checks.
        causal_ruleset = self.bkb.get_causal_ruleset()
        
        # Ensure the returned structure is correct.
        self.assertIsInstance(causal_ruleset, dict, "Causal ruleset should be a dictionary.")
        self.assertIn('Feature1', causal_ruleset, "Feature1 should be a key in the causal ruleset.")
        
        # Check if the S-nodes are correctly mapped.
        expected_snode_indices = set(range(2))  # Assuming two S-nodes were added.
        retrieved_snode_indices = set(causal_ruleset['Feature1'])
        self.assertEqual(expected_snode_indices, retrieved_snode_indices, "S-node indices in the causal ruleset do not match expected values.")
        
        # Ensure no unexpected features are included.
        self.assertEqual(len(causal_ruleset), 1, "Causal ruleset contains unexpected features.")

    def tearDown(self):
        # Cleanup actions after tests.
        pass

if __name__ == '__main__':
    unittest.main()
