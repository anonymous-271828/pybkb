import unittest
import os

from pybkb.bkb import BKB
from pybkb.fusion import fuse, refactor_fuse  # Assume refactor_fuse is also in fusion module.

class FusionTestCase(unittest.TestCase):
    def setUp(self):
        self.wkdir = os.path.dirname(os.path.abspath(__file__))

    def test_pirate_fusion_refactor(self):
        # Repeat of the pirate_fusion test, but using refactor_fuse.
        bkb_paths = [
                'bkbs/fisherman.bkb.json',
                'bkbs/illegal_dumping_ev.bkb.json',
                'bkbs/illegal_fishing_ev.bkb.json',
                'bkbs/pirate.bkb.json',
                'bkbs/tsunami_ev.bkb.json',
                ]
        bkfs = [BKB.load_from_json(os.path.join(self.wkdir, '../', path)) for path in bkb_paths]
        bkb = fuse(bkfs, [1 for _ in range(len(bkfs))], collapse=False)
        rbkb = refactor_fuse(bkfs, [1 for _ in bkfs], collapse=False)
        self.assertTrue(bkb.is_mutex())
        self.assertTrue(rbkb.is_mutex())
        bkb.save_to_json('bkbs/pirate_fused.bkb.json')
        rbkb.save_to_json('bkbs/pirate_refactor_fused.bkb.json')
        self.assertEqual(rbkb, bkb)

    def test_regression_fuse_vs_refactor_fuse(self):
        # A regression test to compare outcomes of fuse and refactor_fuse for the same inputs.
        bkb_paths = ['bkbs/goldfish.bkb.json']*5
        bkfs = [BKB.load_from_json(os.path.join(self.wkdir, '../', path)) for path in bkb_paths]
        srcs = [str(idx) for idx, _ in enumerate(bkb_paths)]
        
        bkb_fused = fuse(bkfs, [1 for _ in bkfs], source_names=srcs, collapse=True)
        bkb_refactored_fused = refactor_fuse(bkfs, [1 for _ in bkfs], source_names=srcs, collapse=True)
        bkb_fused.save_to_json('bkbs/goldfish_fused_collapsed.bkb.json')
        bkb_refactored_fused.save_to_json('bkbs/goldfish_refactor_fused_collapsed.bkb.json')
        self.assertEqual(bkb_fused, bkb_refactored_fused, "Fused BKBs from fuse and refactor_fuse do not match.")

    def test_regression_fuse_vs_refactor_fuse_pirate(self):
        # A regression test to compare outcomes of fuse and refactor_fuse for the same inputs.
        bkb_paths = [
                'bkbs/fisherman.bkb.json',
                'bkbs/illegal_dumping_ev.bkb.json',
                'bkbs/illegal_fishing_ev.bkb.json',
                'bkbs/pirate.bkb.json',
                'bkbs/tsunami_ev.bkb.json',
                ]
        bkfs = [BKB.load_from_json(os.path.join(self.wkdir, '../', path)) for path in bkb_paths]
        srcs = [str(idx) for idx, _ in enumerate(bkb_paths)]
        
        bkb_fused = fuse(bkfs, [1 for _ in bkfs], source_names=srcs, collapse=True)
        bkb_refactored_fused = refactor_fuse(bkfs, [1 for _ in bkfs], source_names=srcs, collapse=True)
        bkb_fused.save_to_json('bkbs/pirate_fused_collapsed.bkb.json')
        bkb_refactored_fused.save_to_json('bkbs/pirate_collapsed_refactor_fused.bkb.json')
        self.assertEqual(bkb_fused, bkb_refactored_fused, "Fused BKBs from fuse and refactor_fuse do not match.")

if __name__ == '__main__':
    unittest.main()
