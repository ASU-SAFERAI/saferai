import unittest
import sys
from pathlib import Path
import uuid

sys.path.append(Path(__file__).resolve().parent.parent.parent.as_posix())

from pre_deploy.query_processor.request_dict import RequestDict, copy_request_dict


class TestRequestDict(unittest.TestCase):
    def test_copy_request_dict_changes_run_id(self):
        original = RequestDict(username='tester', metric_name='metric', metric_phase='phase1', model_name='m1', model_provider='prov',
                               run_id=str(uuid.uuid4()))
        copy1 = copy_request_dict(original)
        copy2 = copy_request_dict(original)
        self.assertNotEqual(original.run_id, copy1.run_id)
        self.assertNotEqual(copy1.run_id, copy2.run_id)
        # Non run_id fields identical
        for field in ['username', 'metric_name', 'metric_phase', 'model_name', 'model_provider']:
            self.assertEqual(getattr(original, field), getattr(copy1, field))
            self.assertEqual(getattr(original, field), getattr(copy2, field))


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
