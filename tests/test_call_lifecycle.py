import json
import unittest
from unittest.mock import Mock, patch

from call_lifecycle import LifecycleDecision, LifecycleReviewer, review_lifecycle


class CallLifecycleTests(unittest.TestCase):
    def test_malformed_or_refused_output_cannot_execute_an_action(self):
        for value in ({}, {'action': 'end', 'outcome': 'completed'},
                      {'action': 'say_goodbye', 'outcome': 'other', 'message': '', 'reason': ''}):
            with self.assertRaises(ValueError):
                LifecycleDecision.from_dict(value)
        response = Mock()
        response.read.return_value = json.dumps({
            'status': 'completed', 'output': [{'content': [{'type': 'refusal', 'refusal': 'No'}]}]
        }).encode()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        with patch('call_lifecycle.urllib.request.urlopen', return_value=response):
            with self.assertRaises(ValueError):
                review_lifecycle('test-key', 'gpt-6-luna', {'history': []})

    def test_reviewer_failure_is_reported_without_hangup_or_farewell(self):
        reviewer = LifecycleReviewer('test-key', 'gpt-6-luna', 'Ask for opening hours.', '')
        with patch('call_lifecycle.review_lifecycle', side_effect=OSError('Unavailable')):
            reviewer.submit(4, 2, [{'speaker': 'agent', 'text': 'When do you close?'}], False)
            revision, input_revision, result = reviewer.results.get(timeout=1)
        self.assertEqual((revision, input_revision), (4, 2))
        self.assertIsInstance(result, OSError)
        reviewer.close()
        reviewer.submit(5, 3, [], True)
        self.assertTrue(reviewer.results.empty())


if __name__ == '__main__':
    unittest.main()
