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

    def test_unresolved_task_evidence_blocks_farewell_and_disconnect(self):
        for state in ('ready_to_close', 'farewell_complete'):
            decision = LifecycleDecision.from_dict({
                'state': state, 'task_status': 'blocked', 'outcome': 'unsuccessful',
                'open_questions': [{'evidence': 'Provider mentioned another option.',
                                    'question': 'What are the terms of the other option?'}],
                'message': 'Goodbye.', 'reason': 'First option exceeds the limit.'})
            self.assertEqual(decision.action, 'keep')
            self.assertFalse(decision.farewell_spoken)
            self.assertEqual(decision.message, 'What are the terms of the other option?')

    def test_closing_cannot_override_ongoing_task_without_evidence(self):
        with self.assertRaisesRegex(ValueError, 'ongoing'):
            LifecycleDecision.from_dict({
                'state': 'farewell_complete', 'task_status': 'ongoing',
                'outcome': 'other', 'open_questions': [], 'message': '',
                'reason': 'Agent said goodbye although work remains.'})

    def test_pending_handoff_gets_material_question_even_without_repair_text(self):
        decision = LifecycleDecision.from_dict({
            'state': 'ongoing', 'task_status': 'ongoing', 'outcome': 'other',
            'open_questions': [{'evidence': 'Another option was mentioned.',
                                'question': 'What does the other option cost?'}],
            'message': '', 'reason': 'Need its price.'}, closing_pending=True)
        self.assertEqual(decision.action, 'keep')
        self.assertEqual(decision.message, 'What does the other option cost?')

    def test_explicit_stop_overrides_unfinished_task(self):
        decision = LifecycleDecision.from_dict({
            'state': 'disconnect_requested', 'task_status': 'ongoing',
            'outcome': 'other', 'open_questions': [
                {'evidence': 'Price not yet given.', 'question': 'What does it cost?'}],
            'message': '', 'reason': 'The person explicitly asked to disconnect now.'})
        self.assertEqual(decision.action, 'end')

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
