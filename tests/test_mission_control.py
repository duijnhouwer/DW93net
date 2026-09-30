"""Runner discovery never guesses PIDs or signals unsupported processes."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch, Mock

from scripts.mission_control import mission_control as control


class MissionControlTests(unittest.TestCase):
    def setUp(self):
        timestamp = patch('run_revision.commit_timestamp', return_value='20000101000000')
        timestamp.start()
        self.addCleanup(timestamp.stop)
        revision = patch('run_revision.revision', return_value='c' * 40)
        revision.start()
        self.addCleanup(revision.stop)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.registry = Path(temporary.name)
        override = patch.object(control, '_registry', return_value=self.registry)
        override.start()
        self.addCleanup(override.stop)

    def child(self, source):
        child = subprocess.Popen([sys.executable, '-u', '-c', source, str(self.registry)],
                                 cwd=Path(__file__).resolve().parents[1],
                                 stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, text=True)

        def cleanup():
            if child.poll() is None:
                child.kill()
            child.communicate(timeout=10)
        self.addCleanup(cleanup)
        return child

    def register(self, pid, **changes):
        record = dict(protocol='dw93net-mission-v1', pid=pid, uid=os.getuid(),
                      ready=True, phase='training', checkpoint=str(self.registry / 'latest.pt'),
                      identity=control._process_identity(pid)) | changes
        (self.registry / 'latest.pt').touch()
        (self.registry / 'runner-control.json').write_text(json.dumps(record))

    def test_absent_runner_does_not_signal(self):
        with patch.object(control.signal, 'pidfd_send_signal') as send:
            with self.assertRaisesRegex(RuntimeError, 'No ready training runner'):
                control.request_finish_training()
            send.assert_not_called()

    def test_real_runner_is_found_signaled_and_unregistered(self):
        child = self.child('''
import sys
from pathlib import Path
from scripts.mission_control import mission_control as control
control._registry = lambda: Path(sys.argv[1])
checkpoint = Path(sys.argv[1]) / 'latest.pt'
checkpoint.touch()
with control.RunnerControl(None, 'recipe.toml', checkpoint) as finish:
    print('READY', flush=True)
    sys.stdin.readline()
    print('REQUESTED=' + str(finish.pending), flush=True)
    print('STOPPED=' + str(finish.stop_requested), flush=True)
''')
        for _ in range(3):
            line = child.stdout.readline().strip()
            if line == 'READY':
                break
        self.assertEqual(line, 'READY')
        self.assertEqual(control.request_finish_training(), child.pid)
        self.assertEqual(control.request_finish_training(), child.pid)
        self.assertEqual(control.request_stop(), child.pid)
        output, errors = child.communicate('\n', timeout=10)
        self.assertEqual(child.returncode, 0, errors)
        self.assertIn('REQUESTED=True', output)
        self.assertIn('STOPPED=True', output)
        self.assertFalse((self.registry / 'runner-control.json').exists())
        with self.assertRaisesRegex(RuntimeError, 'No ready training runner'):
            control.request_finish_training()

    def test_stale_identity_never_signals_reused_pid(self):
        self.register(os.getpid(), identity={'start_ticks': '0', 'boot_id': 'old'})
        with patch.object(control.signal, 'pidfd_send_signal') as send:
            with self.assertRaisesRegex(RuntimeError, 'No ready training runner'):
                control.request_finish_training()
            send.assert_not_called()

    def test_unsupported_process_and_dead_runner_are_rejected(self):
        child = self.child("import sys; print('READY', flush=True); sys.stdin.readline()")
        self.assertEqual(child.stdout.readline().strip(), 'READY')
        self.register(child.pid)
        with self.assertRaisesRegex(RuntimeError, 'no command handler'):
            control.request_finish_training()
        self.assertIsNone(child.poll())
        child.communicate('\n', timeout=10)
        with self.assertRaisesRegex(RuntimeError, 'No ready training runner'):
            control.request_finish_training()

    def test_menu_active_stop_evaluate_refresh_and_exit(self):
        record = dict(recipe='/repo/recipe/test.toml', phase='duffy_wurtz_1993',
                      pid=123, identity={'start_ticks': '123'}, global_epoch=12)
        output = []
        with patch.object(control, 'active_runner', return_value=record) as active, \
                patch.object(control, '_send_command') as send, \
                patch.object(control, 'launch_runner') as launch:
            choices = iter(['2', '3', '4', 'q'])
            control.mission_control(input_fn=lambda _: next(choices), output=output.append)
        self.assertEqual(active.call_count, 4)
        self.assertEqual([c.args for c in send.call_args_list],
                         [('stop', record['identity']), ('evaluate', record['identity'])])
        launch.assert_not_called()
        self.assertTrue(any('duffy_wurtz_1993' in line and 'test' in line for line in output))
        self.assertTrue(any('2: Stop' in line for line in output))

    def test_menu_start_resume_and_idle_evaluation(self):
        root = self.registry
        (root / 'recipe').mkdir()
        (root / 'recipe/test.toml').write_text('[network]\noutput_path = "' + str(root) + '"\n')
        for existed, choice, mode, label in [(False, '2', 'train', 'Start'),
                                              (True, '2', 'train', 'Resume training'),
                                              (True, '3', 'evaluate', 'Resume training')]:
            if existed:
                (root / ('test-20000101000000-' + 'c' * 40)).mkdir(exist_ok=True)
                (root / ('test-20000101000000-' + 'c' * 40) / 'run.json').touch()
            output = []
            child = Mock()
            child.poll.return_value = None
            with patch.object(control, 'active_runner', return_value=None), \
                    patch.object(control, 'launch_runner', return_value=child) as launch:
                choices = iter([choice, 'Q'])
                control.mission_control(root=root, input_fn=lambda _: next(choices), output=output.append)
            launch.assert_called_once_with(mode, root=root)
            child.terminate.assert_not_called()
            self.assertTrue(any('2: ' + label in line for line in output))

    def test_action_failure_refreshes_the_full_menu(self):
        record = dict(recipe='test.toml', phase='training', pid=123, identity={})
        output = []
        choices = iter(['3', 'Q'])
        with patch.object(control, 'active_runner', return_value=record), \
                patch.object(control, '_send_command', side_effect=RuntimeError('No checkpoint')):
            control.mission_control(input_fn=lambda _: next(choices), output=output.append)
        self.assertEqual(sum('2: Stop' in line for line in output), 2)
        self.assertTrue(any('No checkpoint' in line for line in output))

    def test_menu_reports_selected_run_without_launching_or_signaling(self):
        root = self.registry
        (root / 'recipe').mkdir()
        (root / 'recipe/test.toml').write_text('[network]\noutput_path = "' + str(root) + '"\n')
        network = root / ('test-20000101000000-' + 'c' * 40)
        record = dict(recipe=str(root / 'recipe/test.toml'), phase='training', pid=123,
                      identity={}, checkpoint=str(network / 'checkpoints/latest.pt'))
        events = [dict(version=1, event='epoch_completed', run_id=str(path / 'checkpoints'),
                       global_epoch=epoch, training_duration_s=120, epoch_wall_duration_s=150)
                  for path, epoch in [(network, 7), (root / 'other', 999)]]
        network.mkdir()
        log = network / 'training_log.jsonl'
        log.write_text(''.join(json.dumps(event) + '\n' for event in events))
        original = log.read_bytes()
        for current in (None, record):
            output = []
            choices = iter(['1', 'Q'])
            with patch.object(control, 'active_runner', return_value=current), \
                    patch.object(control, '_send_command') as send, \
                    patch.object(control, 'launch_runner') as launch, \
                    patch.object(control, 'show_git_status'), \
                    patch('subprocess.run', return_value=Mock(returncode=0)) as report_process:
                control.mission_control(root=root, input_fn=lambda _: next(choices), output=output.append)
            command = report_process.call_args.args[0]
            self.assertEqual(command[:3], [sys.executable, str(Path(control.__file__).parent / 'backend' / 'report_training_log.py'), str(log)])
            self.assertIn(str(network / 'checkpoints'), command)
            self.assertIn('--dashboard-gui', command)
            self.assertNotIn('--output', command)
            self.assertEqual(sum('1: Report training log' in line for line in output), 2)
            send.assert_not_called()
            launch.assert_not_called()
        self.assertEqual(log.read_bytes(), original)

    def test_report_handles_missing_log_without_creating_it(self):
        output = []
        control.show_training_log(self.registry / 'test', output.append)
        self.assertIn('No training log found', output[0])
        self.assertFalse((self.registry / 'test' / 'training_log.jsonl').exists())

    def test_menu_ctrl_c_never_stops_worker(self):
        record = dict(recipe='test.toml', phase='training', pid=123, identity={})
        with patch.object(control, 'active_runner', return_value=record), \
                patch.object(control, '_send_command') as send:
            control.mission_control(input_fn=Mock(side_effect=KeyboardInterrupt), output=lambda _: None)
        send.assert_not_called()

    def test_launch_detaches_logs_and_registers_startup(self):
        from test_training_recipe import recipe_text, module
        root = self.registry
        (root / 'recipe').mkdir()
        (root / 'output').mkdir()
        recipe_path = root / 'recipe/test.toml'
        recipe_path.write_text(recipe_text(str(root)))
        child = Mock(pid=os.getpid())
        with patch('run_revision.committed_revision', return_value='c' * 40), \
                patch.object(control, 'repository_git_status', return_value={'branch': 'test'}), \
                patch('subprocess.Popen', return_value=child) as launch:
            self.assertIs(control.launch_runner(root=root), child)
        args, kwargs = launch.call_args
        self.assertEqual(args[0][-1], str(root / 'scripts/mission_control/backend/run_training_recipe.py'))
        self.assertTrue(kwargs['start_new_session'])
        self.assertEqual(kwargs['stdin'], subprocess.DEVNULL)
        self.assertEqual(kwargs['stderr'], subprocess.STDOUT)
        record = control.active_runner()
        self.assertFalse(record['ready'])
        self.assertEqual(record['phase'], 'starting')
        self.assertEqual(record['recipe'], str(recipe_path))
        network = Path(record['checkpoint']).parent.parent
        self.assertEqual(record['log_path'], str(network / 'mission-control.log'))
        self.assertTrue((network / 'mission-control.log').is_file())
        self.assertEqual(kwargs['stdout'].name, str(network / 'mission-control.log'))
        self.assertEqual(json.loads((network / 'run.json').read_text())['git_commit'], 'c' * 40)
        # The runner must accept the directory pre-created by mission control.
        checkpoints, _ = module._initialize_output(module.load_recipe(recipe_path), 'test',
                                                    'c' * 40, root=root)
        self.assertEqual(checkpoints, network / 'checkpoints')
        self.assertFalse(list((root / 'output').glob('*-mission-control.log')))
        with self.assertRaisesRegex(RuntimeError, 'starting'):
            control.request_stop()

    def test_git_status_and_saved_run_identity(self):
        root = self.registry / 'repo'
        root.mkdir()

        def git(*args):
            return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()

        git('init', '-q', '-b', 'experiment')
        source = root / 'source.py'
        source.write_text('original')
        git('add', '.')
        git('-c', 'user.name=Test', '-c', 'user.email=test@example.com', 'commit', '-qm', 'initial')
        commit = git('rev-parse', 'HEAD')
        self.assertEqual(control.repository_git_status(root),
                         dict(branch='experiment', commit=commit, dirty=False))
        network = self.registry / 'network'
        network.mkdir()
        manifest = network / 'run.json'
        manifest.write_text(json.dumps(dict(git_commit=commit, git_branch='original-branch')))
        output = []
        control.show_git_status(root, network, output.append)
        self.assertIn('Saved run branch: original-branch', output)
        self.assertIn('Saved run commit: ' + commit, output)
        self.assertIn('Current branch: experiment', output)
        self.assertTrue(any('Repository: CLEAN' in line for line in output))
        source.write_text('changed')
        self.assertTrue(control.repository_git_status(root)['dirty'])
        git('add', '.')
        self.assertTrue(control.repository_git_status(root)['dirty'])
        git('reset', '--hard', 'HEAD')
        (root / 'untracked.py').touch()
        self.assertTrue(control.repository_git_status(root)['dirty'])
        manifest.write_text(json.dumps(dict(git_commit='a' * 40)))
        before = manifest.read_bytes()
        output.clear()
        control.show_git_status(root, network, output.append)
        self.assertIn('Saved run branch: not recorded', output)
        self.assertTrue(any('Repository: DIRTY' in line for line in output))
        self.assertTrue(any('Run revision: MISMATCH' in line for line in output))
        self.assertEqual(manifest.read_bytes(), before)
        git('checkout', '--detach', '-q')
        self.assertEqual(control.repository_git_status(root)['branch'], 'detached HEAD')

    def test_new_run_records_original_branch_without_changing_resume_identity(self):
        from test_training_recipe import module, recipe_text
        root = self.registry
        (root / 'output').mkdir()
        path = root / 'recipe.toml'
        path.write_text(recipe_text(str(root)))
        recipe = module.load_recipe(path)
        checkpoints, _ = module._initialize_output(recipe, 'test', 'c' * 40,
                                                   git_branch='original-branch')
        module._initialize_output(recipe, 'test', 'c' * 40, git_branch='another-branch')
        saved = json.loads((checkpoints.parent / 'run.json').read_text())
        self.assertEqual(saved['git_branch'], 'original-branch')
        self.assertEqual(saved['git_commit'], 'c' * 40)

    def test_evaluate_requires_checkpoint_and_checks_displayed_identity(self):
        self.register(os.getpid())
        (self.registry / 'latest.pt').unlink()
        with patch.object(control.signal, 'pidfd_send_signal') as send:
            with self.assertRaisesRegex(RuntimeError, 'No completed checkpoint'):
                control.request_finish_training()
            with self.assertRaisesRegex(RuntimeError, 'runner changed'):
                control._send_command('stop', {'start_ticks': 'old'})
            send.assert_not_called()


if __name__ == '__main__':
    unittest.main()
