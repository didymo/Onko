from unittest import mock
from unittest.mock import Mock

from PySide6.QtCore import QThreadPool

from src.Model.Worker import Worker


class FakeClass:
    def func_result(self, result):
        pass

    def func_error(self, error):
        pass


# qtbot is a pytest fixture used to test PyQt5. Part of the pytest-qt plugin.
def test_worker_progress_callback(qtbot):
    """
    Testing for the progress_callback parameter being present in the called function when progress_callback=True
    """
    func_to_test = Mock()
    w = Worker(func_to_test, "test", 3, progress_callback=True)

    # This starts the Worker in the threadpool and then blocks the test from progressing until the finished signal is
    # emitted. qtbot is a pytest fixture used to test PyQt5.
    threadpool = QThreadPool()
    with qtbot.waitSignal(w.signals.finished) as blocker:
        threadpool.start(w)

    assert w.fn == func_to_test
    assert w.kwargs['progress_callback'] is not None
    func_to_test.assert_called_with("test", 3, progress_callback=w.kwargs['progress_callback'])


def test_worker_progress_callback_false(qtbot):
    """
    Testing for the progress_callback parameter not being present in the called function when progress_callback=False
    """
    func_to_test = Mock()
    w = Worker(func_to_test, "test", 3, progress_callback=False)

    threadpool = QThreadPool()
    with qtbot.waitSignal(w.signals.finished) as blocker:
        threadpool.start(w)

    assert w.fn == func_to_test
    assert 'progress_callback' not in w.kwargs
    func_to_test.assert_called_with("test", 3)


def test_worker_no_progress_callback(qtbot):
    """
    Testing for the progress_callback parameter not being present in the called function when no progress_callback
    """
    func_to_test = Mock()
    w = Worker(func_to_test, "test", 3)

    threadpool = QThreadPool()
    with qtbot.waitSignal(w.signals.finished) as blocker:
        threadpool.start(w)

    assert w.fn == func_to_test
    assert 'progress_callback' not in w.kwargs
    func_to_test.assert_called_with("test", 3)


def test_worker_result_signal(qtbot, monkeypatch):
    """
    Testing return value of worker's called function through result signal.
    """
    thing = FakeClass()
    thing.func_to_test = Mock(return_value=5, unsafe=True)
    w = Worker(thing.func_to_test, "test", 3)

    with mock.patch.object(FakeClass, 'func_result', wraps=thing.func_result) as mock_func_result:
        w.signals.result.connect(thing.func_result)
        threadpool = QThreadPool()
        with qtbot.waitSignal(w.signals.finished) as blocker:
            threadpool.start(w)

        thing.func_to_test.assert_called_with("test", 3)
        mock_func_result.assert_called_with(5)


def test_worker_error_signal(qtbot):
    """
    Testing return value of worker's called function through result signal.
    """

    thing = FakeClass()
    thing.func_to_test = Mock(side_effect=ValueError())

    w = Worker(thing.func_to_test, "test", 3)

    with mock.patch.object(FakeClass, 'func_error', wraps=thing.func_error):
        w.signals.error.connect(thing.func_error)
        threadpool = QThreadPool()
        with qtbot.waitSignal(w.signals.finished) as blocker:
            threadpool.start(w)

        kall = thing.func_error.call_args
        args, kwargs = kall

        thing.func_to_test.assert_called_with("test", 3)
        assert isinstance(args[0][1], ValueError)
