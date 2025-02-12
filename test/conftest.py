import os
import sys
import sqlite3
from pathlib import Path
import pytest
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import qInstallMessageHandler
import traceback

from src.Model.Configuration import Configuration


def qt_message_handler(msg_type, context, message):
    print(f"Qt Message - Type: {msg_type}")
    print(f"Context: {context}")
    print(f"Message: {message}")
    print(f"Stack trace: {traceback.format_stack()}")


@pytest.fixture(scope="module", autouse=True)
def init_config(request):
    configuration = Configuration('TestConfig.db')
    db_file_path = Path(os.environ['USER_ONKODICOM_HIDDEN']).joinpath('TestConfig.db')
    configuration.set_db_file_path(db_file_path)
    connection = sqlite3.connect(db_file_path)
    configuration.update_default_directory(Path.cwd().joinpath('test', 'testdata'))

    def tear_down():
        connection.close()
        if os.path.isfile(db_file_path):
            os.remove(db_file_path)

    request.addfinalizer(tear_down)
    return connection


@pytest.fixture(scope="session", autouse=True)
def qapp_auto():
    app = QApplication(sys.argv)
    qInstallMessageHandler(qt_message_handler)
    yield app
    app.processEvents()
    app.quit()


def pytest_configure(config):
    """Configure pytest with settings that may help prevent segfaults"""
    config.option.tb_locals = False  # Disable locals in tracebacks
    config.option.showlocals = False  # Disable showing locals in output


@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Cleanup after each test"""
    yield
    import gc
    gc.collect()
