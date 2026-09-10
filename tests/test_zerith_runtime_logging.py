import logging
import re
import tempfile
import unittest
from pathlib import Path

from zerith.zerith_runtime_logging import configure_runtime_logging


class ZerithRuntimeLoggingTest(unittest.TestCase):
    def test_runtime_log_has_millisecond_timestamp_and_no_duplicate_handler(self):
        root_logger = logging.getLogger()
        original_handlers = list(root_logger.handlers)
        original_formatters = {
            handler: handler.formatter for handler in original_handlers
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "runtime" / "zerith_server.log"
            try:
                first = configure_runtime_logging(log_path)
                second = configure_runtime_logging(log_path)
                logging.info("EVENT=TEST_RUNTIME_LOG value=1")

                matching_handlers = [
                    handler
                    for handler in root_logger.handlers
                    if isinstance(handler, logging.FileHandler)
                    and Path(handler.baseFilename).resolve()
                    == log_path.resolve()
                ]
                self.assertEqual(first, log_path.resolve())
                self.assertEqual(second, log_path.resolve())
                self.assertEqual(len(matching_handlers), 1)
                matching_handlers[0].flush()

                output = log_path.read_text(encoding="utf-8")
                self.assertRegex(
                    output,
                    re.compile(
                        r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3} "
                        r"\[INFO\].*EVENT=TEST_RUNTIME_LOG value=1"
                    ),
                )
            finally:
                for handler in list(root_logger.handlers):
                    if handler not in original_handlers:
                        root_logger.removeHandler(handler)
                        handler.close()
                for handler, formatter in original_formatters.items():
                    handler.setFormatter(formatter)


if __name__ == "__main__":
    unittest.main()
