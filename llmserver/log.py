from __future__ import absolute_import, print_function, unicode_literals

import getpass
import json
import logging
import platform
from logging import Formatter, StreamHandler
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

TRACE = 31
STATS = 32
monitor_logger = logging.getLogger("monitor")
monitor_logger.propagate = False


def config_log(
    project,
    module,
    log_root=None,
    app_id="",
    level=logging.INFO,
    print_termninal=False,
    enable_monitor=True,
):
    logging.addLevelName(TRACE, "TRACE")
    logging.addLevelName(STATS, "STATS")

    if log_root is None:
        log_root = user_log_path()
    else:
        log_root = Path(log_root)
    (log_root / project).mkdir(parents=True, exist_ok=True)

    log_file = log_root / project / "{module}.log".format(module=module)
    print(log_file)
    handlers = [TimedRotatingFileHandler(log_file, when="D")]
    if print_termninal:
        handlers.append(StreamHandler())
    logging.basicConfig(
        format="[%(asctime)s.%(msecs)03d] [%(levelname)s] [%(process)s] [%(filename)s] [%(lineno)s] [%(message)s]",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=level,
        handlers=handlers,
    )
    if enable_monitor:
        (log_root / project / "monitor").mkdir(parents=True, exist_ok=True)
        monitor_file = log_root / project / "monitor" / "monitor.log"
        monitor_handlers = [TimedRotatingFileHandler(monitor_file, when="D")]
        if print_termninal:
            monitor_handlers.append(StreamHandler())

        formatter = Formatter(
            fmt="[%(asctime)s.%(msecs)03d] [%(levelname)s] [%(trace_id)s] [{app_id}] [%(module_name)s] [%(filename)s] [%(lineno)s] [%(message)s]".format(
                app_id=app_id
            ),
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        for handler in monitor_handlers:
            handler.setFormatter(formatter)
            monitor_logger.addHandler(handler)
        monitor_logger.setLevel(logging.CRITICAL + 1)
        logging.trace = _trace
        logging.stats = _stats
        logging.monitor = _monitor
    return handlers


def user_log_path():
    user = getpass.getuser()
    if not user:
        user = Path.home().name
    if platform.system() != "Windows":
        path: Path = Path("/logs/") / user
        if not path.exists():
            try:
                path.mkdir(exist_ok=True, parents=True)
            except OSError:
                pass
        if path.is_dir():
            return path
    if hasattr(Path, "home"):
        path = Path.home() / "logs"
    else:
        path = Path("logs")
    if not path.exists():
        path.mkdir(parents=True)
    return path


class Monitor:
    __slots__ = ["trace_id", "module_name"]

    def __init__(self, trace_id="", module_name=""):
        self.trace_id = trace_id
        self.module_name = module_name

    def _convert(self, value):
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    def _construct_message(self, **kwargs):
        return json.dumps({k: self._convert(v) for k, v in kwargs.items()}, ensure_ascii=False)

    def trace(self, **kwargs):
        monitor_logger._log(
            TRACE,
            self._construct_message(**kwargs),
            args=(),
            extra={"trace_id": self.trace_id, "module_name": self.module_name},
        )

    def stats(self, **kwargs):
        monitor_logger._log(
            STATS,
            self._construct_message(**kwargs),
            args=(),
            extra={"trace_id": self.trace_id, "module_name": self.module_name},
        )


def _trace(trace_id="", module_name="", **kwargs):
    return Monitor(trace_id, module_name).trace(**kwargs)


def _stats(trace_id="", module_name="", **kwargs):
    return Monitor(trace_id, module_name).stats(**kwargs)


def _monitor(trace_id="", module_name=""):
    return Monitor(trace_id, module_name)


if __name__ == "__main__":
    config_log("hi", "hi", "log", print_termninal=True)
    logging.info("hi")
