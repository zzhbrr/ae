import os
import json
import logging

# ANSI color codes
RESET = "\033[0m"
COLOR_SEQ = {
    'CYAN': "\033[36m",
    'YELLOW': "\033[33m",
    'RED': "\033[31m",
    'GREEN': "\033[32m",
    'BLUE': "\033[34m",
    'MAGENTA': "\033[35m",
}
LEVEL_COLOR = {
    'DEBUG': COLOR_SEQ['BLUE'],
    'INFO': COLOR_SEQ['GREEN'],
    'WARNING': COLOR_SEQ['YELLOW'],
    'ERROR': COLOR_SEQ['RED'],
    'CRITICAL': COLOR_SEQ['RED'],
}

class ColoredFormatter(logging.Formatter):
    def format(self, record):
        # 彩色时间
        asctime = f"{COLOR_SEQ['CYAN']}{self.formatTime(record, self.datefmt)}{RESET}"
        # 彩色级别
        levelname = f"{LEVEL_COLOR.get(record.levelname, '')}{record.levelname}{RESET}"
        # 彩色文件名/行号/函数名
        filename = f"{COLOR_SEQ['MAGENTA']}{record.filename}{RESET}"
        lineno = f"{COLOR_SEQ['MAGENTA']}{record.lineno}{RESET}"
        funcName = f"{COLOR_SEQ['MAGENTA']}{record.funcName}{RESET}"
        # 消息体默认色
        message = record.getMessage()
        # prefix 处理
        prefix = getattr(record, 'prefix', '')
        log_str = f"[{asctime}{prefix}]-{levelname}-[{filename}:{lineno}:{funcName}()] {message}"
        return log_str

def configure_logger(server_args, prefix: str = ""):
    # if SGLANG_LOGGING_CONFIG_PATH := os.getenv("SGLANG_LOGGING_CONFIG_PATH"):
    #     if not os.path.exists(SGLANG_LOGGING_CONFIG_PATH):
    #         raise Exception(
    #             "Setting SGLANG_LOGGING_CONFIG_PATH from env with "
    #             f"{SGLANG_LOGGING_CONFIG_PATH} but it does not exist!"
    #         )
    #     with open(SGLANG_LOGGING_CONFIG_PATH, encoding="utf-8") as file:
    #         custom_config = json.loads(file.read())
    #     logging.config.dictConfig(custom_config)
    #     return
    # format = f"[%(asctime)s{prefix}] %(message)s"
    # format = f"[%(asctime)s.%(msecs)03d{prefix}] %(message)s"
    # format = f"[%(asctime)s{prefix}]-%(levelname)s-%(filename)s:%(lineno)d:%(funcName)s()--%(message)s"
    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter(datefmt="%Y-%m-%d %H:%M:%S"))
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, server_args.log_level.upper()))

    
'''
import os
import json
import logging

def configure_logger(server_args, prefix: str = ""):
    # if SGLANG_LOGGING_CONFIG_PATH := os.getenv("SGLANG_LOGGING_CONFIG_PATH"):
    #     if not os.path.exists(SGLANG_LOGGING_CONFIG_PATH):
    #         raise Exception(
    #             "Setting SGLANG_LOGGING_CONFIG_PATH from env with "
    #             f"{SGLANG_LOGGING_CONFIG_PATH} but it does not exist!"
    #         )
    #     with open(SGLANG_LOGGING_CONFIG_PATH, encoding="utf-8") as file:
    #         custom_config = json.loads(file.read())
    #     logging.config.dictConfig(custom_config)
    #     return
    # format = f"[%(asctime)s{prefix}] %(message)s"
    # format = f"[%(asctime)s.%(msecs)03d{prefix}] %(message)s"
    format = f"[%(asctime)s{prefix}]-%(levelname)s-%(filename)s:%(lineno)d:%(funcName)s()--%(message)s"
    logging.basicConfig(
        level=getattr(logging, server_args.log_level.upper()),
        format=format,
        datefmt="%Y-%m-%d %H:%M:%S",
        # force=True,
    )
'''