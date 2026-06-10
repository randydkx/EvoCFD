# -*- coding: utf-8 -*-
import os
from datetime import datetime, timedelta
from easydict import EasyDict

# ---------- Color Printing ----------
PStyle = EasyDict({
    'end': '\33[0m',
    'bold': '\33[1m',
    'italic': '\33[3m',
    'underline': '\33[4m',
    'selected': '\33[7m',
    'red': '\33[31m',
    'green': '\33[32m',
    'yellow': '\33[33m',
    'blue': '\33[34m'
})
# ---------- Naive Print Tools ----------
def print_to_logfile(logfile, content, init=False, end='\n'):
    if not isinstance(content, str):
        content = str(content)
    if init:
        with open(logfile, 'w') as f:
            f.write(content + end)
    else:
        with open(logfile, 'a') as f:
            f.write(content + end)

def print_to_console(content, style=None, color=None):
    flag = 0
    if color in PStyle.keys():
        content = f'{PStyle[color]}{content}'
        flag += 1
    if style in PStyle.keys():
        content = f'{PStyle[style]}{content}'
        flag += 1
    if flag > 0:
        content = f'{content}{PStyle.end}'
    print(content, flush=True)

def step_flagging(content):
    print('=================================================')
    print(content, flush=True)
    print('=================================================')

# ---------- Simple Logger ----------
class Logger(object):
    def __init__(self, logging_dir, DEBUG=False, log_all = True):
        # set up logging directory
        self.DEBUG = DEBUG
        self.logging_dir = logging_dir
        self.logfile_path = None
        self.log_all = log_all
        if self.log_all:
            self.logfile_everything_path = None
        if not os.path.exists(self.logging_dir):
            os.makedirs(self.logging_dir, exist_ok=True)
        else:
            print_to_console(f'logging directory \'{self.logging_dir}\' already exists',color='red')

    def set_logfile(self, logfile_name):
        self.logfile_path = f'{self.logging_dir}/{logfile_name}'
        f = open(self.logfile_path, 'a')
        f.close()
        if self.log_all:
            self.logfile_everything_path = f'{self.logging_dir}/' +logfile_name[:-4] + '_all.txt'
            log_all = open(self.logfile_everything_path, 'a')
            log_all.close()

    def debug(self, content, block=False):
        if self.DEBUG:
            assert self.logfile_path is not None
            print_to_logfile(logfile=self.logfile_path, content=content, init=False)
        timestamp = (datetime.now() + timedelta(hours=8)).strftime('%Y-%m-%d %H:%M:%S')
        if not block:
            print_to_console(f'{PStyle.green}{timestamp}{PStyle.end} - | {PStyle.yellow}DEBUG{PStyle.end}    | - {PStyle.yellow}{content}{PStyle.end}')
        else:
            print_to_console(f'{PStyle.green}{timestamp}{PStyle.end} - | {PStyle.yellow}DEBUG{PStyle.end}    | - {PStyle.yellow}\n{content}{PStyle.end}')
        if self.log_all:
            print_to_logfile(logfile=self.logfile_everything_path, content = content,init=False)

    def info(self, content):
        assert self.logfile_path is not None
        print_to_logfile(logfile=self.logfile_path, content=content, init=False)
        timestamp = (datetime.now() + timedelta(hours=8)).strftime('%Y-%m-%d %H:%M:%S')
        print_to_console(f'{PStyle.green}{timestamp}{PStyle.end} - | {PStyle.blue}INFO{PStyle.end}     | - {PStyle.blue}{content}{PStyle.end}')
        if self.log_all:
            print_to_logfile(logfile=self.logfile_everything_path, content = content,init=False)

    def console_style_color(self, content, style = 'bold', color = 'red'):
        print_to_console(content, style, color)

    def console_bold_green(self, content):
        print_to_console(content, 'selected', 'green')


if __name__ == '__main__':
    logger = Logger('log',log_all=False)
    logger.set_logfile('log.txt')
    for i in range(10):
        logger.info(f'this is line {i}')
    logger.debug('this is a debug info')
    logger.console_bold_green('fafd')
    logger.console_style_color('fafasf', 'italic', 'green')