#!/usr/bin/env python3
"""Wait for the current train.py in one tmux session, then send the next command."""

# ===== 通常只需修改下面两项 =====
SESSION_NAME = "0"  # 对应 tmux attach -t 0 中的会话名
NEXT_COMMAND = (
    "python main/rl/train.py task=ResDexHand dexhand=artimano side=BiH "
    "headless=true num_envs=4096 learning_rate=2e-4 test=false randomStateInit=true "
    "'dataIndices=[cs_plate_1104125]' early_stop_epochs=10000 actionsMovingAverage=0.4 "
    "experiment=Taco_cs_plate_1104125_212_artimano new_obj=212 correct=true"
)
# ==============================

import argparse
import fcntl
import hashlib
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass


@dataclass
class Process:
    parent: int
    group: int
    foreground_group: int
    started: str
    tty: int


@dataclass
class Pane:
    session: str
    window: str
    target: str
    pid: int
    dead: bool
    in_mode: bool
    tty: str


def read_process(pid):
    try:
        raw = Path("/proc/{}/stat".format(pid)).read_text()
    except (FileNotFoundError, ProcessLookupError):
        return None
    fields = raw.rsplit(")", 1)[1].split()
    if fields[0] in ("Z", "X", "x"):
        return None
    return Process(int(fields[1]), int(fields[2]), int(fields[5]), fields[19], int(fields[4]))


def read_command(pid):
    try:
        raw = Path("/proc/{}/cmdline".format(pid)).read_bytes()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return []
    return [part.decode(errors="replace") for part in raw.split(b"\0") if part]


def process_snapshot():
    processes = {}
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            pid = int(entry.name)
            try:
                process = read_process(pid)
            except PermissionError:
                continue
            if process is not None:
                processes[pid] = process
    return processes


def is_training(args):
    if not args:
        return False
    executable = args[0].rsplit("/", 1)[-1]
    if executable == "train.py":
        return True
    if not (executable.startswith(("python", "pypy"))
            or executable in ("torchrun", "accelerate", "deepspeed")):
        return False
    return (any(arg.rsplit("/", 1)[-1] == "train.py" for arg in args[1:])
            or any(arg == "-m" and args[i + 1].split(".")[-1] == "train"
                   for i, arg in enumerate(args[:-1])))


def is_descendant(pid, ancestor, processes):
    visited = set()
    while pid > 0 and pid not in visited:
        if pid == ancestor:
            return True
        visited.add(pid)
        process = processes.get(pid)
        if process is None:
            return False
        pid = process.parent
    return False


def belongs_to_pane(pid, process, pane, processes):
    shell = processes.get(pane.pid)
    return ((shell is not None and process.tty != 0 and process.tty == shell.tty)
            or is_descendant(pid, pane.pid, processes))


class Tmux:
    def __init__(self, socket=None, server_name=None):
        self.options = (["-S", socket] if socket else
                        ["-L", server_name] if server_name else [])

    def run(self, *args):
        result = subprocess.run(["tmux", *self.options, *args], text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode:
            raise RuntimeError("tmux 执行失败：" + result.stderr.strip())
        return result.stdout.rstrip("\n")

    def panes(self):
        fmt = "\t".join(["#{session_name}", "#{window_index}", "#{pane_id}",
                          "#{pane_pid}", "#{pane_dead}", "#{pane_in_mode}", "#{pane_tty}"])
        result = []
        for line in self.run("list-panes", "-a", "-F", fmt).splitlines():
            session, window, target, pid, dead, in_mode, tty = line.split("\t")
            result.append(Pane(session, window, target, int(pid), dead == "1", in_mode == "1", tty))
        return result


def choose_pane(tmux, session, target=None):
    panes = [p for p in tmux.panes() if p.session == session and not p.dead]
    if not panes:
        raise RuntimeError("未找到会话 {!r} 的可用 pane；请在能执行 tmux attach -t {} 的"
                           "同一服务器/容器、同一用户下运行。".format(session, session))
    if target:
        if not (target.startswith("%") or ":" in target):
            raise RuntimeError("--pane 请填写实际 pane ID（如 %7）或完整目标（如 0:0.0）。")
        pane_id = tmux.run("display-message", "-p", "-t", target, "#{pane_id}")
        panes = [p for p in panes if p.target == pane_id]
        if not panes:
            raise RuntimeError("--pane 指定的目标不属于会话 {!r}。".format(session))
    processes = process_snapshot()
    training = {pid: p for pid, p in processes.items()
                if pid != os.getpid() and is_training(read_command(pid))}
    matches = {pane.target: {pid: p.started for pid, p in training.items()
                            if belongs_to_pane(pid, p, pane, processes)} for pane in panes}
    active = [pane for pane in panes if matches[pane.target]]
    if not target and active:
        panes = active
    if len(panes) != 1:
        raise RuntimeError("会话 {!r} 内匹配到 {} 个候选 pane；请用 --list-panes 查看，"
                           "再用 --pane 指定实际目标。".format(session, len(panes)))
    pane = panes[0]
    watched = matches[pane.target]
    args = read_command(pane.pid)
    if (not args or args[0].rsplit("/", 1)[-1].lstrip("-")
            not in ("sh", "bash", "zsh", "dash", "ksh", "fish")):
        raise RuntimeError("目标 pane 没有可用于接收命令的 shell。")
    shell = read_process(pane.pid)
    if shell is None:
        raise RuntimeError("无法读取目标 shell；请确认脚本与 tmux 在同一 PID 命名空间内。")
    if not watched and shell.group != shell.foreground_group:
        raise RuntimeError("目标 pane 有其他前台任务，但未识别到 train.py。")
    if not watched:
        logging.warning("目标 pane 已空闲，将立即执行接续命令。")
    return pane, shell.started, watched


def lock_pane(tmux, pane):
    socket = tmux.run("display-message", "-p", "#{socket_path}")
    digest = hashlib.sha256((socket + "|" + pane.target).encode()).hexdigest()[:20]
    path = Path(tempfile.gettempdir()) / "queue-next-{}-{}.lock".format(os.getuid(), digest)
    handle = path.open("a")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise RuntimeError("已有监控脚本占用目标 pane，请勿重复启动。")
    return handle


def monitor(tmux, pane, shell_started, watched, command, interval, shell_timeout):
    ended_at = None
    while True:
        current = next((p for p in tmux.panes() if p.target == pane.target
                        and p.session == pane.session), None)
        if (current is None or current.dead or current.pid != pane.pid or current.window != pane.window):
            raise RuntimeError("目标 pane 已关闭、重建或移动，监控停止。")
        shell = read_process(pane.pid)
        if shell is None or shell.started != shell_started:
            raise RuntimeError("目标 shell 已退出或被替换，监控停止。")
        running = []
        for pid, started in watched.items():
            process = read_process(pid)
            if process is not None and process.started == started:
                running.append(pid)
        if not running:
            if ended_at is None:
                ended_at = time.monotonic()
                logging.info("原训练已结束或窗口空闲，等待 shell 就绪。")
            if not current.in_mode and shell.group == shell.foreground_group:
                tmux.run("send-keys", "-t", pane.target, "-l", command)
                tmux.run("send-keys", "-t", pane.target, "Enter")
                logging.info("已向会话 %s 的 %s:%s (%s) 发送接续命令并回车。",
                             pane.session, pane.session, pane.window, pane.target)
                return
            if time.monotonic() - ended_at >= shell_timeout:
                raise RuntimeError("等待 shell 超时，请检查其他前台任务或退出 tmux 复制模式。")
        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", default=SESSION_NAME, help="tmux 会话名，默认使用文件顶部配置")
    parser.add_argument("--command", default=NEXT_COMMAND, help="接续命令，默认使用文件顶部配置")
    parser.add_argument("--pane", help="可选：指定实际 pane ID（如 %%7）或完整目标（如 0:0.0）")
    parser.add_argument("--interval", type=float, default=5, help="检查间隔秒数，默认 5")
    parser.add_argument("--shell-timeout", type=float, default=300, help="等待 shell 的最长秒数，默认 300")
    server = parser.add_mutually_exclusive_group()
    server.add_argument("--socket", help="可选：tmux -S 使用的 socket 路径")
    server.add_argument("--server-name", help="可选：tmux -L 使用的 server 名称")
    parser.add_argument("--list-panes", action="store_true", help="只列出目标会话的 pane")
    parser.add_argument("--dry-run", action="store_true", help="只检查目标和命令，不发送命令")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if sys.platform != "linux" or not Path("/proc/self/stat").exists():
        parser.error("请在训练所在的 Linux 服务器/容器内运行。")
    if shutil.which("tmux") is None:
        parser.error("未找到 tmux。")
    if args.interval <= 0 or args.shell_timeout <= 0:
        parser.error("检查间隔和等待超时必须大于 0。")
    if not args.command.strip() or any(c in args.command for c in "\r\n"):
        parser.error("接续命令必须为非空的单行 shell 命令。")
    tmux = Tmux(args.socket, args.server_name)
    try:
        if args.list_panes:
            for pane in tmux.panes():
                if pane.session == args.session:
                    print("session={!r} window={} pane={} shell_pid={} tty={} dead={}".format(
                        pane.session, pane.window, pane.target, pane.pid, pane.tty, int(pane.dead)))
            return 0
        pane, shell_started, watched = choose_pane(tmux, args.session, args.pane)
        logging.info("监控 PID=%s；会话=%s 窗口=%s pane=%s tty=%s；训练 PID=%s",
                     os.getpid(), pane.session, pane.window, pane.target, pane.tty,
                     ", ".join(map(str, watched)) or "无")
        logging.info("接续命令：%s", args.command)
        if args.dry_run:
            return 0
        with lock_pane(tmux, pane):
            monitor(tmux, pane, shell_started, watched, args.command, args.interval, args.shell_timeout)
        return 0
    except (RuntimeError, OSError, ValueError) as exc:
        logging.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        logging.info("已停止监控。")
        return 130


if __name__ == "__main__":
    sys.exit(main())
