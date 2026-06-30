"""``CodeWorkSpaceAgent`` — the sandbox a coding agent writes & tests code in.

Where an :class:`~atrium.agents.inference_agent.InferenceAgent` produces *tokens*,
a ``CodeWorkSpaceAgent`` produces *side effects on a code tree*: it clones a repo,
stages file edits, runs builds/tests, and — when asked — commits and pushes the
result to GitHub and opens a pull request. It is the hands of a coding agent.

Two design points the rest of this package turns on:

* **Inherited container, inherited tooling.** Every workspace runs from the *Base
  Docker Image* (:mod:`atrium.agents.code_workspace_agent.sandbox`), which uses a
  Docker multi-stage build to guarantee the minimum push toolchain — ``git`` and
  the GitHub CLI ``gh`` — is always present. Language-specific workspaces derive
  from that base both as a Docker image (``FROM codeworkspace_base``) and as a
  Python subclass (e.g. :class:`PythonCodeWorkspaceAgent`), so a coding agent for
  Python gets a compiler and package manager preinstalled without losing the
  push guarantee.
* **A different security envelope.** Inference/builder agents are WAN-isolated; a
  code workspace cannot be, because pushing to GitHub and installing dependencies
  need public-internet egress. The isolation kept instead — non-root, dropped
  capabilities, no privilege escalation, read-only root, **no Docker socket and
  no GPU** — is shipped in ``policy.yaml`` and re-asserted at construction by
  :meth:`_enforce_workspace_policy`.

A coding task arrives as an A2A :class:`Message` carrying a structured
``DataPart`` (optional repo to clone, file edits to stage, commands to run, and
an optional git push/PR request); the reply reports each step's outcome so the
caller can react. Every step is wrapped in OpenTelemetry spans inherited from
:class:`BaseAgent`.

    BaseAgent → CodeWorkSpaceAgent → PythonCodeWorkspaceAgent
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from dataclasses import fields as dc_fields
from typing import Any, Mapping, Optional

from atrium.agents.code_workspace_agent.sandbox import (
    DEFAULT_TOKEN_ENV,
    WORKSPACE,
    build_sandbox_config,
)
from atrium.core import telemetry as tel
from atrium.core.base_agent import BaseAgent
from atrium.core.errors import PolicyViolationError
from atrium.core.types import ExecutionResult, SandboxConfig, VersionTag
from atrium.protocol import Message, Role, data_part, text_message

__all__ = ["CodeWorkSpaceAgent", "WorkspaceConfig"]

# ---- A2A vocabulary (metadata "kind" + DataPart "status"/"type") ----------- #
KIND_WORKSPACE = "workspace"
STATUS_OK = "ok"
STATUS_ERROR = "error"
RESULT_TYPE = "workspace_result"

#: How much of each step's output (chars) to echo back to the requester.
_LOG_TAIL = 8192

#: Conservative value patterns for fields that become git/gh *arguments*. They
#: are shell-quoted regardless; rejecting a leading ``-`` additionally prevents a
#: value from being mistaken for a command-line option.
_REPO_RE = re.compile(r"^[A-Za-z0-9._:@/+~-]+$")
_REF_RE = re.compile(r"^[A-Za-z0-9._/+-]+$")


@dataclass(slots=True)
class WorkspaceConfig:
    """Host-side configuration for a code workspace (git identity + run limits).

    Kept deliberately small: the *image* decides the toolchain, the
    *SandboxConfig* decides isolation, and this decides only how the agent drives
    git/gh and how long a single command may run.
    """

    git_user_name: str = "Atrium Coding Agent"
    git_user_email: str = "agent@atrium.local"
    default_branch: str = "main"
    #: Name of the env var (set in the sandbox) carrying the GitHub token.
    token_env: str = DEFAULT_TOKEN_ENV
    #: Wall-clock cap for a single in-sandbox command (build/test/git), seconds.
    command_timeout_s: float = 600.0

    @classmethod
    def from_mapping(cls, data: Optional[Mapping[str, Any]]) -> "WorkspaceConfig":
        """Build config from a mapping (e.g. a YAML ``workspace:`` block)."""
        if not data:
            return cls()
        known = {f.name for f in dc_fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(
                f"unknown WorkspaceConfig field(s): {sorted(unknown)}; "
                f"valid keys are {sorted(known)}"
            )
        return cls(**dict(data))


class CodeWorkSpaceAgent(BaseAgent):
    """A sandboxed code workspace driven over A2A.

    Language-agnostic by itself; subclasses specialise it by pointing at a derived
    image (overriding the sandbox factory) and filling in :meth:`setup_commands` /
    :attr:`DEFAULT_TEST_COMMAND`.

    Parameters
    ----------
    agent_id:
        Unique id (also the sandbox name).
    version:
        Agent version; defaults to the package ``__version__`` (drives the image
        tag ``local-registry/<slug>:<version>``).
    config:
        Git identity + run-limit knobs (:class:`WorkspaceConfig`).
    sandbox_config:
        Override the workspace envelope. Defaults to the package
        :func:`~atrium.agents.code_workspace_agent.sandbox.build_sandbox_config`;
        any override is still validated by :meth:`_enforce_workspace_policy`.
    """

    #: Image slug → ``local-registry/codeworkspace_base:<version>``.
    AGENT_SLUG = "codeworkspace_base"

    #: Default test command run when a request asks to test but names no command.
    #: Subclasses set this to their ecosystem's runner (e.g. ``"pytest"``).
    DEFAULT_TEST_COMMAND: Optional[str] = None

    def __init__(
        self,
        agent_id: str,
        version: "str | VersionTag | None" = None,
        *,
        config: Optional[WorkspaceConfig] = None,
        sandbox_config: Optional[SandboxConfig] = None,
    ) -> None:
        # Lazy import keeps the version/sandbox wiring inside the package, so the
        # agent's version and its image tag share one source of truth.
        from atrium.agents.code_workspace_agent import __version__

        version = version or __version__
        sandbox_config = sandbox_config or build_sandbox_config(str(version))
        self.config = config or WorkspaceConfig()
        super().__init__(agent_id, version, sandbox_config)
        self._enforce_workspace_policy()

    # ------------------------------------------------------------------ #
    # Security envelope (re-checked at construction)                     #
    # ------------------------------------------------------------------ #
    def _enforce_workspace_policy(self) -> None:
        """Refuse any sandbox config that breaks the workspace envelope.

        Unlike the inference/builder agents this does *not* require WAN isolation
        (a workspace must reach GitHub), but it still refuses the two privileges a
        code workspace must never hold: a GPU and — critically — a mount of the
        host Docker socket (which would hand an autonomous agent the host daemon).
        """
        if self.sandbox_config.gpu_enabled:
            raise PolicyViolationError(
                f"{type(self).__name__} must not request GPU passthrough "
                "(a code workspace needs none)"
            )
        self.forbid_docker_socket()

    # ------------------------------------------------------------------ #
    # Language hooks (subclass extension points)                         #
    # ------------------------------------------------------------------ #
    def setup_commands(self) -> list[str]:
        """Commands run once after staging, before the request's own commands.

        The base workspace has no ecosystem, so this is empty; language subclasses
        override it to install/sync dependencies (e.g. ``uv sync``).
        """
        return []

    # ------------------------------------------------------------------ #
    # A2A entry point                                                    #
    # ------------------------------------------------------------------ #
    async def handle_task(self, message: Message) -> Message:
        """Run an inbound coding task and report each step's outcome.

        Pipeline: parse + validate → clone (optional) → stage file edits
        (optional) → configure git → setup commands → request commands → commit &
        push + open PR (optional). Stops at the first failing core step and
        returns a structured error; validation errors are returned, not raised.
        """
        request = self.merge_data_parts(message)
        with tel.start_span(
            "workspace.task",
            kind=tel.TOOL,
            attributes={"atrium.agent_id": self.agent_id},
        ):
            try:
                spec = self._parse_request(request)
            except ValueError as exc:
                return self._error_message(message, f"invalid workspace request: {exc}", [])

            await self.start_sandbox()

            steps: list[dict[str, Any]] = []
            project_dir = self._project_dir(spec["repo"])

            # 1) Clone the target repo (if any).
            if spec["repo"]:
                if not self._run_step(
                    steps, "clone",
                    await self.clone(spec["repo"], ref=spec["ref"], dest=project_dir),
                ):
                    return self._error_message(message, "clone failed", steps, failed="clone")

            # 2) Stage file edits into the project tree (if any).
            if spec["files"]:
                staged = await self.write_files_to_sandbox(spec["files"], project_dir)
                if not self._run_step(steps, "stage_files", staged):
                    return self._error_message(message, "staging files failed", steps, failed="stage_files")

            # 3) Configure git identity / gh credentials (best-effort).
            self._run_step(steps, "configure_git", await self.configure_git())

            # 4) Language setup, then the request's own commands (tests/build).
            commands = [*self.setup_commands(), *spec["commands"]]
            for idx, command in enumerate(commands):
                result = await self.run(command, cwd=project_dir)
                if not self._run_step(steps, f"command[{idx}]", result):
                    return self._error_message(
                        message, f"command failed: {command}", steps, failed=f"command[{idx}]"
                    )

            # 5) Commit, push and (optionally) open a PR.
            git = spec["git"]
            if git and git.get("push"):
                pushed = await self.commit_and_push(
                    git["commit_message"], branch=git["branch"], cwd=project_dir
                )
                if not self._run_step(steps, "push", pushed):
                    return self._error_message(message, "push failed", steps, failed="push")
                pr = git.get("pull_request")
                if pr:
                    opened = await self.open_pull_request(
                        pr["title"], pr["body"], base=pr.get("base"), head=git["branch"], cwd=project_dir
                    )
                    if not self._run_step(steps, "pull_request", opened):
                        return self._error_message(
                            message, "opening pull request failed", steps, failed="pull_request"
                        )

            return self._success_message(message, steps, project_dir)

    # ------------------------------------------------------------------ #
    # Workspace operations (also usable directly, not only via A2A)      #
    # ------------------------------------------------------------------ #
    async def run(
        self, command: str, *, cwd: str = WORKSPACE, timeout: Optional[float] = None
    ) -> ExecutionResult:
        """Run ``command`` in ``cwd`` inside the workspace sandbox."""
        scripted = f"cd {shlex.quote(cwd)} && {command}"
        return await self.execute_in_sandbox(
            scripted, timeout=timeout if timeout is not None else self.config.command_timeout_s
        )

    async def configure_git(self) -> ExecutionResult:
        """Set the git identity and wire ``gh`` as the git credential helper.

        ``gh auth setup-git`` makes ``git push`` over HTTPS use the token in
        ``WorkspaceConfig.token_env`` (injected into the sandbox env); it is a
        no-op (swallowed) when no token is present.
        """
        cfg = self.config
        script = (
            f"git config --global user.name {shlex.quote(cfg.git_user_name)} && "
            f"git config --global user.email {shlex.quote(cfg.git_user_email)} && "
            f"git config --global init.defaultBranch {shlex.quote(cfg.default_branch)} && "
            "(gh auth setup-git || true)"
        )
        return await self.run(script)

    async def clone(
        self, repo: str, *, ref: Optional[str] = None, dest: str = WORKSPACE
    ) -> ExecutionResult:
        """Clone ``repo`` into ``dest`` (and check out ``ref`` when given).

        A full URL (``https://``/``git@``) is cloned with ``git``; an
        ``owner/repo`` shorthand goes through ``gh repo clone``.
        """
        repo_q, dest_q = shlex.quote(repo), shlex.quote(dest)
        if "://" in repo or repo.startswith("git@"):
            script = f"git clone {repo_q} {dest_q}"
        else:
            script = f"gh repo clone {repo_q} {dest_q}"
        if ref:
            script += f" && git -C {dest_q} checkout {shlex.quote(ref)}"
        # Clone targets a fresh dir, so run from the workspace root, not `dest`.
        return await self.run(script, cwd=WORKSPACE)

    async def commit_and_push(
        self, commit_message: str, *, branch: str, cwd: str = WORKSPACE
    ) -> ExecutionResult:
        """Create/switch to ``branch``, commit all changes and push to origin."""
        cwd_q = shlex.quote(cwd)
        script = (
            f"git -C {cwd_q} checkout -B {shlex.quote(branch)} && "
            f"git -C {cwd_q} add -A && "
            f"git -C {cwd_q} commit -m {shlex.quote(commit_message)} && "
            f"git -C {cwd_q} push -u origin {shlex.quote(branch)}"
        )
        return await self.run(script, cwd=WORKSPACE)

    async def open_pull_request(
        self,
        title: str,
        body: str,
        *,
        base: Optional[str] = None,
        head: Optional[str] = None,
        cwd: str = WORKSPACE,
    ) -> ExecutionResult:
        """Open a pull request with ``gh pr create`` (prints the PR URL)."""
        argv = ["gh", "pr", "create", "--title", title, "--body", body]
        if base:
            argv += ["--base", base]
        if head:
            argv += ["--head", head]
        return await self.run(" ".join(shlex.quote(a) for a in argv), cwd=cwd)

    # ------------------------------------------------------------------ #
    # Request parsing / validation                                       #
    # ------------------------------------------------------------------ #
    def _parse_request(self, data: Mapping[str, Any]) -> dict[str, Any]:
        """Validate the request, raising ``ValueError`` on any malformed field.

        Returns a normalised spec ``{repo, ref, files, commands, git}``.
        """
        repo = self._coerce_arg(data.get("repo"), _REPO_RE, "repo") if data.get("repo") else None
        ref = self._coerce_arg(data.get("ref"), _REF_RE, "ref") if data.get("ref") else None

        raw_files = data.get("files") or {}
        if not isinstance(raw_files, Mapping):
            raise ValueError("files must be a {filename: content} map")
        files = {f: self.coerce_file_content(f, c) for f, c in raw_files.items()}

        commands = self._coerce_commands(data.get("commands"))
        # A bare {"test": true} runs the subclass default test command.
        if data.get("test") and self.DEFAULT_TEST_COMMAND:
            commands.append(self.DEFAULT_TEST_COMMAND)

        git = self._coerce_git(data.get("git"))

        if not (repo or files or commands or git):
            raise ValueError("empty request: provide at least one of repo/files/commands/git")
        return {"repo": repo, "ref": ref, "files": files, "commands": commands, "git": git}

    @staticmethod
    def _coerce_arg(value: Any, pattern: re.Pattern[str], field: str) -> str:
        """Validate a value destined to become a git/gh argument."""
        if not isinstance(value, str) or not value or value.startswith("-"):
            raise ValueError(f"invalid {field}: {value!r}")
        if not pattern.match(value):
            raise ValueError(f"invalid {field} (disallowed characters): {value!r}")
        return value

    @staticmethod
    def _coerce_commands(value: Any) -> list[str]:
        """Validate the optional list of shell commands to run."""
        if value is None:
            return []
        if not isinstance(value, (list, tuple)):
            raise ValueError("commands must be a list of strings")
        out: list[str] = []
        for cmd in value:
            if not isinstance(cmd, str) or not cmd.strip():
                raise ValueError(f"invalid command: {cmd!r}")
            out.append(cmd)
        return out

    def _coerce_git(self, value: Any) -> Optional[dict[str, Any]]:
        """Validate the optional git push/PR sub-request."""
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise ValueError("git must be a mapping")
        if not value.get("push"):
            return None  # nothing to do unless an explicit push is requested
        branch = self._coerce_arg(value.get("branch"), _REF_RE, "git.branch")
        commit_message = value.get("commit_message")
        if not isinstance(commit_message, str) or not commit_message.strip():
            raise ValueError("git.commit_message is required when push is set")
        out: dict[str, Any] = {"push": True, "branch": branch, "commit_message": commit_message}

        pr = value.get("pull_request")
        if pr is not None:
            if not isinstance(pr, Mapping):
                raise ValueError("git.pull_request must be a mapping")
            title = pr.get("title")
            body = pr.get("body", "")
            if not isinstance(title, str) or not title.strip():
                raise ValueError("git.pull_request.title is required")
            if not isinstance(body, str):
                raise ValueError("git.pull_request.body must be a string")
            base = pr.get("base")
            if base is not None:
                base = self._coerce_arg(base, _REF_RE, "git.pull_request.base")
            out["pull_request"] = {"title": title, "body": body, "base": base}
        return out

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _project_dir(repo: Optional[str]) -> str:
        """Where the project lives in the sandbox: a clone subdir, else /workspace."""
        if not repo:
            return WORKSPACE
        name = repo.rstrip("/").split("/")[-1]
        if name.endswith(".git"):
            name = name[:-4]
        name = re.sub(r"[^A-Za-z0-9._-]", "", name) or "project"
        return f"{WORKSPACE}/{name}"

    @staticmethod
    def _run_step(steps: list[dict[str, Any]], name: str, result: ExecutionResult) -> bool:
        """Record ``result`` as a named step and return whether it succeeded."""
        steps.append(
            {
                "name": name,
                "exit_code": result.exit_code,
                "duration_s": result.duration_s,
                "stdout_tail": (result.stdout or "")[-_LOG_TAIL:],
                "stderr_tail": (result.stderr or "")[-_LOG_TAIL:],
            }
        )
        return result.succeeded

    # ------------------------------------------------------------------ #
    # Reply assembly                                                     #
    # ------------------------------------------------------------------ #
    def _reply(
        self, request: Message, text: str, status: str, payload: Mapping[str, Any]
    ) -> Message:
        return text_message(
            text,
            role=Role.ROLE_AGENT,
            context_id=request.context_id or None,
            task_id=request.task_id or None,
            metadata={"kind": KIND_WORKSPACE, "status": status},
            extra_parts=[data_part({"type": RESULT_TYPE, "status": status, **payload})],
        )

    def _success_message(
        self, request: Message, steps: list[dict[str, Any]], project_dir: str
    ) -> Message:
        return self._reply(
            request,
            f"Workspace task completed ({len(steps)} steps)",
            STATUS_OK,
            {"steps": steps, "project_dir": project_dir},
        )

    def _error_message(
        self,
        request: Message,
        reason: str,
        steps: list[dict[str, Any]],
        *,
        failed: Optional[str] = None,
    ) -> Message:
        payload: dict[str, Any] = {"reason": reason, "steps": steps}
        if failed is not None:
            payload["failed_step"] = failed
        return self._reply(request, f"Workspace task failed: {reason}", STATUS_ERROR, payload)
