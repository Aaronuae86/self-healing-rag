"""Static Docker contract tests; no models or Docker daemon are required."""

from __future__ import annotations

import json
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DockerConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
        cls.dockerignore = (PROJECT_ROOT / ".dockerignore").read_text(encoding="utf-8")

    def test_dockerfile_uses_python_312_and_installs_requirements(self) -> None:
        self.assertIn("FROM python:3.12-slim", self.dockerfile)
        self.assertIn("apt-get install --yes --no-install-recommends libgomp1", self.dockerfile)
        self.assertIn("COPY requirements.txt ./requirements.txt", self.dockerfile)
        self.assertIn("python -m pip install -r requirements.txt", self.dockerfile)

    def test_dockerfile_copies_only_required_runtime_sources(self) -> None:
        self.assertNotIn("COPY . ", self.dockerfile)
        self.assertIn("COPY src/api.py ./src/api.py", self.dockerfile)
        self.assertIn("COPY src/rag ./src/rag", self.dockerfile)
        self.assertIn(
            "COPY data/phase1_corpus.json ./data/phase1_corpus.json",
            self.dockerfile,
        )

    def test_dockerfile_exposes_and_binds_port_8000(self) -> None:
        self.assertIn("EXPOSE 8000", self.dockerfile)
        command_line = next(
            line.removeprefix("CMD ")
            for line in self.dockerfile.splitlines()
            if line.startswith("CMD ")
        )
        command = json.loads(command_line)
        self.assertIn("src.api:app", command)
        self.assertEqual(command[command.index("--host") + 1], "0.0.0.0")
        self.assertEqual(command[command.index("--port") + 1], "8000")

    def test_hugging_face_cache_is_external_volume_ready(self) -> None:
        self.assertIn("HF_HOME=/cache/huggingface", self.dockerfile)

    def test_dockerignore_excludes_development_artifacts_but_keeps_corpus(self) -> None:
        for pattern in (
            ".git/",
            ".venv/",
            "**/__pycache__/",
            "*.pyc",
            "*.py[cod]",
            "**/.ipynb_checkpoints/",
            "results/",
            ".cache/",
        ):
            with self.subTest(pattern=pattern):
                self.assertIn(pattern, self.dockerignore)
        self.assertIn("data/*", self.dockerignore)
        self.assertIn("!data/phase1_corpus.json", self.dockerignore)


if __name__ == "__main__":
    unittest.main()
