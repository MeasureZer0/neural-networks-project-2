import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from visualisation.demo_app.app import app

__all__ = ["app"]


if __name__ == "__main__":
    app.run(debug=True)
