# %% [markdown]
"""
# Autonomous System Demo

This Colab-style notebook demonstrates how to use the key **core** and **api** components of the autonomous codebase generation system as well as the `hash_verifier` utility.

*The file uses `# %%` cell markers so it can be opened directly in Google Colab or VS Code as an interactive notebook without conversion.*
"""

# %%
"""Utility: ensure repository root is on `sys.path` so local packages import correctly"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
print(f"Added project root to PYTHONPATH: {ROOT}")

# %% [markdown]
"""
## Core Components
Below we exercise each core component: **AutonomousAgent**, **CodeGenerator**, **Documenter**, **GitManager**, **TestRunner**, and **Validator**.
"""

# %%
from core.autonomous_agent import create_autonomousagent
from core.code_generator import create_codegenerator
from core.documenter import create_documenter
from core.git_manager import create_gitmanager
from core.test_runner import create_testrunner
from core.validator import create_validator

components = [
    create_autonomousagent(),
    create_codegenerator(),
    create_documenter(),
    create_gitmanager(),
    create_testrunner(),
    create_validator(),
]

for c in components:
    print(f"\n=== {c.__class__.__name__} ===")
    c.initialize()
    result = c.execute()
    print(result)
    c.cleanup()

# %% [markdown]
"""
## API Components
Demonstrate simple request/response cycles for the REST, GraphQL, and WebSocket APIs.
"""

# %%
from api.rest import create_rest, RestRequest
from api.graphql import create_graphql, GraphqlRequest
from api.websocket import create_websocket, WebsocketRequest

rest = create_rest()
rest.initialize()
rest_resp = rest.execute(RestRequest(data={"message": "hello from REST"}))
print("REST response:", rest_resp.to_dict())
rest.cleanup()

graphql = create_graphql()
graphql.initialize()
graph_resp = graphql.execute(GraphqlRequest(data={"query": "{ hello }"}))
print("GraphQL response:", graph_resp.to_dict())
graphql.cleanup()

ws = create_websocket()
ws.initialize()
ws_resp = ws.execute(WebsocketRequest(data={"event": "ping"}))
print("WebSocket response:", ws_resp.to_dict())
ws.cleanup()

# %% [markdown]
"""
## Hash Verifier Utility
Verify that the sample file included with the tests matches its expected SHA-256 digest.
"""

# %%
from hash_verifier import compute_sha256, verify_hashes

sample_data_dir = ROOT / "tests" / "data"
sample_file = sample_data_dir / "sample.txt"
print("Sample.txt SHA-256:", compute_sha256(sample_file))

csv_path = sample_data_dir / "sample_hashes.csv"
print("Verification results:", verify_hashes(csv_path, base_path=sample_data_dir))

# %% [markdown]
"""
---
## Next Steps
Feel free to modify the cells, extend the functionality, or integrate these components into your own experiments. Because the notebook references the local package paths, it should run **as-is** in Google Colab after uploading the entire repository.
"""