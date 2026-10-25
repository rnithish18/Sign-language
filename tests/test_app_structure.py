"""
test_app_structure.py
---------------------
Structural assertions for app.py:
1. The first module-level Streamlit call in app.py is st.set_page_config.
2. st.set_page_config is called exactly once in app.py.
3. Project modules imported by app.py (src/* and utils/*) make no st.* calls at import time.
"""

import ast
from pathlib import Path
import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
APP_PATH = REPO_ROOT / "app.py"


def test_first_module_level_call_is_set_page_config():
    """Assert that the first module-level Streamlit call in app.py is set_page_config."""
    assert APP_PATH.is_file(), f"app.py not found at {APP_PATH}"
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"), filename=str(APP_PATH))

    first_st_call = None
    first_st_line = None

    for node in tree.body:
        # Check module-level statement for calls to st.<func>
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute):
                if isinstance(sub.func.value, ast.Name) and sub.func.value.id == "st":
                    first_st_call = sub.func.attr
                    first_st_line = node.lineno
                    break
        if first_st_call is not None:
            break

    assert first_st_call == "set_page_config", (
        f"Expected first module-level Streamlit call to be 'set_page_config', "
        f"got '{first_st_call}' at line {first_st_line}"
    )


def test_set_page_config_called_exactly_once():
    """Assert that set_page_config is called exactly once across app.py."""
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"), filename=str(APP_PATH))
    set_page_config_calls = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "st":
                if node.func.attr == "set_page_config":
                    set_page_config_calls.append(node.lineno)

    assert len(set_page_config_calls) == 1, (
        f"Expected exactly 1 call to st.set_page_config, found {len(set_page_config_calls)}: "
        f"at lines {set_page_config_calls}"
    )


def test_imported_project_modules_have_no_import_time_st_calls():
    """Assert that imported modules in src/ and utils/ make no st.* calls at import time."""
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"), filename=str(APP_PATH))
    imported_modules = set()

    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith("src.") or node.module.startswith("utils."):
                imported_modules.add(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("src.") or alias.name.startswith("utils."):
                    imported_modules.add(alias.name)

    assert imported_modules, "No project modules detected in app.py imports"

    import_time_st_calls = []
    for mod_name in sorted(imported_modules):
        mod_path = REPO_ROOT / (mod_name.replace(".", "/") + ".py")
        if not mod_path.is_file():
            continue
        mod_tree = ast.parse(mod_path.read_text(encoding="utf-8"), filename=str(mod_path))
        # Top-level body statements only (outside of def or class)
        for stmt in mod_tree.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                # Also check decorator list on top-level defs
                for dec in stmt.decorator_list:
                    for sub in ast.walk(dec):
                        if isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Name) and sub.value.id == "st":
                            import_time_st_calls.append((mod_name, stmt.lineno, f"decorator @st.{sub.attr}"))
            else:
                for sub in ast.walk(stmt):
                    if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute):
                        if isinstance(sub.func.value, ast.Name) and sub.func.value.id == "st":
                            import_time_st_calls.append((mod_name, stmt.lineno, f"st.{sub.func.attr}()"))

    assert not import_time_st_calls, (
        f"Found import-time Streamlit calls in project modules: {import_time_st_calls}"
    )
