"""
Unit Tests cho Tool Layer — Module 7.

Test coverage cho:
    - ToolResult: data model
    - BaseTool: validate_args, safe_run, to_function_declaration
    - FAQTool: run với mock embedder/vector store
    - BranchSearchTool: run với mock geocode/CSV
    - WebSearchTool: run với mock DuckDuckGo
    - ToolRegistry: register, get, profiles, execute
"""

import math
from dataclasses import dataclass, field
from typing import ClassVar
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
import numpy as np
import pandas as pd

# Thêm path để import modules
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from core.exceptions import (
    ToolError,
    ToolExecutionError,
    ToolNotFoundError,
    ToolValidationError,
)
from tools.base import BaseTool, ToolArgsSchema, ToolCategory, ToolResult
from tools.faq_tool import FAQArgs, FAQTool
from tools.branch_tool import BranchSearchArgs, BranchSearchTool
from tools.web_search_tool import WebSearchArgs, WebSearchTool
from tools.registry import ToolRegistry


# ═══════════════════════════════════════════════════════════════════════
# FIXTURES & HELPERS
# ═══════════════════════════════════════════════════════════════════════

class DummyArgs(ToolArgsSchema):
    """Args schema cho DummyTool test."""
    message: str
    count: int = 1


class DummyTool(BaseTool):
    """Concrete tool dùng cho test BaseTool."""
    name: ClassVar[str] = "dummy"
    description: ClassVar[str] = "A dummy tool for testing."
    args_schema: ClassVar[type[ToolArgsSchema]] = DummyArgs
    category: ClassVar[ToolCategory] = ToolCategory.UTILITY

    def run(self, **kwargs) -> ToolResult:
        args = self.validate_args(**kwargs)
        return ToolResult(
            context=f"{args.message} x{args.count}",
            source=self.name,
            metadata={"count": args.count},
        )


class ErrorTool(BaseTool):
    """Tool luôn raise exception — test safe_run."""
    name: ClassVar[str] = "error_tool"
    description: ClassVar[str] = "Always fails."
    args_schema: ClassVar[type[ToolArgsSchema]] = ToolArgsSchema

    def run(self, **kwargs) -> ToolResult:
        raise RuntimeError("Boom!")


@pytest.fixture(autouse=True)
def clear_registry():
    """Xóa registry trước mỗi test để tránh state leak."""
    ToolRegistry.clear()
    yield
    ToolRegistry.clear()


# ═══════════════════════════════════════════════════════════════════════
# TEST ToolResult
# ═══════════════════════════════════════════════════════════════════════

class TestToolResult:
    def test_success_result(self):
        r = ToolResult(context="hello", source="test")
        assert r.success is True
        assert r.is_error is False
        assert r.error is None

    def test_error_result(self):
        r = ToolResult(context="", source="test", success=False, error="fail")
        assert r.is_error is True
        assert r.error == "fail"

    def test_to_observation_success(self):
        r = ToolResult(context="data here", source="faq")
        assert r.to_observation() == "[faq] data here"

    def test_to_observation_error(self):
        r = ToolResult(context="", source="faq", success=False, error="timeout")
        assert "ERROR" in r.to_observation()
        assert "timeout" in r.to_observation()

    def test_str_truncates_long_context(self):
        r = ToolResult(context="A" * 200, source="t")
        s = str(r)
        assert "..." in s

    def test_metadata_default_empty(self):
        r = ToolResult(context="x", source="s")
        assert r.metadata == {}


# ═══════════════════════════════════════════════════════════════════════
# TEST BaseTool (via DummyTool)
# ═══════════════════════════════════════════════════════════════════════

class TestBaseTool:
    def test_run_happy_path(self):
        tool = DummyTool()
        result = tool.run(message="hi", count=2)
        assert result.success
        assert result.context == "hi x2"
        assert result.source == "dummy"

    def test_validate_args_valid(self):
        tool = DummyTool()
        args = tool.validate_args(message="ok")
        assert args.message == "ok"
        assert args.count == 1  # default

    def test_validate_args_invalid_raises(self):
        tool = DummyTool()
        with pytest.raises(ToolValidationError):
            tool.validate_args()  # missing required 'message'

    def test_safe_run_success(self):
        tool = DummyTool()
        result = tool.safe_run(message="test", count=3)
        assert result.success
        assert result.context == "test x3"

    def test_safe_run_catches_exception(self):
        tool = ErrorTool()
        result = tool.safe_run()
        assert result.success is False
        assert "Boom!" in result.error
        assert result.source == "error_tool"

    def test_safe_run_catches_validation_error(self):
        tool = DummyTool()
        result = tool.safe_run()  # missing 'message'
        assert result.success is False
        assert "validation" in result.error.lower()

    def test_to_function_declaration(self):
        tool = DummyTool()
        decl = tool.to_function_declaration()
        assert decl["name"] == "dummy"
        assert "description" in decl
        assert "parameters" in decl
        assert "message" in decl["parameters"]["properties"]

    def test_get_info(self):
        tool = DummyTool()
        info = tool.get_info()
        assert info["name"] == "dummy"
        assert info["category"] == "utility"

    def test_repr_and_str(self):
        tool = DummyTool()
        assert "dummy" in repr(tool)
        assert "dummy" in str(tool)


# ═══════════════════════════════════════════════════════════════════════
# TEST FAQTool
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class MockQueryResult:
    documents: list = field(default_factory=list)
    metadatas: list = field(default_factory=list)
    distances: list = field(default_factory=list)
    ids: list = field(default_factory=list)

    @property
    def is_empty(self):
        return len(self.documents) == 0


class TestFAQArgs:
    def test_valid_args(self):
        a = FAQArgs(query="test query")
        assert a.query == "test query"
        assert a.n_results == 3
        assert a.domain == "banking_faq"

    def test_empty_query_rejected(self):
        with pytest.raises(Exception):
            FAQArgs(query="")

    def test_n_results_bounds(self):
        with pytest.raises(Exception):
            FAQArgs(query="q", n_results=0)
        with pytest.raises(Exception):
            FAQArgs(query="q", n_results=11)


class TestFAQTool:
    def _make_tool(self):
        tool = FAQTool()
        tool._embedder = MagicMock()
        tool._vector_store = MagicMock()
        return tool

    def test_metadata(self):
        assert FAQTool.name == "faq_search"
        assert FAQTool.category == ToolCategory.RETRIEVAL

    def test_run_with_results(self):
        tool = self._make_tool()
        tool._embedder.embed.return_value = [0.1] * 128
        tool._vector_store.query.return_value = MockQueryResult(
            documents=["Answer 1", "Answer 2"],
            metadatas=[{"source_file": "faq.csv"}, {}],
            distances=[0.1, 0.3],
            ids=["1", "2"],
        )
        result = tool.run(query="lãi suất")
        assert result.success
        assert "Kết quả 1" in result.context
        assert result.metadata["n_results"] == 2

    def test_run_empty_results(self):
        tool = self._make_tool()
        tool._embedder.embed.return_value = [0.1] * 128
        tool._vector_store.query.return_value = MockQueryResult()
        result = tool.run(query="xyz nonsense")
        assert result.success
        assert "Không tìm thấy" in result.context

    def test_run_embed_error(self):
        tool = self._make_tool()
        tool._embedder.embed.side_effect = Exception("API error")
        result = tool.safe_run(query="test")
        assert result.success is False

    def test_run_vector_store_error(self):
        tool = self._make_tool()
        tool._embedder.embed.return_value = [0.1] * 128
        tool._vector_store.query.side_effect = Exception("DB down")
        result = tool.safe_run(query="test")
        assert result.success is False

    def test_validation_error(self):
        tool = self._make_tool()
        result = tool.safe_run(query="", n_results=3)
        assert result.success is False


# ═══════════════════════════════════════════════════════════════════════
# TEST BranchSearchTool
# ═══════════════════════════════════════════════════════════════════════

class TestBranchSearchArgs:
    def test_valid_args(self):
        a = BranchSearchArgs(location="Ha Noi")
        assert a.location == "Ha Noi"
        assert a.top_k == 3

    def test_empty_location_rejected(self):
        with pytest.raises(Exception):
            BranchSearchArgs(location="")


class TestBranchSearchTool:
    def test_metadata(self):
        assert BranchSearchTool.name == "branch_search"
        assert BranchSearchTool.category == ToolCategory.GEOSPATIAL

    @patch("tools.branch_tool.geocode")
    def test_run_happy_path(self, mock_geocode):
        mock_geocode.return_value = (21.0285, 105.8542)
        tool = BranchSearchTool()
        # Mock BallTree + DataFrame
        df = pd.DataFrame({
            "branch_name": ["Branch A", "Branch B"],
            "branch_address": ["Addr A", "Addr B"],
            "lattitude": [21.03, 21.05],
            "longtitude": [105.85, 105.86],
        })
        coords_rad = np.deg2rad(df[["lattitude", "longtitude"]].values)
        from sklearn.neighbors import BallTree
        tree = BallTree(coords_rad, metric="haversine")
        tool._ball_tree = tree
        tool._branch_df = df

        result = tool.run(location="Ha Noi", top_k=2)
        assert result.success
        assert "Chi nhanh 1" in result.context
        assert result.metadata["n_results"] == 2

    @patch("tools.branch_tool.geocode")
    def test_run_geocode_fails(self, mock_geocode):
        mock_geocode.return_value = None
        tool = BranchSearchTool()
        result = tool.safe_run(location="Unknown Place XYZ")
        assert result.success is False

    @patch("tools.branch_tool.geocode")
    def test_run_empty_dataframe(self, mock_geocode):
        mock_geocode.return_value = (10.0, 106.0)
        tool = BranchSearchTool()
        tool._ball_tree = MagicMock()
        tool._branch_df = pd.DataFrame()
        result = tool.run(location="TP HCM")
        assert result.success
        assert "Khong tim thay" in result.context

    def test_format_results_empty(self):
        out = BranchSearchTool._format_results([], 10.0, 106.0)
        assert "Khong tim thay" in out

    def test_format_results_with_data(self):
        branches = [
            {"branch_name": "B1", "branch_address": "A1",
             "latitude": 21.0, "longitude": 105.8, "distance_km": 1.5},
        ]
        out = BranchSearchTool._format_results(branches, 21.0, 105.8, "HN")
        assert "B1" in out
        assert "1.50" in out


# ═══════════════════════════════════════════════════════════════════════
# TEST WebSearchTool
# ═══════════════════════════════════════════════════════════════════════

class TestWebSearchArgs:
    def test_valid_args(self):
        a = WebSearchArgs(query="test")
        assert a.max_results == 5
        assert a.region == "wt-wt"
        assert a.extract_content is True

    def test_empty_query_rejected(self):
        with pytest.raises(Exception):
            WebSearchArgs(query="")


class TestWebSearchTool:
    def test_metadata(self):
        assert WebSearchTool.name == "web_search"
        assert WebSearchTool.category == ToolCategory.WEB

    @patch.object(WebSearchTool, "_search_ddg")
    def test_run_snippet_mode(self, mock_ddg):
        mock_ddg.return_value = [
            {"title": "Result 1", "href": "https://example.com", "body": "snippet 1"},
        ]
        tool = WebSearchTool()
        result = tool.run(query="test query", extract_content=False)
        assert result.success
        assert "Result 1" in result.context
        assert result.metadata["n_results"] == 1

    @patch.object(WebSearchTool, "_extract_contents")
    @patch.object(WebSearchTool, "_search_ddg")
    def test_run_extract_mode(self, mock_ddg, mock_extract):
        raw = [{"title": "T", "href": "https://x.com", "body": "s"}]
        mock_ddg.return_value = raw
        mock_extract.return_value = [
            {**raw[0], "extracted_content": "Full content here"}
        ]
        tool = WebSearchTool()
        result = tool.run(query="test", extract_content=True)
        assert result.success
        assert "Full content" in result.context

    @patch.object(WebSearchTool, "_search_ddg")
    def test_run_no_results(self, mock_ddg):
        mock_ddg.return_value = []
        tool = WebSearchTool()
        result = tool.run(query="abcxyz123", extract_content=False)
        assert result.success
        assert "Không tìm thấy" in result.context

    @patch.object(WebSearchTool, "_search_ddg")
    def test_run_ddg_error(self, mock_ddg):
        mock_ddg.side_effect = ToolExecutionError("DuckDuckGo failed")
        tool = WebSearchTool()
        result = tool.safe_run(query="test", extract_content=False)
        assert result.success is False

    def test_format_results_snippet(self):
        results = [
            {"title": "T1", "href": "http://a.com", "body": "S1"},
        ]
        out = WebSearchTool._format_results(results, "q", extract_mode=False)
        assert "T1" in out
        assert "Tóm tắt: S1" in out

    def test_format_results_extract(self):
        results = [
            {"title": "T1", "href": "http://a.com", "body": "S1",
             "extracted_content": "Full text"},
        ]
        out = WebSearchTool._format_results(results, "q", extract_mode=True)
        assert "Full text" in out

    def test_format_results_empty(self):
        out = WebSearchTool._format_results([], "q")
        assert "Không tìm thấy" in out


# ═══════════════════════════════════════════════════════════════════════
# TEST ToolRegistry
# ═══════════════════════════════════════════════════════════════════════

class TestToolRegistry:
    def test_register_and_get(self):
        tool = DummyTool()
        ToolRegistry.register(tool)
        assert ToolRegistry.has("dummy")
        retrieved = ToolRegistry.get("dummy")
        assert retrieved.name == "dummy"

    def test_get_nonexistent_raises(self):
        with pytest.raises(ToolNotFoundError):
            ToolRegistry.get("nonexistent")

    def test_register_many(self):
        ToolRegistry.register_many([DummyTool(), ErrorTool()])
        assert len(ToolRegistry.available_tools()) == 2

    def test_unregister(self):
        ToolRegistry.register(DummyTool())
        ToolRegistry.unregister("dummy")
        assert not ToolRegistry.has("dummy")

    def test_unregister_nonexistent_raises(self):
        with pytest.raises(ToolNotFoundError):
            ToolRegistry.unregister("nope")

    def test_register_no_name_raises(self):
        class BadTool(BaseTool):
            name = ""
            description = "bad"
            def run(self, **kw):
                return ToolResult(context="", source="")
        with pytest.raises(ToolError):
            ToolRegistry.register(BadTool())

    def test_set_profile_and_get_tools(self):
        ToolRegistry.register(DummyTool())
        ToolRegistry.register(ErrorTool())
        ToolRegistry.set_profile("test_agent", ["dummy"])
        tools = ToolRegistry.get_tools_for_profile("test_agent")
        assert len(tools) == 1
        assert tools[0].name == "dummy"

    def test_set_profile_invalid_tool_raises(self):
        with pytest.raises(ToolNotFoundError):
            ToolRegistry.set_profile("agent", ["nonexistent"])

    def test_grant_and_revoke_tool(self):
        ToolRegistry.register(DummyTool())
        ToolRegistry.register(ErrorTool())
        ToolRegistry.set_profile("p", ["dummy"])
        assert ToolRegistry.is_tool_allowed("p", "dummy")
        assert not ToolRegistry.is_tool_allowed("p", "error_tool")
        ToolRegistry.grant_tool("p", "error_tool")
        assert ToolRegistry.is_tool_allowed("p", "error_tool")
        ToolRegistry.revoke_tool("p", "error_tool")
        assert not ToolRegistry.is_tool_allowed("p", "error_tool")

    def test_is_tool_allowed_unknown_profile(self):
        assert not ToolRegistry.is_tool_allowed("unknown", "dummy")

    def test_get_tools_for_unknown_profile(self):
        tools = ToolRegistry.get_tools_for_profile("unknown")
        assert tools == []

    def test_execute_shortcut(self):
        ToolRegistry.register(DummyTool())
        result = ToolRegistry.execute("dummy", message="hi")
        assert result.success
        assert result.context == "hi x1"

    def test_get_by_category(self):
        ToolRegistry.register(DummyTool())
        tools = ToolRegistry.get_by_category(ToolCategory.UTILITY)
        assert len(tools) == 1

    def test_get_function_declarations(self):
        ToolRegistry.register(DummyTool())
        decls = ToolRegistry.get_function_declarations()
        assert len(decls) == 1
        assert decls[0]["name"] == "dummy"

    def test_get_function_declarations_by_profile(self):
        ToolRegistry.register(DummyTool())
        ToolRegistry.register(ErrorTool())
        ToolRegistry.set_profile("p", ["dummy"])
        decls = ToolRegistry.get_function_declarations("p")
        assert len(decls) == 1

    def test_get_tool_descriptions(self):
        ToolRegistry.register(DummyTool())
        desc = ToolRegistry.get_tool_descriptions()
        assert "AVAILABLE TOOLS" in desc
        assert "dummy" in desc

    def test_get_all(self):
        ToolRegistry.register(DummyTool())
        all_tools = ToolRegistry.get_all()
        assert "dummy" in all_tools

    def test_get_info(self):
        ToolRegistry.register(DummyTool())
        info = ToolRegistry.get_info()
        assert info["total_tools"] == 1

    def test_clear(self):
        ToolRegistry.register(DummyTool())
        ToolRegistry.set_profile("p", ["dummy"])
        ToolRegistry.clear()
        assert ToolRegistry.available_tools() == []
        assert ToolRegistry.available_profiles() == []

    def test_case_insensitive_lookup(self):
        ToolRegistry.register(DummyTool())
        assert ToolRegistry.get("DUMMY").name == "dummy"

    def test_overwrite_warning(self):
        ToolRegistry.register(DummyTool())
        ToolRegistry.register(DummyTool())  # should not raise
        assert len(ToolRegistry.available_tools()) == 1
