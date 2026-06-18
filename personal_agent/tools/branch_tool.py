"""
Branch Search Tool cho Banking AI Agent — Tool Layer.

Tool nay tim chi nhanh ngan hang gan nhat dua tren vi tri nguoi dung,
su dung sklearn BallTree voi haversine metric de tinh khoang cach dia ly.

Input chi can "location" (ten dia diem) — tool tu dong geocode thanh toa do
GPS qua Nominatim, roi dung BallTree de tim k chi nhanh gan nhat.

Du lieu branch duoc load truc tiep tu CSV file:
    data/raw/branch_distance/branch_info.csv

Kien truc:
    - BranchSearchArgs(ToolArgsSchema): Pydantic model validate input tu LLM
    - BranchSearchTool(BaseTool): Strategy cu the cho branch geospatial search
    - Su dung sklearn BallTree (metric=haversine) de tim nearest neighbors
    - Su dung OpenStreetMap Nominatim API de geocode dia chi text -> toa do

Luong chay:
    1. LLM goi tool "branch_search" voi args {location, top_k}
    2. BaseTool.safe_run() goi BranchSearchTool.run()
    3. run() validate args -> geocode location thanh toa do
       -> query BallTree -> tra ve top_k chi nhanh gan nhat
    4. Format ket qua thanh text context -> tra ToolResult

Cach dang ky:
    Duoc tu dong dang ky trong registry.py -> _register_default_tools()

Vi du:
    from tools.branch_tool import BranchSearchTool

    tool = BranchSearchTool()
    result = tool.safe_run(location="Ha Noi", top_k=3)
    print(result.context)
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
from pydantic import Field
from sklearn.neighbors import BallTree

from core.exceptions import ToolExecutionError
from core.logger import get_logger

from tools.base import BaseTool, ToolArgsSchema, ToolCategory, ToolResult

logger = get_logger(__name__)


# =====================================================================
# ARGS SCHEMA -- Pydantic model cho input validation
# =====================================================================

class BranchSearchArgs(ToolArgsSchema):
    """
    Input arguments cho BranchSearchTool.

    LLM chi can gui ten dia diem (location).
    Tool se tu dong geocode thanh toa do GPS qua Nominatim.

    Fields:
        location: Ten dia diem dang text (geocode tu dong). BAT BUOC.
        top_k: So chi nhanh gan nhat can tra ve (1-10, mac dinh 3).
    """
    location: str = Field(
        min_length=1,
        max_length=500,
        description=(
            "Ten dia diem cua nguoi dung (vi du: 'Ha Noi', 'Quan 1 TP HCM'). "
            "Se tu dong chuyen thanh toa do GPS."
        ),
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="So chi nhanh gan nhat can tra ve (1-10).",
    )


# =====================================================================
# NOMINATIM GEOCODER -- Convert dia chi text -> toa do GPS
# =====================================================================

# Timeout cho Nominatim API request (giay)
_NOMINATIM_TIMEOUT = 10

# User-Agent header bat buoc theo Nominatim Usage Policy
_NOMINATIM_USER_AGENT = "BankingAgent"


def geocode(address: str) -> tuple[float, float] | None:
    """
    Convert dia chi text thanh toa do GPS (latitude, longitude)
    su dung OpenStreetMap Nominatim API.

    Nominatim la dich vu geocoding mien phi, khong can API key.
    Tuan thu Nominatim Usage Policy: max 1 request/giay, User-Agent bat buoc.

    Args:
        address: Dia chi hoac ten dia diem (vi du: "Ha Noi", "Quan 1 TP HCM").

    Returns:
        Tuple (latitude, longitude) neu tim thay, None neu khong.

    Raises:
        Khong raise exception -- tra None neu geocode that bai.

    Vi du:
        >>> geocode("Ha Noi")
        (21.0285, 105.8542)

        >>> geocode("dia chi khong ton tai xyz123")
        None
    """
    import requests

    try:
        resp = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={
                "q": address,
                "format": "json",
                "limit": 1,
            },
            headers={"User-Agent": _NOMINATIM_USER_AGENT},
            timeout=_NOMINATIM_TIMEOUT,
        )
        resp.raise_for_status()

        results = resp.json()
        if results:
            lat = float(results[0]["lat"])
            lon = float(results[0]["lon"])
            logger.info(
                f"Geocoded '{address}' -> ({lat}, {lon}) "
                f"[display: {results[0].get('display_name', 'N/A')}]"
            )
            return lat, lon

        logger.warning(f"Geocode returned no results for: '{address}'")
        return None

    except requests.RequestException as e:
        logger.error(f"Nominatim API request failed: {e}")
        return None
    except (KeyError, ValueError, IndexError) as e:
        logger.error(f"Failed to parse Nominatim response: {e}")
        return None


# =====================================================================
# BRANCH SEARCH TOOL -- Tool tim chi nhanh gan nhat
# =====================================================================

# Ban kinh Trai Dat (km)
_EARTH_RADIUS_KM = 6371.0


class BranchSearchTool(BaseTool):
    """
    Tool tim chi nhanh ngan hang gan nhat dua tren vi tri nguoi dung.

    Input chi can ten dia diem (location) — tool tu dong geocode thanh
    toa do GPS qua Nominatim, roi dung sklearn BallTree (metric=haversine)
    de tim k chi nhanh gan nhat tu CSV data.

    Du lieu branch duoc load tu CSV va xay dung BallTree 1 lan (lazy init),
    sau do reuse cho cac query tiep theo.

    Attributes:
        name: "branch_search" -- ten tool (LLM dung ten nay de goi).
        description: Mo ta cho LLM biet khi nao nen dung tool.
        category: GEOSPATIAL -- tool lien quan vi tri dia ly.
        args_schema: BranchSearchArgs -- validate input.

    Luong chay chi tiet:
        1. validate_args() -> BranchSearchArgs (location, top_k)
        2. geocode(location) -> (latitude, longitude)
        3. _get_ball_tree() -> BallTree (lazy init tu CSV)
        4. BallTree.query() -> top_k nearest neighbors
        5. _format_results() -> formatted context string
        6. Return ToolResult(context=..., source="branch_search", metadata=...)
    """

    # --- Metadata (override BaseTool) ---------------------------------
    name: ClassVar[str] = "branch_search"
    description: ClassVar[str] = (
        "Tìm chi nhánh ngân hàng gần nhất dựa trên vị trí người dùng. "
        "Truyền tên địa điểm (ví dụ: 'Hà Nội', 'Quận 1 TP HCM'). "
        "Sử dụng khi người dùng hỏi về chi nhánh gần đây, "
        "địa điểm giao dịch, hoặc muốn tìm ngân hàng gần nhất."
    )
    category: ClassVar[ToolCategory] = ToolCategory.GEOSPATIAL
    args_schema: ClassVar[type[ToolArgsSchema]] = BranchSearchArgs
    version: ClassVar[str] = "2.0.0"

    # --- Config -------------------------------------------------------
    # Duong dan tuong doi toi CSV file (tu thu muc personal_agent/)
    _CSV_PATH: ClassVar[str] = os.path.join(
        "data", "raw", "branch_distance", "branch_info.csv"
    )

    # --- Internal state (lazy-initialized) ----------------------------
    _ball_tree: BallTree | None = None
    _branch_df: pd.DataFrame | None = None

    # --- Core logic ---------------------------------------------------

    def run(self, **kwargs) -> ToolResult:
        """
        Thuc thi branch search: geocode location -> query BallTree -> format.

        Args:
            **kwargs: Arguments tu LLM, se duoc validate thanh BranchSearchArgs.
                - location (str): Ten dia diem (geocode tu dong). BAT BUOC.
                - top_k (int, default=3): So chi nhanh gan nhat.

        Returns:
            ToolResult voi context chua danh sach chi nhanh gan nhat.

        Raises:
            ToolValidationError: Input khong hop le (qua validate_args).
            ToolExecutionError: Loi khi geocode, load CSV, hoac query BallTree.
        """
        # -- Step 1: Validate input ------------------------------------
        args = self.validate_args(**kwargs)

        # -- Step 2: Geocode location -> toa do GPS --------------------
        coords = geocode(args.location)

        if coords is None:
            raise ToolExecutionError(
                f"Khong the xac dinh toa do cho dia diem: '{args.location}'. "
                f"Vui long thu lai voi ten dia diem cu the hon "
                f"(vi du: 'Quan Hoan Kiem, Ha Noi').",
                details={
                    "tool_name": self.name,
                    "location": args.location,
                    "error": "Nominatim returned no results",
                },
            )

        user_lat, user_lon = coords

        logger.info(
            f"Branch search: location='{args.location}', "
            f"lat={user_lat}, lon={user_lon}, top_k={args.top_k}"
        )

        # -- Step 3: Load CSV + build BallTree (lazy init) -------------
        ball_tree, branch_df = self._get_ball_tree()

        if branch_df.empty:
            logger.info("Branch search: no branch data found in CSV")
            return ToolResult(
                context="Khong tim thay du lieu chi nhanh ngan hang trong he thong.",
                source=self.name,
                metadata={
                    "latitude": user_lat,
                    "longitude": user_lon,
                    "location": args.location,
                    "n_results": 0,
                },
            )

        # -- Step 4: Query BallTree cho top_k nearest neighbors --------
        # BallTree voi haversine metric yeu cau toa do o dang radian
        user_point = np.array([[math.radians(user_lat), math.radians(user_lon)]])

        # Dam bao top_k khong vuot qua so luong branches
        k = min(args.top_k, len(branch_df))

        distances_rad, indices = ball_tree.query(user_point, k=k)

        # Chuyen khoang cach tu radian -> km (haversine tra ve radian)
        distances_km = distances_rad[0] * _EARTH_RADIUS_KM

        # -- Step 5: Xay dung danh sach ket qua -----------------------
        ranked_branches = []
        for idx, dist_km in zip(indices[0], distances_km):
            row = branch_df.iloc[idx]
            ranked_branches.append({
                "branch_name": row.get("branch_name", "N/A"),
                "branch_address": row.get("branch_address", "N/A"),
                "latitude": row.get("lattitude", 0.0),
                "longitude": row.get("longtitude", 0.0),
                "distance_km": float(dist_km),
            })

        # -- Step 6: Format ket qua thanh text context -----------------
        context = self._format_results(
            ranked_branches, user_lat, user_lon, args.location,
        )

        logger.info(
            f"Branch search: found {len(ranked_branches)} nearest branches "
            f"for location '{args.location}' ({user_lat}, {user_lon})"
        )

        return ToolResult(
            context=context,
            source=self.name,
            metadata={
                "latitude": user_lat,
                "longitude": user_lon,
                "location": args.location,
                "n_results": len(ranked_branches),
                "nearest_distance_km": (
                    round(ranked_branches[0]["distance_km"], 2)
                    if ranked_branches
                    else None
                ),
            },
        )

    # --- Private helper methods ---------------------------------------

    def _get_ball_tree(self) -> tuple[BallTree, pd.DataFrame]:
        """
        Lazy-initialize BallTree tu CSV data.

        Load CSV 1 lan, xay dung BallTree voi haversine metric,
        sau do cache lai de reuse cho cac query tiep theo.

        Returns:
            Tuple (BallTree, DataFrame) — BallTree da build va DataFrame goc.

        Raises:
            ToolExecutionError: Khi khong tim thay CSV hoac parse loi.
        """
        if self._ball_tree is not None and self._branch_df is not None:
            return self._ball_tree, self._branch_df

        try:
            # Resolve duong dan CSV tuyet doi tu vi tri module hien tai
            # personal_agent/tools/branch_tool.py -> personal_agent/
            module_dir = Path(__file__).resolve().parent.parent
            csv_path = module_dir / self._CSV_PATH

            if not csv_path.exists():
                raise FileNotFoundError(
                    f"Branch CSV file not found: {csv_path}"
                )

            logger.info(f"Loading branch data from: {csv_path}")

            # Doc CSV
            df = pd.read_csv(csv_path, encoding="utf-8")

            # Validate cac cot bat buoc
            required_cols = {"branch_name", "branch_address", "lattitude", "longtitude"}
            missing_cols = required_cols - set(df.columns)
            if missing_cols:
                raise ValueError(
                    f"CSV missing required columns: {missing_cols}. "
                    f"Available columns: {list(df.columns)}"
                )

            # Loai bo hang co toa do NaN hoac khong hop le
            df = df.dropna(subset=["lattitude", "longtitude"])
            df["lattitude"] = pd.to_numeric(df["lattitude"], errors="coerce")
            df["longtitude"] = pd.to_numeric(df["longtitude"], errors="coerce")
            df = df.dropna(subset=["lattitude", "longtitude"])

            if df.empty:
                logger.warning("No valid branch coordinates found in CSV")
                self._branch_df = df
                self._ball_tree = None
                return self._ball_tree, self._branch_df

            # Xay dung BallTree voi haversine metric
            # haversine yeu cau toa do o dang radian: [lat_rad, lon_rad]
            coords_rad = np.deg2rad(
                df[["lattitude", "longtitude"]].values
            )

            self._ball_tree = BallTree(coords_rad, metric="haversine")
            self._branch_df = df.reset_index(drop=True)

            logger.info(
                f"BallTree built with {len(self._branch_df)} branches "
                f"(metric=haversine)"
            )

            return self._ball_tree, self._branch_df

        except FileNotFoundError as e:
            raise ToolExecutionError(
                f"Branch data file not found: {e}",
                details={
                    "tool_name": self.name,
                    "csv_path": str(csv_path),
                    "error": str(e),
                },
            ) from e

        except Exception as e:
            raise ToolExecutionError(
                f"Failed to load branch data from CSV: {e}",
                details={
                    "tool_name": self.name,
                    "error": str(e),
                },
            ) from e

    @staticmethod
    def _format_results(
        ranked_branches: list[dict],
        user_lat: float,
        user_lon: float,
        location: str | None = None,
    ) -> str:
        """
        Format danh sach chi nhanh da xep hang thanh text context cho agent.

        Output format:
            Vi tri cua ban: Ha Noi (21.0285, 105.8542)

            === Chi nhanh 1 (cach 1.23 km) ===
            Ten: M&N Bank - Ha Noi
            Dia chi: 123 Pho Hue, Hai Ba Trung, Ha Noi
            Toa do: (21.0123, 105.8456)

            === Chi nhanh 2 (cach 3.45 km) ===
            ...

        Args:
            ranked_branches: Danh sach dicts da sort theo distance.
            user_lat: Vi do GPS cua nguoi dung.
            user_lon: Kinh do GPS cua nguoi dung.
            location: Ten dia diem goc (tu input).

        Returns:
            Formatted text string.
        """
        if not ranked_branches:
            return "Khong tim thay chi nhanh nao phu hop."

        # Header hien thi vi tri user
        if location:
            loc_header = f"Vi tri cua ban: {location} ({user_lat}, {user_lon})"
        else:
            loc_header = f"Vi tri cua ban: ({user_lat}, {user_lon})"

        parts = [loc_header]

        for i, branch in enumerate(ranked_branches):
            distance_str = f"{branch['distance_km']:.2f}"

            header = f"=== Chi nhanh {i + 1} (cach {distance_str} km) ==="

            lines = [
                header,
                f"Ten: {branch['branch_name']}",
                f"Dia chi: {branch['branch_address']}",
                f"Toa do: ({branch['latitude']}, {branch['longitude']})",
            ]

            parts.append("\n".join(lines))

        return "\n\n".join(parts)
