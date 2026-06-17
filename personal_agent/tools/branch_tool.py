"""
Branch Search Tool cho Banking AI Agent — Tool Layer.

Tool nay tim chi nhanh ngan hang gan nhat dua tren vi tri nguoi dung,
su dung Haversine formula de tinh khoang cach dia ly.

Ho tro 2 cach xac dinh vi tri:
    1. Toa do GPS truc tiep (latitude, longitude)
    2. Ten dia diem dang text (vi du: "Ha Noi") -> tu geocode qua Nominatim

Kien truc:
    - BranchSearchArgs(ToolArgsSchema): Pydantic model validate input tu LLM
    - BranchSearchTool(BaseTool): Strategy cu the cho branch geospatial search
    - Su dung VectorStore facade tu knowledge_base module (domain="branch_info")
    - Su dung Haversine formula de tinh khoang cach giua 2 toa do GPS
    - Su dung OpenStreetMap Nominatim API de geocode dia chi text -> toa do

Luong chay:
    1. LLM goi tool "branch_search" voi args {location} hoac {latitude, longitude}
    2. BaseTool.safe_run() goi BranchSearchTool.run()
    3. run() validate args -> geocode neu can -> query vector store (domain=branch_info)
       -> tinh distance bang Haversine -> sort -> format top_k results
    4. Format ket qua thanh text context -> tra ToolResult

Cach dang ky:
    Duoc tu dong dang ky trong registry.py -> _register_default_tools()

Vi du:
    from tools.branch_tool import BranchSearchTool

    # Cach 1: Dung ten dia diem (LLM thuong dung cach nay)
    tool = BranchSearchTool()
    result = tool.safe_run(location="Ha Noi", top_k=3)

    # Cach 2: Dung toa do GPS truc tiep
    result = tool.safe_run(latitude=21.0285, longitude=105.8542, top_k=3)

    print(result.context)
"""

from __future__ import annotations

import math
from typing import ClassVar

from pydantic import Field, model_validator

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

    LLM co the gui HOAC:
        - location (str): Ten dia diem (vi du: "Ha Noi", "Quan 1 TP HCM")
          -> Tool se tu geocode thanh toa do GPS qua Nominatim.
        - latitude + longitude (float): Toa do GPS truc tiep.

    It nhat 1 trong 2 cach phai duoc cung cap.

    Fields:
        location: Ten dia diem dang text (geocode tu dong).
        latitude: Vi do GPS cua nguoi dung (-90 den 90).
        longitude: Kinh do GPS cua nguoi dung (-180 den 180).
        top_k: So chi nhanh gan nhat can tra ve (1-10, mac dinh 3).
    """
    location: str | None = Field(
        default=None,
        min_length=1,
        max_length=500,
        description=(
            "Ten dia diem cua nguoi dung (vi du: 'Ha Noi', 'Quan 1 TP HCM'). "
            "Se tu dong chuyen thanh toa do GPS."
        ),
    )
    latitude: float | None = Field(
        default=None,
        ge=-90.0,
        le=90.0,
        description="Vi do GPS cua nguoi dung (latitude). Dung khi da co toa do.",
    )
    longitude: float | None = Field(
        default=None,
        ge=-180.0,
        le=180.0,
        description="Kinh do GPS cua nguoi dung (longitude). Dung khi da co toa do.",
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="So chi nhanh gan nhat can tra ve (1-10).",
    )

    @model_validator(mode="after")
    def validate_location_or_coordinates(self) -> BranchSearchArgs:
        """
        Dam bao it nhat 1 cach xac dinh vi tri duoc cung cap:
            - location (text)
            - latitude + longitude (toa do)

        Neu ca 2 duoc cung cap, uu tien toa do GPS (chinh xac hon).
        """
        has_location = self.location is not None
        has_coords = self.latitude is not None and self.longitude is not None

        if not has_location and not has_coords:
            raise ValueError(
                "Must provide 'location' (place name) "
                "or both 'latitude' + 'longitude' (GPS coordinates)."
            )

        # Canh bao neu chi co 1 trong 2 toa do (thieu 1 cai)
        if (self.latitude is None) != (self.longitude is None):
            raise ValueError(
                "Must provide BOTH 'latitude' AND 'longitude', "
                "cannot provide only one."
            )

        return self


# =====================================================================
# HAVERSINE FORMULA -- Tinh khoang cach giua 2 toa do GPS
# =====================================================================

# Ban kinh Trai Dat (km)
_EARTH_RADIUS_KM = 6371.0


def haversine_distance(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """
    Tinh khoang cach giua 2 diem tren be mat Trai Dat
    su dung Haversine formula.

    Args:
        lat1: Vi do diem 1 (do).
        lon1: Kinh do diem 1 (do).
        lat2: Vi do diem 2 (do).
        lon2: Kinh do diem 2 (do).

    Returns:
        Khoang cach tinh bang km.

    Vi du:
        >>> haversine_distance(21.0285, 105.8542, 21.0278, 105.8342)
        2.11  # ~2.11 km
    """
    # Chuyen tu do sang radian
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    # Haversine formula
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad)
        * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return _EARTH_RADIUS_KM * c


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

class BranchSearchTool(BaseTool):
    """
    Tool tim chi nhanh ngan hang gan nhat dua tren vi tri nguoi dung.

    Ho tro 2 cach xac dinh vi tri:
        1. Ten dia diem (location) -> tu geocode qua Nominatim
        2. Toa do GPS truc tiep (latitude, longitude)

    Truy van toan bo du lieu branch tu vector store (domain=branch_info),
    tinh khoang cach bang Haversine formula, va tra ve top_k chi nhanh
    gan nhat.

    Attributes:
        name: "branch_search" -- ten tool (LLM dung ten nay de goi).
        description: Mo ta cho LLM biet khi nao nen dung tool.
        category: GEOSPATIAL -- tool lien quan vi tri dia ly.
        args_schema: BranchSearchArgs -- validate input.

    Luồng chạy chi tiết:
        1. validate_args() -> BranchSearchArgs (location hoac lat/lon, top_k)
        2. _resolve_coordinates() -> geocode neu can -> (latitude, longitude)
        3. _get_vector_store() -> VectorStore instance (lazy init)
        4. VectorStore.query(domain="branch_info") -> QueryResult (all branches)
        5. _calculate_and_rank() -> list of (branch_info, distance_km)
        6. Sort by distance -> lay top_k
        7. _format_results() -> formatted context string
        8. Return ToolResult(context=..., source="branch_search", metadata=...)
    """

    # --- Metadata (override BaseTool) ---------------------------------
    name: ClassVar[str] = "branch_search"
    description: ClassVar[str] = (
        "Tìm chi nhánh ngân hàng gần nhất dựa trên vị trí người dùng. "
        "Có thể truyền tên địa điểm (ví dụ: 'Hà Nội', 'Quận 1 TP HCM') "
        "hoặc tọa độ GPS (latitude, longitude). "
        "Sử dụng khi người dùng hỏi về chi nhánh gần đây, "
        "địa điểm giao dịch, hoặc muốn tìm ngân hàng gần nhất."
    )
    category: ClassVar[ToolCategory] = ToolCategory.GEOSPATIAL
    args_schema: ClassVar[type[ToolArgsSchema]] = BranchSearchArgs
    version: ClassVar[str] = "1.1.0"

    # --- Config -------------------------------------------------------
    # Domain trong vector store chua du lieu branch
    _BRANCH_DOMAIN: ClassVar[str] = "branch_info"

    # So luong branch toi da lay tu vector store moi lan query
    # (lay du nhieu de sort distance chinh xac)
    _MAX_FETCH: ClassVar[int] = 200

    # --- Internal state (lazy-initialized) ----------------------------
    _vector_store = None

    # --- Core logic ---------------------------------------------------

    def run(self, **kwargs) -> ToolResult:
        """
        Thuc thi branch search: resolve vi tri -> query branches -> sort -> format.

        Args:
            **kwargs: Arguments tu LLM, se duoc validate thanh BranchSearchArgs.
                - location (str, optional): Ten dia diem (geocode tu dong).
                - latitude (float, optional): Vi do GPS.
                - longitude (float, optional): Kinh do GPS.
                - top_k (int, default=3): So chi nhanh gan nhat.

                Phai cung cap location HOAC (latitude + longitude).

        Returns:
            ToolResult voi context chua danh sach chi nhanh gan nhat.

        Raises:
            ToolValidationError: Input khong hop le (qua validate_args).
            ToolExecutionError: Loi khi geocode, query vector store, hoac tinh distance.
        """
        # -- Step 1: Validate input ------------------------------------
        args = self.validate_args(**kwargs)

        # -- Step 2: Resolve toa do (geocode neu can) ------------------
        user_lat, user_lon, resolved_from = self._resolve_coordinates(args)

        logger.info(
            f"Branch search: lat={user_lat}, lon={user_lon}, "
            f"top_k={args.top_k}, resolved_from={resolved_from}"
        )

        # -- Step 3: Query toan bo branches tu vector store ------------
        query_result = self._query_branches()

        # -- Step 4: Xu ly ket qua rong -------------------------------
        if query_result.is_empty:
            logger.info("Branch search: no branch data found in vector store")
            return ToolResult(
                context="Khong tim thay du lieu chi nhanh ngan hang trong he thong.",
                source=self.name,
                metadata={
                    "latitude": user_lat,
                    "longitude": user_lon,
                    "location": args.location,
                    "resolved_from": resolved_from,
                    "n_results": 0,
                },
            )

        # -- Step 5: Tinh khoang cach va sap xep ----------------------
        ranked_branches = self._calculate_and_rank(
            query_result=query_result,
            user_lat=user_lat,
            user_lon=user_lon,
            top_k=args.top_k,
        )

        # -- Step 6: Format ket qua thanh text context -----------------
        context = self._format_results(
            ranked_branches, user_lat, user_lon, args.location,
        )

        logger.info(
            f"Branch search: found {len(ranked_branches)} nearest branches "
            f"for location ({user_lat}, {user_lon})"
        )

        return ToolResult(
            context=context,
            source=self.name,
            metadata={
                "latitude": user_lat,
                "longitude": user_lon,
                "location": args.location,
                "resolved_from": resolved_from,
                "n_results": len(ranked_branches),
                "nearest_distance_km": (
                    round(ranked_branches[0]["distance_km"], 2)
                    if ranked_branches
                    else None
                ),
            },
        )

    # --- Private helper methods ---------------------------------------

    def _resolve_coordinates(
        self,
        args: BranchSearchArgs,
    ) -> tuple[float, float, str]:
        """
        Xac dinh toa do GPS tu args -- dung truc tiep hoac geocode.

        Uu tien: toa do GPS > geocode tu location text.

        Args:
            args: Validated BranchSearchArgs.

        Returns:
            Tuple (latitude, longitude, resolved_from).
            resolved_from: "coordinates" hoac "geocode".

        Raises:
            ToolExecutionError: Khi geocode that bai (location khong tim thay).
        """
        # Uu tien toa do GPS truc tiep (chinh xac hon)
        if args.latitude is not None and args.longitude is not None:
            return args.latitude, args.longitude, "coordinates"

        # Geocode tu ten dia diem
        result = geocode(args.location)

        if result is None:
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

        return result[0], result[1], "geocode"

    def _get_vector_store(self):
        """
        Lazy-initialize VectorStore facade.

        Tao instance lan dau khi can, sau do reuse.
        Tach rieng de de test (mock) va tranh import luc module load.
        """
        if self._vector_store is None:
            from knowledge_base.vector_store import VectorStore
            self._vector_store = VectorStore()
            logger.debug("BranchSearchTool: VectorStore initialized")
        return self._vector_store

    def _query_branches(self):
        """
        Query toan bo branch documents tu vector store.

        Su dung domain filter "branch_info" de chi lay du lieu
        chi nhanh, khong lan FAQ hoac data khac.

        Returns:
            QueryResult tu vector store chua tat ca branches.

        Raises:
            ToolExecutionError: Khi query vector store that bai.
        """
        try:
            store = self._get_vector_store()

            # Lay so luong branch docs thuc te trong store
            branch_count = store.count_by_domain(self._BRANCH_DOMAIN)

            if branch_count == 0:
                logger.warning(
                    f"No documents found in domain '{self._BRANCH_DOMAIN}'"
                )
                from knowledge_base.vector_store import QueryResult
                return QueryResult()

            # Fetch tat ca branches (hoac toi da _MAX_FETCH)
            fetch_count = min(branch_count, self._MAX_FETCH)

            result = store.query(
                query_text="branch",  # Dummy query text de lay tat ca
                n_results=fetch_count,
                domain=self._BRANCH_DOMAIN,
            )
            return result

        except Exception as e:
            raise ToolExecutionError(
                f"Failed to query branch data from vector store: {e}",
                details={
                    "tool_name": self.name,
                    "domain": self._BRANCH_DOMAIN,
                    "error": str(e),
                },
            ) from e

    def _calculate_and_rank(
        self,
        query_result,
        user_lat: float,
        user_lon: float,
        top_k: int,
    ) -> list[dict]:
        """
        Tinh khoang cach Haversine va xep hang theo khoang cach gan nhat.

        Args:
            query_result: QueryResult tu vector store.
            user_lat: Vi do GPS cua nguoi dung.
            user_lon: Kinh do GPS cua nguoi dung.
            top_k: So chi nhanh can tra ve.

        Returns:
            Danh sach top_k dicts, moi dict chua:
                - branch_name (str): Ten chi nhanh.
                - branch_address (str): Dia chi chi nhanh.
                - latitude (float): Vi do chi nhanh.
                - longitude (float): Kinh do chi nhanh.
                - distance_km (float): Khoang cach tu user (km).
                - document (str): Noi dung document goc.

        Raises:
            ToolExecutionError: Khi parse metadata that bai.
        """
        branches_with_distance = []

        for i, doc in enumerate(query_result.documents):
            metadata = (
                query_result.metadatas[i]
                if i < len(query_result.metadatas)
                else {}
            )

            # Parse toa do tu metadata
            # Chu y: field name trong CSV la "lattitude" va "longtitude"
            # (typo tu du lieu goc -- giu nguyen de tuong thich)
            branch_lat = self._parse_coordinate(
                metadata, ["lattitude", "latitude"], "latitude"
            )
            branch_lon = self._parse_coordinate(
                metadata, ["longtitude", "longitude"], "longitude"
            )

            # Bo qua branch khong co toa do hop le
            if branch_lat is None or branch_lon is None:
                logger.debug(
                    f"Skipping branch at index {i}: missing coordinates. "
                    f"metadata={metadata}"
                )
                continue

            # Tinh khoang cach Haversine
            distance = haversine_distance(
                user_lat, user_lon, branch_lat, branch_lon
            )

            branches_with_distance.append({
                "branch_name": metadata.get("branch_name", "N/A"),
                "branch_address": metadata.get("branch_address", "N/A"),
                "latitude": branch_lat,
                "longitude": branch_lon,
                "distance_km": distance,
                "document": doc,
            })

        # Sap xep theo khoang cach tang dan
        branches_with_distance.sort(key=lambda x: x["distance_km"])

        # Tra ve top_k
        return branches_with_distance[:top_k]

    @staticmethod
    def _parse_coordinate(
        metadata: dict,
        field_names: list[str],
        coord_type: str,
    ) -> float | None:
        """
        Parse toa do GPS tu metadata, ho tro nhieu ten field.

        Du lieu goc co typo: "lattitude" va "longtitude",
        nhung method nay cung fallback sang ten dung chinh ta.

        Args:
            metadata: Dict metadata tu vector store.
            field_names: Danh sach ten field can thu (uu tien theo thu tu).
            coord_type: "latitude" hoac "longitude" (dung cho logging).

        Returns:
            float neu parse thanh cong, None neu khong tim thay.
        """
        for field_name in field_names:
            value = metadata.get(field_name)
            if value is not None:
                try:
                    return float(value)
                except (ValueError, TypeError):
                    logger.warning(
                        f"Invalid {coord_type} value: "
                        f"{field_name}={value!r}"
                    )
        return None

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
            location: Ten dia diem goc (neu dung geocode).

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
