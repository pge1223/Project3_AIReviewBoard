"""
Secure Streaming Downloader
===========================
requests.Session 기반으로 URL을 안전하게 가져오는 저수준 유틸리티.

설계 원칙:
- 리다이렉트는 자동으로 따라가지 않고, 매 hop마다 URL 스킴/호스트/IP를 재검증한다 (SSRF 방지).
- HEAD 요청은 참고용 힌트로만 쓰고, 실제 판단은 스트리밍 GET의 실제 응답 헤더와
  바이트 내용(매직 시그니처)으로 확정한다 (서버가 Range/HEAD를 무시하거나 헤더를 속일 수 있음).
- 크기 제한은 Content-Length 사전 검사 + 스트리밍 중 실제 누적 바이트 검사를 이중으로 적용한다.
"""

import ipaddress
import logging
import socket
import zipfile
from pathlib import Path
from urllib.parse import urlparse, urljoin

import requests

from ai.rag.loaders.config import (
    CONNECT_TIMEOUT_SECONDS,
    READ_TIMEOUT_SECONDS,
    MAX_REDIRECTS,
    DOWNLOAD_CHUNK_SIZE_BYTES,
    USER_AGENT,
)
from ai.rag.loaders.exceptions import (
    InvalidUrlError,
    BlockedUrlError,
    UrlFetchError,
    TooManyRedirectsError,
    DownloadSizeLimitExceededError,
)

logger = logging.getLogger(__name__)

_ALLOWED_SCHEMES = ("http", "https")
_REDIRECT_STATUS_CODES = (301, 302, 303, 307, 308)


class FetchedResponse:
    """리다이렉트 검증을 마치고 얻은, 아직 본문을 읽지 않은 스트리밍 응답"""

    def __init__(self, response: requests.Response, final_url: str):
        self.response = response
        self.final_url = final_url

    @property
    def headers(self) -> dict:
        return self.response.headers

    def close(self) -> None:
        self.response.close()


class PeekedStream:
    """본문의 첫 청크를 미리 읽어둔 상태의 스트림 (HTML 여부 판별 등에 사용)"""

    def __init__(self, fetched: FetchedResponse, first_chunk: bytes, chunk_iter):
        self.fetched = fetched
        self.first_chunk = first_chunk
        self._chunk_iter = chunk_iter

    def remaining_chunks(self):
        return self._chunk_iter

    def close(self) -> None:
        self.fetched.close()


def _resolve_all_ips(hostname: str) -> list[str]:
    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror as exc:
        raise BlockedUrlError(f"호스트를 확인할 수 없습니다: {hostname}") from exc
    ips = {info[4][0] for info in infos}
    if not ips:
        raise BlockedUrlError(f"호스트를 확인할 수 없습니다: {hostname}")
    return list(ips)


def _is_blocked_ip(ip_str: str) -> bool:
    ip = ipaddress.ip_address(ip_str.split("%")[0])  # IPv6 zone id(%eth0 등) 제거
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


def validate_url_or_raise(url: str) -> None:
    """
    스킴 화이트리스트 + 호스트 IP(IPv4/IPv6) 검증.
    SSRF 방지 목적이며, 최초 요청뿐 아니라 리다이렉트 각 hop에서도 호출된다.
    """
    parsed = urlparse(url)
    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise InvalidUrlError(f"지원하지 않는 URL 스킴입니다: {parsed.scheme or '(없음)'}")
    if not parsed.hostname:
        raise InvalidUrlError("URL에 호스트가 없습니다.")

    for ip_str in _resolve_all_ips(parsed.hostname):
        if _is_blocked_ip(ip_str):
            raise BlockedUrlError(
                f"내부/사설 네트워크 대상 URL은 허용되지 않습니다: {parsed.hostname} -> {ip_str}"
            )


def probe_head(session: requests.Session, url: str) -> requests.Response | None:
    """
    HEAD 요청 — 참고용 힌트로만 사용한다.
    실패하거나 서버가 HEAD를 지원하지 않으면 None을 반환하며, 호출자는 반드시
    스트리밍 GET(open_stream)으로 실제 판단을 내려야 한다.
    """
    try:
        validate_url_or_raise(url)
        return session.head(
            url,
            timeout=(CONNECT_TIMEOUT_SECONDS, READ_TIMEOUT_SECONDS),
            allow_redirects=False,
            headers={"User-Agent": USER_AGENT},
        )
    except (requests.RequestException, InvalidUrlError, BlockedUrlError):
        return None


def open_stream(
    session: requests.Session,
    url: str,
    *,
    referer: str | None = None,
    max_redirects: int = MAX_REDIRECTS,
) -> FetchedResponse:
    """
    리다이렉트를 자동으로 따르지 않고(allow_redirects=False), Location 헤더를 직접 읽어
    한 단계씩 처리한다. 매 hop마다 URL/IP를 재검증한다.
    """
    current_url = url
    headers = {"User-Agent": USER_AGENT}
    if referer:
        headers["Referer"] = referer

    for _ in range(max_redirects + 1):
        validate_url_or_raise(current_url)

        try:
            response = session.get(
                current_url,
                timeout=(CONNECT_TIMEOUT_SECONDS, READ_TIMEOUT_SECONDS),
                allow_redirects=False,
                stream=True,
                headers=headers,
            )
        except requests.Timeout as exc:
            logger.warning("[open_stream] 요청 시간 초과: url=%s cause=%s: %s", current_url, type(exc).__name__, exc)
            raise UrlFetchError(f"요청 시간이 초과되었습니다: {current_url}") from exc
        except requests.RequestException as exc:
            logger.warning("[open_stream] 요청 실패: url=%s cause=%s: %s", current_url, type(exc).__name__, exc)
            raise UrlFetchError(f"요청에 실패했습니다: {current_url}") from exc

        if response.status_code in _REDIRECT_STATUS_CODES:
            location = response.headers.get("Location")
            response.close()
            if not location:
                raise UrlFetchError(f"리다이렉트 응답에 Location 헤더가 없습니다: {current_url}")
            current_url = urljoin(current_url, location)
            continue

        return FetchedResponse(response=response, final_url=current_url)

    raise TooManyRedirectsError(f"리다이렉트 허용 횟수({max_redirects})를 초과했습니다: {url}")


def _check_content_length_header(fetched: FetchedResponse, max_size_bytes: int) -> None:
    content_length = fetched.headers.get("Content-Length")
    if content_length is None:
        return
    try:
        declared_size = int(content_length)
    except ValueError:
        return  # 헤더 값이 숫자가 아니면 무시하고 스트리밍 누적 검사로 대체
    if declared_size > max_size_bytes:
        fetched.close()
        raise DownloadSizeLimitExceededError(
            f"Content-Length({declared_size} bytes)가 제한({max_size_bytes} bytes)을 초과합니다: {fetched.final_url}"
        )


def peek_stream(fetched: FetchedResponse, max_size_bytes: int, peek_size: int = 4096) -> PeekedStream:
    """
    Content-Length를 먼저 검사한 뒤, 본문의 첫 청크를 미리 읽어 반환한다.
    (HTML 여부를 실제 바이트로 판별하기 위함 — 서버가 Content-Type을 생략/오기하는 경우 대비)
    """
    _check_content_length_header(fetched, max_size_bytes)
    chunk_iter = fetched.response.iter_content(chunk_size=max(DOWNLOAD_CHUNK_SIZE_BYTES, peek_size))
    first_chunk = next(chunk_iter, b"")
    return PeekedStream(fetched, first_chunk, chunk_iter)


def consume_peeked_as_text(peeked: PeekedStream, max_size_bytes: int) -> tuple[str, str]:
    """PeekedStream을 텍스트로 소비한다 (HTML 페이지 본문용). Returns: (text, encoding)"""
    buf = bytearray(peeked.first_chunk)
    try:
        for chunk in peeked.remaining_chunks():
            buf.extend(chunk)
            if len(buf) > max_size_bytes:
                raise DownloadSizeLimitExceededError(
                    f"다운로드 크기가 제한({max_size_bytes} bytes)을 초과하여 중단했습니다: {peeked.fetched.final_url}"
                )
        if len(buf) > max_size_bytes:
            raise DownloadSizeLimitExceededError(
                f"다운로드 크기가 제한({max_size_bytes} bytes)을 초과하여 중단했습니다: {peeked.fetched.final_url}"
            )
    finally:
        peeked.close()

    encoding = peeked.fetched.response.encoding or peeked.fetched.response.apparent_encoding or "utf-8"
    text = bytes(buf).decode(encoding, errors="replace")
    return text, encoding


def consume_peeked_to_file(peeked: PeekedStream, dest_path: Path, max_size_bytes: int) -> int:
    """
    PeekedStream을 파일로 저장한다 (첨부파일용). 제한 초과 시 부분 파일을 삭제하고 예외를 던진다.
    Returns: 저장된 바이트 수
    """
    total = 0
    try:
        with open(dest_path, "wb") as f:
            if peeked.first_chunk:
                total += len(peeked.first_chunk)
                if total > max_size_bytes:
                    raise DownloadSizeLimitExceededError(
                        f"다운로드 크기가 제한({max_size_bytes} bytes)을 초과하여 중단했습니다: {peeked.fetched.final_url}"
                    )
                f.write(peeked.first_chunk)

            for chunk in peeked.remaining_chunks():
                total += len(chunk)
                if total > max_size_bytes:
                    raise DownloadSizeLimitExceededError(
                        f"다운로드 크기가 제한({max_size_bytes} bytes)을 초과하여 중단했습니다: {peeked.fetched.final_url}"
                    )
                f.write(chunk)
    except DownloadSizeLimitExceededError:
        dest_path.unlink(missing_ok=True)
        raise
    finally:
        peeked.close()

    return total


def guess_filename_from_response(fetched: FetchedResponse, fallback: str = "download") -> str:
    """Content-Disposition 헤더 우선, 없으면 URL 경로에서 파일명을 유추한다."""
    content_disposition = fetched.headers.get("Content-Disposition", "")
    if content_disposition:
        # filename*=UTF-8''... 또는 filename="..." 패턴 모두 시도
        for part in content_disposition.split(";"):
            part = part.strip()
            if part.lower().startswith("filename*="):
                value = part.split("=", 1)[1].strip()
                if "''" in value:
                    value = value.split("''", 1)[1]
                name = requests.utils.unquote(value.strip('"'))
                if name:
                    return _sanitize_filename(name)
            elif part.lower().startswith("filename="):
                name = part.split("=", 1)[1].strip().strip('"')
                if name:
                    return _sanitize_filename(name)

    path_name = Path(urlparse(fetched.final_url).path).name
    return _sanitize_filename(path_name) if path_name else fallback


def _sanitize_filename(name: str) -> str:
    """경로 조작 문자 제거 (../, 절대경로, null byte 등) — 임시 디렉토리 밖으로 나가지 못하게 함"""
    name = name.replace("\x00", "")
    name = Path(name).name  # 디렉토리 구분자 및 .. 제거
    return name or "download"


_CFBF_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"  # OLE 복합 문서 (구버전 HWP/DOC/XLS 등)


_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def sniff_file_signature(path: Path) -> str:
    """
    매직바이트로 실제 파일 형식을 판별한다 (확장자/Content-Type을 신뢰하지 않음).
    Returns: "pdf" | "docx" | "pptx" | "hwpx" | "hwp_legacy" | "jpeg" | "png" | "zip_unknown" | "unknown"
    """
    with open(path, "rb") as f:
        header = f.read(8)

    if header.startswith(b"%PDF-"):
        return "pdf"

    # 가은/Claude(2026-07-18): 공고 포스터가 이미지 한 장으로 올라오는 경우(url_loader.py의
    # 이미지->PDF 변환+OCR 재사용 경로)를 위해 추가.
    if header.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if header == _PNG_MAGIC:
        return "png"

    if header == _CFBF_MAGIC:
        # 확장자 없는 다운로드 링크가 실제로는 구버전(.hwp) 문서인 경우를 식별하기 위함
        return "hwp_legacy"

    if header.startswith(b"PK\x03\x04") or header.startswith(b"PK\x05\x06"):
        try:
            with zipfile.ZipFile(path) as zf:
                names = set(zf.namelist())
        except zipfile.BadZipFile:
            return "unknown"
        if "word/document.xml" in names:
            return "docx"
        if "ppt/presentation.xml" in names:
            return "pptx"
        if any(name.startswith("Contents/") for name in names):
            # HWPX도 ZIP 컨테이너 구조를 사용함
            return "hwpx"
        return "zip_unknown"

    return "unknown"
