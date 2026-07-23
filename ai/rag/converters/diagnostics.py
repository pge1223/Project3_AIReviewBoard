"""
HWP/HWPX Conversion Environment Diagnostics
==================================================
실제 HWP 업로드를 시도하기 전에 "변환이 가능한 상태인가"를 한 번에 점검한다.
백엔드 startup에서 1회 실행해 결과를 캐싱하는 용도로 설계했다 — soffice/unopkg/java
서브프로세스를 요청마다(예: /health 매 호출) 반복 실행하지 않는다.

실행 파일 탐색은 hwp_pdf_converter.find_executable()을 그대로 재사용한다 — 진단과
실제 변환이 서로 다른 탐색 로직(예: 우선순위가 다른 PATH 탐색)을 쓰면 "진단은
통과했는데 실제 변환은 실패"하는 괴리가 생기기 때문이다.

이 모듈의 함수는 예외를 던지지 않는다(각 서브체크가 자체적으로 실패를 흡수한다) —
LibreOffice/Java가 없거나 subprocess가 timeout/인코딩 오류로 죽어도 서버 기동
자체가 실패하면 안 되기 때문이다. 실행 파일의 절대 경로나 subprocess의 전체 출력은
반환값/로그 어디에도 담지 않는다 — 항목별 boolean과 안전한 reason 문자열만 남긴다.
"""

import logging
import re
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from ai.rag.converters.config import HwpConversionConfig
from ai.rag.converters.hwp_pdf_converter import find_executable

logger = logging.getLogger(__name__)

_VERSION_CHECK_TIMEOUT_SECONDS = 15
_CONVERT_PROBE_TIMEOUT_SECONDS = 25
_UNOPKG_TIMEOUT_SECONDS = 20

_UNOPKG_NAMES: tuple[str, ...] = ("unopkg.exe", "unopkg")
_PDF_MAGIC = b"%PDF-"

_REASON_DISABLED = "HWP conversion is disabled"
_REASON_LIBREOFFICE_NOT_FOUND = "LibreOffice(soffice) executable was not found"
_REASON_LIBREOFFICE_CONVERT_FAILED = "LibreOffice(soffice) failed a minimal headless conversion self-test"
_REASON_JAVA_NOT_FOUND = "Java runtime was not found"
_REASON_JAVA_VERSION_FAILED = "Java runtime failed to start (-version check failed)"
_REASON_UNOPKG_NOT_FOUND = "unopkg executable was not found next to LibreOffice"
_REASON_UNOPKG_FAILED = "unopkg list failed to run (see server logs for details)"
_REASON_H2ORESTART_NOT_REGISTERED = "H2Orestart extension is not registered for the backend runtime user"
_REASON_TEMP_DIR_NOT_WRITABLE = "HWP conversion temp directory is not writable"
# 진단 코드 자체에서 예상하지 못한 예외가 났을 때 쓰는 고정 문자열 — 원본 예외 메시지나
# 경로는 여기 담지 않는다(상세는 logger.exception으로 서버 로그에만 남긴다).
_REASON_DIAGNOSTICS_FAILED = "HWP diagnostics failed unexpectedly (see server logs)"


class HwpDiagnosticsResult(BaseModel):
    """/health 및 startup 로그에 그대로 노출해도 안전한, 절대 경로/원본 출력이 없는 진단 결과."""

    enabled: bool
    available: bool
    libreoffice: bool
    h2orestart: bool
    java: bool
    temp_dir_writable: bool
    reason: Optional[str] = None


def _run_quiet(command: list[str], *, timeout: int) -> bool:
    """command가 종료코드 0으로 끝나면 True. timeout/실행불가/인코딩 오류 등 어떤 예외도
    삼키고 False를 반환한다 — 여기서 예외가 새 나가면 startup 전체가 죽을 수 있다."""
    try:
        completed = subprocess.run(
            command,
            shell=False,
            check=False,
            timeout=timeout,
            capture_output=True,
            text=True,
            errors="replace",
        )
        return completed.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _check_soffice_can_convert(soffice_path: str, config: HwpConversionConfig) -> bool:
    """soffice가 실제로 헤드리스 변환을 수행할 수 있는지, 최소 텍스트 파일 하나를 PDF로
    변환해보고 확인한다.

    `soffice --version`/`--help`은 이 프로젝트가 검증한 실제 Windows 설치본에서 문서
    변환과 무관하게 안정적으로 hang되는 것이 확인됐다(2026-07-20, 같은 조건에서
    `--convert-to pdf`는 항상 ~1-3초 내 정상 종료). 즉 `--version` 성공 여부는 실제
    변환 가능 여부와 상관관계가 없어 오탐(available=False인데 실제 업로드는 성공)을
    만든다 — 그래서 실제 변환 코드(hwp_pdf_converter.HwpPdfConverter.convert)와 같은
    `--convert-to pdf` 방식으로 아주 작은 파일을 변환해보는 쪽이 더 정확한 probe다.
    """
    probe_dir = config.resolve_temp_dir() / "hwp_conversion" / "_diagnostics_probe"
    probe_name = uuid.uuid4().hex
    probe_source = probe_dir / f"{probe_name}.txt"
    probe_output = probe_dir / f"{probe_name}.pdf"
    # 가은/Claude(2026-07-22, 용준 협의 — 동시 변환 실패 대응): 진단 프로브도 실제 변환기
    # (hwp_pdf_converter.py)와 동일하게 독립 임시 프로필을 쓴다 — 기본 프로필을 쓰면 서버
    # 기동 진단이 그 순간의 실제 업로드 변환이나 사용자가 열어 둔 LibreOffice와 부딪혀
    # 오탐(available=False)을 만들거나 반대로 실제 변환을 깨뜨릴 수 있다.
    probe_profile_dir = probe_dir / f"{probe_name}_profile"
    try:
        probe_dir.mkdir(parents=True, exist_ok=True)
        probe_profile_dir.mkdir(parents=True, exist_ok=True)
        probe_source.write_text("HWP conversion diagnostics probe", encoding="utf-8")

        completed = subprocess.run(
            [
                soffice_path,
                "--headless",
                "--norestore",
                f"-env:UserInstallation={probe_profile_dir.resolve().as_uri()}",
                "--convert-to",
                "pdf",
                "--outdir",
                str(probe_dir),
                str(probe_source),
            ],
            shell=False,
            check=False,
            timeout=_CONVERT_PROBE_TIMEOUT_SECONDS,
            capture_output=True,
            text=True,
            errors="replace",
        )
        if completed.returncode != 0 or not probe_output.exists():
            return False
        return probe_output.read_bytes().startswith(_PDF_MAGIC)
    except (OSError, subprocess.SubprocessError):
        return False
    finally:
        for path in (probe_source, probe_output):
            try:
                if path.exists():
                    path.unlink()
            except OSError:
                pass
        try:
            if probe_profile_dir.exists():
                shutil.rmtree(probe_profile_dir)
        except OSError:
            pass
        try:
            probe_dir.rmdir()
        except OSError:
            pass  # 비어있지 않거나(동시 실행) 이미 없으면 그냥 둔다 — 치명적이지 않다.


def _find_unopkg(soffice_path: str) -> Optional[str]:
    """탐지된 soffice 실행 파일과 같은 program/ 디렉터리에서 unopkg를 찾는다."""
    program_dir = Path(soffice_path).resolve().parent
    for name in _UNOPKG_NAMES:
        candidate = program_dir / name
        if candidate.exists():
            return str(candidate)
    return None


_IDENTIFIER_LINE_RE = re.compile(r"(?im)^[ \t]*identifier:[ \t]*(.+?)[ \t]*$")
_REGISTERED_LINE_RE = re.compile(r"(?im)^[ \t]*is registered:[ \t]*(yes|no)[ \t]*$")


def _is_h2orestart_registered(unopkg_list_output: str) -> bool:
    """`unopkg list` 출력에서 H2Orestart 확장 "블록"만 골라 그 블록 안의 등록 상태를
    확인한다. 출력 어딘가에 "h2orestart" 문자열과 "is registered: yes"가 각각
    (서로 무관하게) 있기만 해도 통과시키던 기존 방식은, 예를 들어 H2Orestart는
    등록 해제(is registered: no)됐지만 다른 확장이 registered: yes인 경우를 잘못
    통과시킬 수 있었다 — 이 함수는 반드시 같은 "Identifier: ...H2Orestart" 블록
    안에서 등록 상태를 찾는다.

    `unopkg list`는 각 확장을 "Identifier: <id>" 줄로 시작하고, 그 직후(중첩된
    "bundled Packages: {" 하위 항목보다 앞)에 그 확장 자체의 "is registered: yes/no"
    줄이 나온다 — 이 함수는 각 Identifier 블록에서 첫 번째로 나오는 "is registered:"
    줄을 그 확장의 상태로 취급한다.
    """
    normalized = unopkg_list_output.replace("\r\n", "\n").replace("\r", "\n")
    identifier_matches = list(_IDENTIFIER_LINE_RE.finditer(normalized))

    for index, match in enumerate(identifier_matches):
        identifier = match.group(1).strip()
        if "h2orestart" not in identifier.lower():
            continue

        block_start = match.end()
        block_end = (
            identifier_matches[index + 1].start()
            if index + 1 < len(identifier_matches)
            else len(normalized)
        )
        block = normalized[block_start:block_end]

        registered_match = _REGISTERED_LINE_RE.search(block)
        if registered_match is None:
            # 식별자는 있지만 등록 상태를 확인할 수 없음 — 안전하게 미등록으로 취급.
            return False
        return registered_match.group(1).lower() == "yes"

    # H2Orestart 식별자 자체가 출력에 없음.
    return False


def _check_h2orestart(soffice_path: str) -> tuple[bool, Optional[str]]:
    unopkg_path = _find_unopkg(soffice_path)
    if unopkg_path is None:
        return False, _REASON_UNOPKG_NOT_FOUND

    try:
        completed = subprocess.run(
            [unopkg_path, "list"],
            shell=False,
            check=False,
            timeout=_UNOPKG_TIMEOUT_SECONDS,
            capture_output=True,
            text=True,
            errors="replace",
        )
    except (OSError, subprocess.SubprocessError):
        return False, _REASON_UNOPKG_FAILED

    if completed.returncode != 0:
        return False, _REASON_UNOPKG_FAILED

    output = (completed.stdout or "") + (completed.stderr or "")
    if _is_h2orestart_registered(output):
        return True, None
    return False, _REASON_H2ORESTART_NOT_REGISTERED


def _check_temp_dir_writable(config: HwpConversionConfig) -> bool:
    try:
        work_dir = config.resolve_temp_dir() / "hwp_conversion"
        work_dir.mkdir(parents=True, exist_ok=True)
        probe_path = work_dir / ".hwp_diagnostics_write_check"
        probe_path.write_bytes(b"ok")
        probe_path.unlink()
        return True
    except OSError:
        return False


def _diagnostics_failed_result(*, enabled: bool) -> HwpDiagnosticsResult:
    """진단 중 예상하지 못한 예외가 났을 때 반환하는 결과 — 실패를 "정상"으로
    위장하지 않는다. enabled=True로 두면 /health의 status가 "degraded"가 되어
    실제 장애가 드러난다(반대로 enabled=False는 "의도적 비활성화"로 해석되어
    status="ok"가 되므로, 진짜 장애를 이 값으로 감추면 안 된다)."""
    return HwpDiagnosticsResult(
        enabled=enabled,
        available=False,
        libreoffice=False,
        h2orestart=False,
        java=False,
        temp_dir_writable=False,
        reason=_REASON_DIAGNOSTICS_FAILED,
    )


def run_hwp_diagnostics(config: Optional[HwpConversionConfig] = None) -> HwpDiagnosticsResult:
    """HWP 변환에 필요한 환경을 점검한다. 절대 예외를 던지지 않는다.

    fail-safe 정책: HwpConversionConfig 생성 자체가 실패해 "활성화 여부"조차 알 수
    없으면(즉 config.enabled를 확인하기도 전에 예외가 나면) enabled=True로 간주해
    /health가 "degraded"로 표시되게 한다 — 알 수 없는 상태를 "정상(ok)"으로 위장하지
    않기 위함이다. config.enabled를 확인한 뒤(즉 이미 활성화 상태였다는 게 확실한
    뒤)에 예외가 나면 실제 enabled=True 값을 그대로 보존한다.
    """
    try:
        config = config or HwpConversionConfig()
    except Exception:
        logger.exception(
            "[HWP_CONVERTER_DIAGNOSTICS_ERROR] HwpConversionConfig 생성 중 예상하지 못한 오류가 발생했습니다"
        )
        return _diagnostics_failed_result(enabled=True)

    try:
        if not config.enabled:
            return HwpDiagnosticsResult(
                enabled=False,
                available=False,
                libreoffice=False,
                h2orestart=False,
                java=False,
                temp_dir_writable=False,
                reason=_REASON_DISABLED,
            )

        checks: list[tuple[bool, Optional[str]]] = []

        soffice_path = find_executable(config.executable_path)
        libreoffice_found = soffice_path is not None
        libreoffice_ready = False
        if not libreoffice_found:
            checks.append((False, _REASON_LIBREOFFICE_NOT_FOUND))
        else:
            libreoffice_ready = _check_soffice_can_convert(soffice_path, config)
            checks.append((libreoffice_ready, None if libreoffice_ready else _REASON_LIBREOFFICE_CONVERT_FAILED))

        java_path = shutil.which("java")
        java_ready = False
        if java_path is None:
            checks.append((False, _REASON_JAVA_NOT_FOUND))
        else:
            java_ready = _run_quiet([java_path, "-version"], timeout=_VERSION_CHECK_TIMEOUT_SECONDS)
            checks.append((java_ready, None if java_ready else _REASON_JAVA_VERSION_FAILED))

        if libreoffice_found:
            h2orestart_ready, h2orestart_reason = _check_h2orestart(soffice_path)
        else:
            h2orestart_ready, h2orestart_reason = False, _REASON_LIBREOFFICE_NOT_FOUND
        checks.append((h2orestart_ready, h2orestart_reason))

        temp_dir_writable = _check_temp_dir_writable(config)
        checks.append((temp_dir_writable, None if temp_dir_writable else _REASON_TEMP_DIR_NOT_WRITABLE))

        libreoffice_ok = libreoffice_found and libreoffice_ready
        available = libreoffice_ok and java_ready and temp_dir_writable

        reason: Optional[str] = None
        if not available:
            for ok, failure_reason in checks:
                if not ok and failure_reason:
                    reason = failure_reason
                    break

        return HwpDiagnosticsResult(
            enabled=True,
            available=available,
            libreoffice=libreoffice_ok,
            h2orestart=h2orestart_ready,
            java=java_ready,
            temp_dir_writable=temp_dir_writable,
            reason=reason,
        )
    except Exception:
        logger.exception("[HWP_CONVERTER_DIAGNOSTICS_ERROR] 진단 중 예상하지 못한 오류가 발생했습니다")
        # 여기 도달했다는 것 자체가 위의 "if not config.enabled: return"을 통과했다는
        # 뜻이므로(즉 config.enabled는 True로 확인된 뒤), enabled=True를 그대로 보존한다.
        return _diagnostics_failed_result(enabled=True)


def log_hwp_diagnostics(result: HwpDiagnosticsResult) -> None:
    """요구된 로그 포맷으로 진단 결과를 한 번 남긴다 (경로/민감정보 없음)."""
    if not result.enabled:
        logger.info("[HWP_CONVERTER_DISABLED] HWP conversion is disabled by configuration")
    elif result.available:
        logger.info(
            "[HWP_CONVERTER_READY] libreoffice=%s h2orestart=%s java=%s temp_dir_writable=%s",
            str(result.libreoffice).lower(),
            str(result.h2orestart).lower(),
            str(result.java).lower(),
            str(result.temp_dir_writable).lower(),
        )
    else:
        logger.warning("[HWP_CONVERTER_UNAVAILABLE] reason=%s", result.reason)


__all__ = [
    "HwpDiagnosticsResult",
    "run_hwp_diagnostics",
    "log_hwp_diagnostics",
]
