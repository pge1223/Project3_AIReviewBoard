# 작성자: 재인/Claude (2026-07-21)
# 목적: "AI 피드백" 워크벤치가 기획서를 워드/한글 원본과 똑같은 페이지 모습으로 보여주기
#   위해 추가. hwp_pdf_converter.py(HWP/HWPX 전용, 시그니처 검증 포함)와는 별개 파일 -
#   그 파일의 SUPPORTED_EXTENSIONS/검증 로직을 건드리면 기존 RAG 색인 파이프라인(HWP만
#   PDF로 변환해서 파싱하는 흐름)에 영향을 줄 수 있어 전혀 손대지 않는다. 여기서는
#   docx/hwp/hwpx/pptx 등 LibreOffice가 열 수 있는 형식을 "미리보기용 PDF"로만 변환한다
#   (색인/청킹과 무관, 순수 화면 표시용). LibreOffice CLI 호출 방식은 hwp_pdf_converter.py와
#   동일하되, HWP 전용 시그니처 검증은 하지 않는다(여기선 이미 업로드 시점에 검증된
#   파일을 다시 열 뿐이라 중복 검증이 필요 없음).
import logging
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Optional

from ai.rag.converters.config import HwpConversionConfig
from ai.rag.converters.exceptions import (
    ConversionProcessError,
    ConversionTimeoutError,
    ConverterUnavailableError,
    ConvertedFileNotFoundError,
    InvalidConvertedPdfError,
    InvalidSourceFileError,
)
from ai.rag.converters.hwp_pdf_converter import SOFFICE_DEFAULT_PROFILE_LOCK, find_executable

logger = logging.getLogger(__name__)

_PDF_MAGIC: bytes = b"%PDF-"


def _looks_like_pdf(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(len(_PDF_MAGIC)) == _PDF_MAGIC
    except OSError:
        return False


def convert_to_preview_pdf(
    source_path: Path,
    *,
    output_dir: Path,
    config: Optional[HwpConversionConfig] = None,
) -> Path:
    """source_path(docx/hwp/hwpx/pptx 등)를 미리보기용 PDF로 변환해 경로를 반환한다.
    실패하면 hwp_pdf_converter.py와 같은 예외 타입을 던진다(호출부가 이미 그 예외들을
    처리하는 패턴에 맞추기 위함). 변환된 파일은 호출부가 정리해야 한다(원본은 안 지움)."""
    effective_config = config or HwpConversionConfig()
    source_path = Path(source_path)

    if not source_path.exists():
        raise InvalidSourceFileError(
            f"원본 파일을 찾을 수 없습니다: {source_path.name}",
            user_message="문서 파일을 찾을 수 없습니다.",
        )

    executable = find_executable(effective_config.executable_path)
    if executable is None:
        raise ConverterUnavailableError(
            "미리보기 변환 도구(LibreOffice)를 찾을 수 없습니다. "
            "HWP_CONVERTER_EXECUTABLE 환경변수 또는 PATH를 확인하세요."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    # 원본 파일명을 신뢰하지 않고(동시 요청 충돌·경로 조작 방지) 무작위 이름으로 복사 후 변환.
    safe_stem = uuid.uuid4().hex
    staged_source = output_dir / f"{safe_stem}{source_path.suffix}"
    shutil.copyfile(source_path, staged_source)

    # 재인/Claude(2026-07-21) 실측 발견: LibreOffice headless는 기본적으로 프로필(작업공간)을
    # 하나만 쓰기 때문에, 두 변환 요청이 동시에 들어오면 나중 요청이 그 프로필을 이미 쓰고
    # 있는 걸 보고 종료 코드 1로 실패한다(워크벤치 진입 시 React 18 StrictMode가 effect를
    # 두 번 실행해서 실제로 재현됨). -env:UserInstallation로 요청마다 완전히 독립된 임시
    # 프로필을 주면 동시 실행이 서로 안 부딪힌다.
    #
    # 가은/Claude(2026-07-22, 용준 협의 — 동시 변환 실패 대응, 재인님 확인 필요): 단,
    # HWP/HWPX만은 격리 프로필을 쓰면 안 된다는 것이 실측으로 확인됐다 — HWP 필터
    # (H2Orestart 확장)가 사용자 기본 프로필 안에만 설치돼 있어서, 빈 격리 프로필로 HWP를
    # 변환하면 확장이 없어 soffice가 0xC0000409로 크래시한다(즉 HWP 미리보기는 이 격리가
    # 들어간 뒤로 항상 실패하고 있었다). 그래서 HWP/HWPX는 기본 프로필 +
    # SOFFICE_DEFAULT_PROFILE_LOCK(hwp_pdf_converter.py와 공유, 기본 프로필 동시 사용
    # 직렬화)으로 변환하고, docx/pptx 등 확장이 필요 없는 형식은 기존 격리 방식을 그대로
    # 유지한다(StrictMode 동시 요청 병렬 처리도 그대로 유지됨).
    is_hwp_family = source_path.suffix.lower() in {".hwp", ".hwpx"}
    profile_dir: Optional[Path] = None

    command = [executable, "--headless", "--norestore"]
    if not is_hwp_family:
        profile_dir = output_dir / f"{safe_stem}_profile"
        profile_dir.mkdir(parents=True, exist_ok=True)
        command.append(f"-env:UserInstallation={profile_dir.resolve().as_uri()}")
    command += ["--convert-to", "pdf", "--outdir", str(output_dir), str(staged_source)]

    logger.info(
        "[PREVIEW_PDF_CONVERSION_START] original_ext=%s 라이브레오피스 headless 서브프로세스로 "
        "PDF 변환 요청: %s",
        source_path.suffix.lstrip("."), " ".join(command),
    )
    start = time.monotonic()
    try:
        if is_hwp_family:
            with SOFFICE_DEFAULT_PROFILE_LOCK:
                completed = subprocess.run(
                    command, shell=False, check=False,
                    timeout=effective_config.timeout_seconds, capture_output=True, text=True,
                )
        else:
            completed = subprocess.run(
                command, shell=False, check=False,
                timeout=effective_config.timeout_seconds, capture_output=True, text=True,
            )
    except subprocess.TimeoutExpired as exc:
        _cleanup(staged_source)
        _cleanup_dir(profile_dir)
        raise ConversionTimeoutError(
            f"미리보기 변환이 timeout_seconds={effective_config.timeout_seconds}초를 초과했습니다."
        ) from exc

    duration_ms = int((time.monotonic() - start) * 1000)
    converted_path = output_dir / f"{safe_stem}.pdf"

    logger.info(
        "[PREVIEW_PDF_CONVERSION_RESPONSE] 라이브레오피스 응답(종료 코드=%d, %dms) stdout=%s",
        completed.returncode, duration_ms, (completed.stdout or "").strip()[:300],
    )

    if completed.returncode != 0:
        _cleanup(staged_source, converted_path)
        _cleanup_dir(profile_dir)
        raise ConversionProcessError(
            f"미리보기 변환 프로세스가 종료 코드 {completed.returncode}로 실패했습니다: "
            f"{(completed.stderr or '')[:300]}"
        )

    _cleanup(staged_source)
    _cleanup_dir(profile_dir)

    if not converted_path.exists() or converted_path.stat().st_size == 0:
        raise ConvertedFileNotFoundError("미리보기 PDF가 생성되지 않았거나 비어 있습니다.")
    if not _looks_like_pdf(converted_path):
        _cleanup(converted_path)
        raise InvalidConvertedPdfError("변환된 미리보기 파일이 유효한 PDF가 아닙니다.")

    logger.info(
        "[PREVIEW_PDF_CONVERSION_COMPLETE] original_ext=%s duration_ms=%d 결과 PDF=%s (%dKB) - "
        "프론트가 이 파일을 pdf.js로 그대로 그림",
        source_path.suffix.lstrip("."), duration_ms, converted_path.name,
        converted_path.stat().st_size // 1024,
    )
    return converted_path


def _cleanup(*paths: Path) -> None:
    for path in paths:
        try:
            if path.exists():
                path.unlink()
        except OSError as exc:
            logger.warning("[PREVIEW_PDF_CLEANUP_FAILED] path_name=%s error=%s", path.name, exc)


def _cleanup_dir(path: Optional[Path]) -> None:
    # HWP/HWPX 경로는 격리 프로필을 만들지 않으므로(None) 정리할 것도 없다.
    if path is None:
        return
    try:
        if path.exists():
            shutil.rmtree(path)
    except OSError as exc:
        logger.warning("[PREVIEW_PDF_CLEANUP_FAILED] path_name=%s error=%s", path.name, exc)
