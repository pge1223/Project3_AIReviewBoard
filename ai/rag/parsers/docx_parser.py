"""
DOCX Parser using python-docx
=============================
"""

from pathlib import Path

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P

from ai.rag.parsers.base_parser import BaseParser
from ai.rag.parsers.schemas import (
    FileType,
    LocationType,
    BlockType,
    DocumentBlock,
    DocumentExtractionResult,
)
from ai.rag.parsers.exceptions import CorruptedDocumentError, EmptyDocumentError


class DOCXParser(BaseParser):
    """python-docx 기반 DOCX 파서"""

    def get_file_type(self) -> FileType:
        return FileType.DOCX

    def get_page_count(self) -> int | None:
        """LibreOffice로 PDF 변환 후 페이지 수 반환"""
        try:
            import tempfile
            import fitz  # pymupdf
            from ai.rag.converters.preview_pdf_converter import convert_to_preview_pdf

            with tempfile.TemporaryDirectory() as tmp_dir:
                pdf_path = convert_to_preview_pdf(
                    self.file_path,
                    output_dir=Path(tmp_dir),
                )
                doc = fitz.open(str(pdf_path))
                page_count = len(doc)
                doc.close()
                return page_count
        except Exception:
            return 1  # 변환 실패 시 폴백

    def _iter_block_items(self, parent) -> list:
        """
        문서 내 블록(문단, 표)을 순서대로 순회

        python-docx의 iter_inner_content()를 사용하여
        문단과 표가 실제 문서에서 등장하는 순서를 유지
        """
        items = []
        for child in parent.element.body:
            if isinstance(child, CT_P):
                items.append(("paragraph", Paragraph(child, parent)))
            elif isinstance(child, CT_Tbl):
                items.append(("table", Table(child, parent)))
        return items

    def _classify_paragraph(self, para: Paragraph) -> BlockType:
        """문단 유형 분류"""
        style_name = para.style.name.lower() if para.style else ""

        # 제목 스타일 감지
        if any(keyword in style_name for keyword in ["heading", "title", "head"]):
            return BlockType.TITLE

        # 목록 스타일 감지
        if any(keyword in style_name for keyword in ["list", "bullet", "number"]):
            return BlockType.LIST

        # 빈 문단
        if not para.text.strip():
            return BlockType.TEXT  # 빈 텍스트로 처리

        return BlockType.TEXT

    def _table_to_text(self, table: Table) -> str:
        """표를 텍스트로 변환 (탭 구분)"""
        rows = []
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            rows.append("\t".join(cells))
        return "\n".join(rows)

    def _paragraph_runs(self, para: Paragraph) -> list[dict]:
        """문단을 run 단위로 쪼개 굵게/기울임 서식을 유지한다. content(순수 텍스트)는
        RAG 청킹·임베딩이 그대로 쓰므로 안 건드리고, 이 정보는 metadata["runs"]에만
        추가로 담는다 - 원문을 워드처럼 보여주는 화면(재인/Claude, 2026-07-21)에서만 씀."""
        runs = []
        for run in para.runs:
            if not run.text:
                continue
            runs.append({
                "text": run.text,
                "bold": bool(run.bold),
                "italic": bool(run.italic),
            })
        return runs

    def _is_list_item(self, para: Paragraph) -> bool:
        """목록 항목인지 확인"""
        if not para.text.strip():
            return False
        # 번호 매기기 목록 또는 불릿 목록 확인
        num_pr = para._element.find(
            ".//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}numPr"
        )
        return num_pr is not None

    def parse(self) -> DocumentExtractionResult:
        """DOCX 문서 파싱"""
        file_size = self.file_path.stat().st_size
        warnings: list[str] = []
        blocks: list[DocumentBlock] = []
        order = 0

        try:
            doc = Document(str(self.file_path))
        except Exception as e:
            raise CorruptedDocumentError(f"DOCX 파일을 열 수 없습니다: {e}")

        document_id = self.generate_document_id(self.file_path)

        # 문서 내 모든 블록을 순서대로 순회
        for item_type, item in self._iter_block_items(doc):

            if item_type == "paragraph":
                para = item
                text = para.text.strip()

                if not text:
                    # 빈 문단은 건너뜀
                    continue

                # 목록 여부 확인
                block_type = BlockType.TEXT
                if self._is_list_item(para):
                    block_type = BlockType.LIST
                else:
                    block_type = self._classify_paragraph(para)

                block_obj = DocumentBlock(
                    block_id=self.generate_block_id(
                        document_id,
                        LocationType.DOCUMENT,
                        None,
                        order,
                    ),
                    block_type=block_type,
                    content=text,
                    location_type=LocationType.DOCUMENT,
                    location_number=None,
                    order=order,
                    metadata={
                        "style": para.style.name if para.style else None,
                        "runs": self._paragraph_runs(para),
                    },
                )
                blocks.append(block_obj)
                order += 1

            elif item_type == "table":
                table = item
                table_text = self._table_to_text(table)

                if table_text.strip():
                    block_obj = DocumentBlock(
                        block_id=self.generate_block_id(
                            document_id,
                            LocationType.DOCUMENT,
                            None,
                            order,
                        ),
                        block_type=BlockType.TABLE,
                        content=table_text,
                        location_type=LocationType.DOCUMENT,
                        location_number=None,
                        order=order,
                        metadata={
                            "rows": len(table.rows),
                            "columns": len(table.columns) if table.rows else 0,
                        },
                    )
                    blocks.append(block_obj)
                    order += 1

        if len(blocks) == 0:
            raise EmptyDocumentError(
                "DOCX에서 텍스트를 추출할 수 없습니다. 빈 문서이거나 손상된 파일일 수 있습니다."
            )

        return self.create_result(
            file_size=file_size,
            page_count=self.get_page_count(),
            blocks=blocks,
            warnings=warnings,
        )
