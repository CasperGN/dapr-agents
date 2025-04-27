from dapr_agents.document.reader.base import ReaderBase
from dapr_agents.types.document import Document
from typing import List, Dict, Optional
from pathlib import Path
import pymupdf


class PyMuPDFReader(ReaderBase):
    """
    Reader for PDF documents using PyMuPDF.
    """

    def load(
        self, file_path: Path, additional_metadata: Optional[Dict] = None
    ) -> List[Document]:
        """
        Load content from a PDF file using PyMuPDF.

        Args:
            file_path (Path): Path to the PDF file.
            additional_metadata (Optional[Dict]): Additional metadata to include.

        Returns:
            List[Document]: A list of Document objects.
        """

        file_path = Path(str(file_path))
        doc = pymupdf.open(file_path)
        total_pages = len(doc)
        documents = []

        for page_num, page in enumerate(doc.pages):  # type: ignore[arg-type,var-annotated]
            text = page.get_text()  # enumerate expects Iterable[Never] but pages return an enumerator for _pages_ret
            metadata = {  # type check complains of 'page' not being annotated. 'page' is of type pymudpdf.Page
                "file_path": file_path,
                "page_number": page_num + 1,
                "total_pages": total_pages,
            }
            if additional_metadata:
                metadata.update(additional_metadata)

            documents.append(Document(text=text.strip(), metadata=metadata))

        doc.close()
        return documents
