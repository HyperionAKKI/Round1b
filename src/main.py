# main.py - COMPLETE FIXED VERSION with proper subsection analysis
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

from pdf_processor import PDFProcessor
from persona_analyzer import PersonaAnalyzer
from utils.config_loader import ConfigLoader

class DocumentIntelligenceSystem:
    def __init__(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        )
        self.logger = logging.getLogger(__name__)
        self.pdf_processor = PDFProcessor()
        self.persona_analyzer = PersonaAnalyzer()

    def run(self, input_dir: str | os.PathLike, output_dir: str | os.PathLike) -> None:
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1️⃣ Load & normalize config
        cfg_path = input_dir / "config.json"
        cfg = ConfigLoader(cfg_path).load()
        self.logger.info(
            "Parsed config – persona=%s | job=%s | %d docs",
            cfg["persona"],
            cfg["job_to_be_done"][:60] + ("…" if len(cfg["job_to_be_done"]) > 60 else ""),
            len(cfg["documents"]),
        )

        # 2️⃣ Resolve PDF paths
        pdf_paths = [input_dir / name for name in cfg["documents"]]
        valid_paths = [p for p in pdf_paths if p.exists()]
        if not valid_paths:
            raise FileNotFoundError("None of the PDF files listed in config were found.")
        self.logger.info("Found %d/%d PDFs", len(valid_paths), len(pdf_paths))

        # 3️⃣ Extract text & structure with improved section detection
        documents = self.pdf_processor.process_document_collection([str(p) for p in valid_paths])
        if not documents:
            raise RuntimeError("Extraction produced no text – aborting.")

        # Log section extraction results
        total_sections = sum(len(doc.get('sections', [])) for doc in documents)
        self.logger.info(f"Extracted {total_sections} sections from {len(documents)} documents")

        # 4️⃣ Persona relevance analysis with travel-specific scoring
        ranked_sections = self.persona_analyzer.analyze_persona_relevance(
            documents, cfg["persona"], cfg["job_to_be_done"]
        )

        if not ranked_sections:
            self.logger.warning("No sections found after persona analysis")
            ranked_sections = []

        # 5️⃣ Generate output with proper subsection analysis
        out_data = self._build_output(documents, cfg, ranked_sections)
        out_file = output_dir / "challenge1b_output.json"
      #  out_file.write_text(json.dumps(out_data, indent=2, ensure_ascii=False))
        out_file.write_text(
            json.dumps(out_data, indent=2, ensure_ascii=False),
            encoding='utf-8'
        )

   

        
        self.logger.info("✅ Results written to %s", out_file)
        self.logger.info(f"📊 Total sections analyzed: {len(ranked_sections)}")
        self.logger.info(f"📋 Top sections extracted: {len(out_data.get('extracted_sections', []))}")
        self.logger.info(f"🔍 Subsection analysis items: {len(out_data.get('subsection_analysis', []))}")

    def _build_output(
        self,
        docs: List[Dict[str, Any]],
        cfg: Dict[str, Any],
        ranked: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build output matching the exact required format"""
        
        # Get top 10 sections for extraction
        top_sections = ranked[:10] if ranked else []
        
        # Build extracted_sections array
        extracted_sections = []
        for section in top_sections:
            extracted_sections.append({
                "document": section["document"],
                "section_title": section["section_title"],
                "importance_rank": section["importance_rank"],
                "page_number": section["page_number"]
                # Note: removed relevance_score to match required format exactly
            })
        
        # Generate subsection analysis from top 5 sections
        subsection_analysis = []
        if ranked:
            subsection_analysis = self.persona_analyzer.generate_subsection_analysis(ranked, max_sections=5)
        
        # Build metadata
        metadata = {
            "input_documents": [d["file_name"] for d in docs],
            "persona": cfg["persona"],
            "job_to_be_done": cfg["job_to_be_done"],
            "processing_timestamp": datetime.now().isoformat(timespec="seconds").replace('+00:00', '')
        }

        return {
            "metadata": metadata,
            "extracted_sections": extracted_sections,
            "subsection_analysis": subsection_analysis
        }

def main() -> None:
    """CLI wrapper"""
    system = DocumentIntelligenceSystem()
    system.run("/app/input", "/app/output")

if __name__ == "__main__":
    main()