# pdf_processor.py - FIXED VERSION
import re
import logging
from pathlib import Path
from typing import List, Dict, Any
import fitz  # PyMuPDF

class PDFProcessor:
    def __init__(self):
        self.logger = logging.getLogger("pdf_processor")

    def extract_text_with_structure(self, pdf_path: str) -> Dict[str, Any]:
        """Extract text and identify meaningful content sections, not just any heading"""
        self.logger.info("Processing %s", pdf_path)
        doc_data = {
            "file_name": Path(pdf_path).name,
            "pages": [],
            "full_text": "",
            "sections": [],
        }
        
        all_text = ""
        all_pages_data = []
        
        # First pass: collect all text
        with fitz.open(pdf_path) as doc:
            for page_num, page in enumerate(doc, 1):
                raw_text = page.get_text("text", sort=True)
                if raw_text.strip():
                    cleaned = self._clean_text(raw_text)
                    all_text += cleaned + "\n\n"
                    all_pages_data.append({
                        "page_number": page_num,
                        "text": cleaned
                    })
        
        doc_data["full_text"] = all_text
        doc_data["pages"] = all_pages_data
        
        # Second pass: identify meaningful sections across the entire document
        doc_data["sections"] = self._identify_meaningful_sections(all_text, all_pages_data)
        
        return doc_data

    def _clean_text(self, text: str) -> str:
        """Clean text while preserving structure"""
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)  # Collapse spaces but keep newlines
        text = re.sub(r"\n{3,}", "\n\n", text)  # Max 2 consecutive newlines
        text = re.sub(r"-\n([a-z])", r"\1", text, flags=re.IGNORECASE)  # Fix hyphenation
        return text.strip()

    def _identify_meaningful_sections(self, full_text: str, pages_data: List[Dict]) -> List[Dict[str, Any]]:
        """
        Identify substantial, meaningful content sections rather than just headings.
        Focus on content blocks that would be valuable for travel planning.
        """
        sections = []
        
        # Split text into large content blocks based on major topic changes
        # Look for substantial headings that indicate major content sections
        major_section_patterns = [
            r"^([A-Z][^.!?]*(?:Guide|Adventures?|Activities|Experiences?|Tips|Tricks|Entertainment|Nightlife|Cuisine|Food|Cities|History|Culture|Hotels?|Restaurants?)[^.!?]*)$",
            r"^([A-Z][^.!?]{20,100})$",  # Long descriptive titles
            r"^((?:[A-Z][a-z]+ ){2,}[A-Z][a-z]+)$",  # Multi-word titles
        ]
        
        paragraphs = re.split(r'\n\s*\n', full_text)
        current_section = None
        content_buffer = []
        section_id = 1
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            # Check if this paragraph starts a new major section
            lines = para.split('\n')
            first_line = lines[0].strip()
            
            is_major_heading = any(re.match(pattern, first_line, re.IGNORECASE) 
                                 for pattern in major_section_patterns)
            
            if is_major_heading and len(first_line) > 10 and len(first_line) < 150:
                # Save previous section if it has substantial content
                if current_section and len(' '.join(content_buffer).split()) > 50:
                    current_section['content'] = ' '.join(content_buffer).strip()
                    current_section['word_count'] = len(current_section['content'].split())
                    sections.append(current_section)
                
                # Start new section
                current_section = {
                    'section_id': f'section_{section_id}',
                    'title': first_line,
                    'content': '',
                    'page_number': self._find_page_number(first_line, pages_data),
                    'section_type': 'content',
                    'word_count': 0
                }
                section_id += 1
                content_buffer = []
                
                # Add remaining content from this paragraph
                if len(lines) > 1:
                    content_buffer.append('\n'.join(lines[1:]))
            else:
                # Add to current section content
                if current_section:
                    content_buffer.append(para)
        
        # Don't forget the last section
        if current_section and len(' '.join(content_buffer).split()) > 50:
            current_section['content'] = ' '.join(content_buffer).strip()
            current_section['word_count'] = len(current_section['content'].split())
            sections.append(current_section)
        
        # If we didn't find enough major sections, fall back to paragraph-based sectioning
        if len(sections) < 3:
            sections = self._fallback_paragraph_sections(full_text, pages_data)
        
        return sections

    def _find_page_number(self, heading: str, pages_data: List[Dict]) -> int:
        """Find which page a heading appears on"""
        for page in pages_data:
            if heading in page['text']:
                return page['page_number']
        return 1  # Default to page 1 if not found

    def _fallback_paragraph_sections(self, full_text: str, pages_data: List[Dict]) -> List[Dict[str, Any]]:
        """Fallback method: create sections from substantial paragraphs"""
        sections = []
        paragraphs = re.split(r'\n\s*\n', full_text)
        section_id = 1
        
        for para in paragraphs:
            para = para.strip()
            words = para.split()
            
            # Only create sections from substantial paragraphs
            if len(words) > 100:  # Substantial content
                # Use first sentence or first 50 chars as title
                first_sentence = para.split('.')[0][:50] + "..."
                
                sections.append({
                    'section_id': f'section_{section_id}',
                    'title': first_sentence,
                    'content': para,
                    'page_number': self._find_page_number(para[:50], pages_data),
                    'section_type': 'content',
                    'word_count': len(words)
                })
                section_id += 1
                
                if len(sections) >= 15:  # Limit number of sections
                    break
        
        return sections

    def process_document_collection(self, pdf_paths):
        """Process multiple PDFs"""
        documents = []
        for path in pdf_paths:
            try:
                doc_data = self.extract_text_with_structure(path)
                if doc_data["full_text"].strip():
                    documents.append(doc_data)
                else:
                    self.logger.warning("No text extracted from %s", path)
            except Exception as exc:
                self.logger.error("Failed to process %s: %s", path, exc)
        return documents