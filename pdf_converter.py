import os
from pathlib import Path
import pymupdf4llm
import pymupdf
import pymupdf.layout


def try_layout_conversion(pdf_path, md_path):
    """
    Advanced fallback using pymupdf.layout for better structure preservation.
    """
    try:
        doc = pymupdf.open(pdf_path)
        markdown_text = ""
        
        for page_num, page in enumerate(doc, 1):
            markdown_text += f"\n## Page {page_num}\n\n"
            
            # Use layout analysis for better text extraction
            blocks = page.get_text("dict", sort=True)["blocks"]
            
            for block in blocks:
                if block["type"] == 0:  # Text block
                    for line in block.get("lines", []):
                        line_text = ""
                        for span in line.get("spans", []):
                            line_text += span.get("text", "")
                        if line_text.strip():
                            markdown_text += line_text + "\n"
                    markdown_text += "\n"
                elif block["type"] == 1:  # Image block
                    markdown_text += "*[Image]*\n\n"
            
            markdown_text += "\n---\n"
        
        doc.close()
        
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(markdown_text)
        
        print(f"✓ Converted (layout): {pdf_path.name}")
        return str(md_path)
    
    except Exception as e:
        print(f"⚠ Layout extraction failed: {e}")
        return try_simple_conversion(pdf_path, md_path)


def try_simple_conversion(pdf_path, md_path):
    """
    Simple fallback using basic text extraction as last resort.
    """
    try:
        doc = pymupdf.open(pdf_path)
        markdown_text = ""
        
        for page_num, page in enumerate(doc, 1):
            markdown_text += f"\n## Page {page_num}\n\n"
            text = page.get_text("text")
            if text.strip():
                markdown_text += text
            else:
                markdown_text += "*[Empty or image-only page]*"
            markdown_text += "\n\n---\n"
        
        doc.close()
        
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(markdown_text)
        
        print(f"✓ Converted (simple): {pdf_path.name}")
        return str(md_path)
    
    except Exception as e:
        print(f"✗ All conversion methods failed for {pdf_path.name}: {e}")
        return None


def convert_pdf_to_markdown(pdf_path, output_dir="data/Markdown", use_layout=True):
    """
    Convert a PDF to Markdown using multiple strategies:
    1. pymupdf4llm (best for complex layouts and tables)
    2. pymupdf.layout (good structure preservation)
    3. Simple text extraction (fallback)
    
    Args:
        pdf_path (str or Path): Path to the PDF file
        output_dir (str): Directory to save the markdown file
        use_layout (bool): Whether to use layout analysis in fallback
    
    Returns:
        str: Path to the generated markdown file, or None if conversion failed
    """
    pdf_path = Path(pdf_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create output filename
    md_filename = pdf_path.stem + ".md"
    md_path = Path(output_dir) / md_filename
    
    # Skip if already converted
    if md_path.exists():
        print(f"⊘ Skipped (exists): {pdf_path.name}")
        return str(md_path)
    
    # Strategy 1: Try pymupdf4llm first (best quality)
    try:
        markdown_text = pymupdf4llm.to_markdown(
            str(pdf_path),
            page_chunks=False,
            write_images=False,
        )
        
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(markdown_text)
        
        print(f"✓ Converted (pymupdf4llm): {pdf_path.name}")
        return str(md_path)
    
    except ValueError as e:
        # Handle the specific "min() iterable is empty" error
        if "min()" in str(e) or "empty" in str(e).lower():
            print(f"⚠ pymupdf4llm failed for {pdf_path.name}: {e}")
            print(f"  Trying layout-aware extraction...")
            if use_layout:
                return try_layout_conversion(pdf_path, md_path)
            else:
                return try_simple_conversion(pdf_path, md_path)
        else:
            print(f"✗ Error converting {pdf_path.name}: {e}")
            return None
    
    except Exception as e:
        print(f"⚠ Unexpected error for {pdf_path.name}: {e}")
        print(f"  Trying fallback conversion...")
        if use_layout:
            return try_layout_conversion(pdf_path, md_path)
        else:
            return try_simple_conversion(pdf_path, md_path)


def convert_pdf_folder(pdf_folder, output_dir="data/Markdown", use_layout=True):
    """
    Convert all PDFs in a folder to markdown, skipping already converted files.
    
    Args:
        pdf_folder (str): Path to folder containing PDFs
        output_dir (str): Directory to save markdown files
        use_layout (bool): Whether to use layout analysis in fallback
    
    Returns:
        tuple: (conversions dict, stats dict)
    """
    pdf_folder = Path(pdf_folder)
    conversions = {}
    stats = {
        "success": 0, 
        "failed": 0, 
        "skipped": 0,
        "methods": {"pymupdf4llm": 0, "layout": 0, "simple": 0}
    }
    
    pdf_files = list(pdf_folder.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files\n")
    
    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"[{i}/{len(pdf_files)}] ", end="")
        md_path = convert_pdf_to_markdown(pdf_path, output_dir, use_layout)
        if md_path:
            conversions[pdf_path.name] = md_path
            stats["success"] += 1
        else:
            stats["failed"] += 1
    
    return conversions, stats


# Example usage
if __name__ == "__main__":
    pdf_folder = "data/PDF_test"
    output_dir = "data/Markdown"
    
    print(f"Converting PDFs from {pdf_folder} to {output_dir}")
    print("=" * 60)
    
    conversions, stats = convert_pdf_folder(pdf_folder, output_dir, use_layout=True)
    
    print("\n" + "=" * 60)
    print(f"Summary:")
    print(f"  ✓ Successfully converted: {stats['success']}")
    print(f"  ✗ Failed: {stats['failed']}")
    print(f"\nOutput directory: {output_dir}")