#!/usr/bin/env python3
"""
PDF Manipulator MCP Server
Provides tools for merging, splitting, extracting text, and converting PDFs
"""

import asyncio
import os
from pathlib import Path
from typing import Any, Sequence, Coroutine
import json
'''
# PDF path
import tempfile
import base64
'''

from mcp.server.models import InitializationOptions
import mcp.types as types
from mcp.server import NotificationOptions, Server
import mcp.server.stdio

# PDF processing imports
import PyPDF2
from mcp.types import TextContent
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import pdfplumber
from pdf2image import convert_from_path
#import fitz  # PyMuPDF

from PIL import Image

# Initialize the MCP server
server = Server("pdf-manipulator")


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List available PDF manipulation tools."""
    return [
        types.Tool(
            name="merge_pdfs",
            description="Merge multiple PDF files into a single PDF",
            inputSchema={
                "type": "object",
                "properties": {
                    "input_files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of PDF file paths to merge"
                    },
                    "output_file": {
                        "type": "string",
                        "description": "Output file path for the merged PDF"
                    }
                },
                "required": ["input_files", "output_file"]
            }
        ),
        types.Tool(
            name="split_pdf",
            description="Split a PDF file into individual pages or page ranges",
            inputSchema={
                "type": "object",
                "properties": {
                    "input_file": {
                        "type": "string",
                        "description": "Path to the PDF file to split"
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "Directory to save split PDF files"
                    },
                    "page_ranges": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional page ranges (e.g., '1-3', '5', '7-10'). If not provided, splits into individual pages"
                    }
                },
                "required": ["input_file", "output_dir"]
            }
        ),
        types.Tool(
            name="extract_text",
            description="Extract text content from a PDF file",
            inputSchema={
                "type": "object",
                "properties": {
                    "input_file": {
                        "type": "string",
                        "description": "Path to the PDF file to extract text from"
                    },
                    "pages": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Optional list of page numbers to extract (1-indexed). If not provided, extracts all pages"
                    },
                    "output_file": {
                        "type": "string",
                        "description": "Optional output file path to save extracted text"
                    }
                },
                "required": ["input_file"]
            }
        ),
        types.Tool(
            name="pdf_to_images",
            description="Convert PDF pages to image files",
            inputSchema={
                "type": "object",
                "properties": {
                    "input_file": {
                        "type": "string",
                        "description": "Path to the PDF file to convert"
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "Directory to save image files"
                    },
                    "image_format": {
                        "type": "string",
                        "enum": ["PNG", "JPEG", "TIFF"],
                        "default": "PNG",
                        "description": "Output image format"
                    },
                    "dpi": {
                        "type": "integer",
                        "default": 200,
                        "description": "DPI for image conversion"
                    }
                },
                "required": ["input_file", "output_dir"]
            }
        ),
        types.Tool(
            name="create_pdf_from_text",
            description="Create a PDF file from text content",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text content to convert to PDF"
                    },
                    "output_file": {
                        "type": "string",
                        "description": "Output PDF file path"
                    },
                    "font_size": {
                        "type": "integer",
                        "default": 12,
                        "description": "Font size for the text"
                    }
                },
                "required": ["text", "output_file"]
            }
        ),
        types.Tool(
            name="get_pdf_info",
            description="Get information about a PDF file (pages, size, metadata)",
            inputSchema={
                "type": "object",
                "properties": {
                    "input_file": {
                        "type": "string",
                        "description": "Path to the PDF file to analyze"
                    }
                },
                "required": ["input_file"]
            }
        )
    ]


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
    """Handle tool calls for PDF manipulation."""

    try:
        if name == "merge_pdfs":
            return await merge_pdfs(arguments)
        elif name == "split_pdf":
            return await split_pdf(arguments)
        elif name == "extract_text":
            return await extract_text(arguments)
        elif name == "pdf_to_images":
            return await pdf_to_images(arguments)
        elif name == "create_pdf_from_text":
            return await create_pdf_from_text(arguments)
        elif name == "get_pdf_info":
            return await get_pdf_info(arguments)
        else:
            raise ValueError(f"Unknown tool: {name}")

    except Exception as e:
        return [types.TextContent(type="text", text=f"Error: {str(e)}")]


async def merge_pdfs(args: dict) -> list[TextContent] | None:
    """Merge multiple PDF files into one."""
    input_files = args["input_files"]
    output_file = args["output_file"]

    # Validate input files
    for file_path in input_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Input file not found: {file_path}")
        if not file_path.lower().endswith('.pdf'):
            raise ValueError(f"Not a PDF file: {file_path}")

    # Create PDF merger
    merger = PyPDF2.PdfMerger()

    try:
        for pdf_file in input_files:
            merger.append(pdf_file)

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Write merged PDF
        with open(output_file, 'wb') as output:
            merger.write(output)
        '''
        return [types.TextContent(
            type="text",
            text=f"Successfully merged {len(input_files)} PDF files into {output_file}"
        )]
        '''
    finally:
        merger.close()

    return [types.TextContent(
        type="text",
        text=f"Successfully merged {len(input_files)} PDF files into {output_file}"
    )]


async def split_pdf(args: dict) -> list[types.TextContent]:
    """Split a PDF file into multiple files."""
    input_file = args["input_file"]
    output_dir = args["output_dir"]
    page_ranges = args.get("page_ranges", [])

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    with open(input_file, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        total_pages = len(pdf_reader.pages)

        if not page_ranges:
            # Split into individual pages
            for page_num in range(total_pages):
                pdf_writer = PyPDF2.PdfWriter()
                pdf_writer.add_page(pdf_reader.pages[page_num])

                output_filename = os.path.join(output_dir, f"page_{page_num + 1}.pdf")
                with open(output_filename, 'wb') as output_file:
                    pdf_writer.write(output_file)

            return [types.TextContent(
                type="text",
                text=f"Split PDF into {total_pages} individual pages in {output_dir}"
            )]
        else:
            # Split by page ranges
            for i, page_range in enumerate(page_ranges):
                pdf_writer = PyPDF2.PdfWriter()

                if '-' in page_range:
                    start, end = map(int, page_range.split('-'))
                    for page_num in range(start - 1, min(end, total_pages)):
                        pdf_writer.add_page(pdf_reader.pages[page_num])
                else:
                    page_num = int(page_range) - 1
                    if 0 <= page_num < total_pages:
                        pdf_writer.add_page(pdf_reader.pages[page_num])

                output_filename = os.path.join(output_dir, f"pages_{page_range}.pdf")
                with open(output_filename, 'wb') as output_file:
                    pdf_writer.write(output_file)

            return [types.TextContent(
                type="text",
                text=f"Split PDF into {len(page_ranges)} files based on specified ranges in {output_dir}"
            )]


async def extract_text(args: dict) -> list[types.TextContent]:
    """Extract text from a PDF file."""
    input_file = args["input_file"]
    pages = args.get("pages", [])
    output_file = args.get("output_file")

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    extracted_text = []

    with pdfplumber.open(input_file) as pdf:
        total_pages = len(pdf.pages)

        if not pages:
            pages = list(range(1, total_pages + 1))

        for page_num in pages:
            if 1 <= page_num <= total_pages:
                page = pdf.pages[page_num - 1]
                text = page.extract_text()
                if text:
                    extracted_text.append(f"--- Page {page_num} ---\n{text}\n")

    full_text = "\n".join(extracted_text)

    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(full_text)

        return [types.TextContent(
            type="text",
            text=f"Extracted text from {len(pages)} pages and saved to {output_file}"
        )]
    else:
        return [types.TextContent(
            type="text",
            text=f"Extracted text from PDF:\n\n{full_text}"
        )]


async def pdf_to_images(args: dict) -> list[types.TextContent]:
    """Convert PDF pages to images."""
    input_file = args["input_file"]
    output_dir = args["output_dir"]
    image_format = args.get("image_format", "PNG")
    dpi = args.get("dpi", 200)

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Convert PDF to images
    images = convert_from_path(input_file, dpi=dpi)

    saved_files = []
    for i, image in enumerate(images):
        output_filename = os.path.join(output_dir, f"page_{i + 1}.{image_format.lower()}")
        image.save(output_filename, image_format)
        saved_files.append(output_filename)

    return [types.TextContent(
        type="text",
        text=f"Converted {len(images)} pages to {image_format} images in {output_dir}"
    )]




async def create_pdf_from_text(args: dict) -> list[types.TextContent]:
    """Create a PDF from text content."""
    text = args["text"]
    output_file = args["output_file"]
    font_size = args.get("font_size", 12)

    # Create output directory if needed
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Create PDF
    c = canvas.Canvas(output_file, pagesize=letter)
    width, height = letter

    # Set up text formatting
    c.setFont("Helvetica", font_size)

    # Split text into lines and pages
    lines = text.split('\n')
    y = height - 72  # Start 1 inch from top
    line_height = font_size + 4

    for line in lines:
        if y < 72:  # If near bottom of page, create new page
            c.showPage()
            c.setFont("Helvetica", font_size)
            y = height - 72

        c.drawString(72, y, line)  # 1 inch from left margin
        y -= line_height

    c.save()

    return [types.TextContent(
        type="text",
        text=f"Created PDF from text content: {output_file}"
    )]


async def get_pdf_info(args: dict) -> list[types.TextContent]:
    """Get information about a PDF file."""
    input_file = args["input_file"]

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    with open(input_file, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)

        info = {
            "file_path": input_file,
            "file_size_mb": round(os.path.getsize(input_file) / (1024 * 1024), 2),
            "total_pages": len(pdf_reader.pages),
            "metadata": {}
        }

        # Get metadata if available
        if pdf_reader.metadata:
            for key, value in pdf_reader.metadata.items():
                if value:
                    info["metadata"][key] = str(value)

    return [types.TextContent(
        type="text",
        text=f"PDF Information:\n{json.dumps(info, indent=2)}"
    )]


async def main():
    # Run the server using stdin/stdout streams
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="pdf-manipulator",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())